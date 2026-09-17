"""Frozen VGGT / VGGT-Omega aggregators with missing-camera-safe batching.

Both backbones share the aggregator contract used here:

    aggregator(images[B, V, 3, H, W] in [0, 1]) -> (per_layer_outputs, patch_token_start)

The final list entry has shape [B, V, patch_token_start + patches, 2048] (frame | global
features). VGGT (`vggt.models.aggregator`, 14px patches, 1 camera + 4 register tokens)
returns every layer; VGGT-Omega (`vggt_omega.models.aggregator`, 16px patches, 1 camera +
16 register tokens) returns None for layers outside `cached_layer_indices`. The first view
of every call is the reference frame for both models, so cameras keep `image_keys` order.
"""

from collections.abc import Mapping
import importlib
import pathlib
import sys
import warnings

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models.geometry_config import GeometryConfig
from openpi.models_pytorch.cvge import GeometryContext

# openpi/src/openpi/models_pytorch/vggt_encoder.py -> workspace containing openpi/ and the checkouts.
_WORKSPACE = pathlib.Path(__file__).resolve().parents[4]


def _resolve_source(config: GeometryConfig) -> pathlib.Path | None:
    """Directory to prepend to sys.path, or None to rely on an installed package."""
    relative = pathlib.Path(config.package) / "models" / "aggregator.py"
    if config.vggt_source_path is not None:
        source = pathlib.Path(config.vggt_source_path).expanduser().resolve()
        if not (source / relative).is_file():
            raise FileNotFoundError(f"vggt_source_path must contain {relative.as_posix()}: {source}")
        return source
    for sibling in config.sibling_dirs:
        candidate = _WORKSPACE / sibling
        if (candidate / relative).is_file():
            return candidate
    return None


def _import_aggregator_module(config: GeometryConfig):
    source = _resolve_source(config)
    # Both projects use absolute package imports. Keep their distinct namespaces and
    # reject an already-imported copy that does not come from the requested checkout.
    if source is not None and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    name = f"{config.package}.models.aggregator"
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        candidates = [str(_WORKSPACE / sibling) for sibling in config.sibling_dirs]
        raise ImportError(
            f"Cannot import {name}. Install the {config.backbone} package, place a checkout at one of "
            f"{candidates}, or set geometry.vggt_source_path"
        ) from exc
    module_file = getattr(module, "__file__", None)
    if source is not None and module_file and not pathlib.Path(module_file).resolve().is_relative_to(source):
        raise ImportError(f"Another {config.package} is already imported from {module_file}; requested {source}")
    return module


def _configure_aggregator(aggregator: nn.Module, config: GeometryConfig) -> None:
    """Apply backbone-specific settings to a constructed or injected aggregator."""
    if config.backbone != "vggt_omega":
        return
    # CVGE consumes only the final layer, so Omega need not retain its three additional
    # dense-head feature maps. All backbone blocks still execute.
    depth = getattr(aggregator, "depth", None)
    if isinstance(depth, int) and hasattr(aggregator, "cached_layer_indices"):
        aggregator.cached_layer_indices = {depth - 1}
    # The released checkpoints were trained with RoPE normalize_coords="max"; upstream
    # VGGTOmega warns on a mismatch, so do the same instead of refusing to run.
    components = (("aggregator", aggregator), ("aggregator.patch_embed", getattr(aggregator, "patch_embed", None)))
    for name, component in components:
        rope = getattr(component, "rope_embed", None)
        normalize = getattr(rope, "normalize_coords", None)
        if normalize is not None and normalize != "max":
            warnings.warn(
                f"VGGT-Omega {name} RoPE normalize_coords is {normalize!r}; the released weights use 'max'",
                stacklevel=2,
            )


def _make_aggregator(config: GeometryConfig) -> nn.Module:
    module = _import_aggregator_module(config)
    # Defaults reproduce the released 1B models: VGGT (DINOv2-L/14, 4 registers) and
    # VGGT-Omega (DINOv3-L/16, 16 registers, register-attention inter-frame blocks).
    return module.Aggregator()


def _load_aggregator_weights(aggregator: nn.Module, path: pathlib.Path) -> None:
    if path.suffix == ".safetensors":
        state = safetensors.torch.load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected a VGGT/VGGT-Omega state dict in {path}")
    for wrapper in ("state_dict", "model"):
        if isinstance(state.get(wrapper), Mapping):
            state = state[wrapper]
            break
    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError(f"Empty or invalid VGGT/VGGT-Omega state dict in {path}")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    # Official full-model weights (VGGT `model.pt`, `vggt_omega_1b_*.pt`) include prediction
    # heads. Only the aggregator is used here; aggregator-only exports are also accepted.
    # Never relax strict loading of the backbone itself, including Omega inter_frame_blocks.
    if any(key.startswith("aggregator.") for key in state):
        state = {
            key.removeprefix("aggregator."): value for key, value in state.items() if key.startswith("aggregator.")
        }
    aggregator.load_state_dict(state, strict=True)


class VGGTEncoder(nn.Module):
    def __init__(self, config: GeometryConfig, *, load_weights: bool = True, aggregator: nn.Module | None = None):
        super().__init__()
        self.config = config
        path = None
        if load_weights:
            if config.vggt_weights_path is None:
                raise ValueError(
                    "Set geometry.vggt_weights_path when initializing CVGE from an original pi0/pi0.5 checkpoint"
                )
            path = pathlib.Path(config.vggt_weights_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"VGGT weights not found: {path}")
        self.aggregator = _make_aggregator(config) if aggregator is None else aggregator
        _configure_aggregator(self.aggregator, config)
        self.patch_size = getattr(self.aggregator, "patch_size", None)
        # VGGT names the first patch index `patch_start_idx`; VGGT-Omega uses `patch_token_start`.
        self.patch_token_start = next(
            (
                getattr(self.aggregator, attribute)
                for attribute in ("patch_token_start", "patch_start_idx")
                if isinstance(getattr(self.aggregator, attribute, None), int)
            ),
            None,
        )
        if self.patch_size != config.patch_size:
            raise ValueError(f"{config.backbone} requires patch size {config.patch_size}, got {self.patch_size}")
        if not isinstance(self.patch_token_start, int) or self.patch_token_start < 1:
            raise ValueError(f"{config.backbone} aggregator must expose a positive patch_token_start/patch_start_idx")
        if path is not None:
            _load_aggregator_weights(self.aggregator, path)
        self.requires_grad_(requires_grad=False)
        self.train(mode=False)

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002 -- nn.Module API
        # Parent model.train() must never activate VGGT stochastic training layers.
        return super().train(mode=False)

    @property
    def tokens_per_view(self) -> int:
        return self.config.patch_tokens_per_view + self.patch_token_start

    def _prepare_view(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("VGGT expects batched RGB images")
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError("VGGT expects three RGB channels")
        # Input is the unaugmented camera RGB preserved by the model transform, in [-1, 1].
        # Both aggregators take [0, 1] and apply their own ImageNet normalization internally.
        image = (image.float() * 0.5 + 0.5).clamp(0, 1)
        size = (self.config.image_size, self.config.image_size)
        if image.shape[-2:] != size:
            # Preserve the raw camera aspect ratio. A direct square resize changes
            # ray-to-pixel geometry; deterministic letterboxing only adds a fixed
            # scale and offset and is shared by every sample from a camera stream.
            omega = self.config.backbone == "vggt_omega"
            height, width = image.shape[-2:]
            scale = min(size[0] / height, size[1] / width)
            resized_height = max(1, int(height * scale))
            resized_width = max(1, int(width * scale))
            image = F.interpolate(
                image,
                size=(resized_height, resized_width),
                mode="bicubic" if omega else "bilinear",
                align_corners=False,
                antialias=omega,
            )
            image = image.clamp(0, 1)
            pad_height = size[0] - resized_height
            pad_width = size[1] - resized_width
            image = F.pad(
                image,
                (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2),
                value=0.0,
            )
        return image

    @torch.no_grad()
    def forward(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        camera_to_world: torch.Tensor | None = None,
    ) -> GeometryContext:
        if len(images) != len(self.config.image_keys) or len(image_masks) != len(images):
            raise ValueError("VGGT images and masks must follow geometry.image_keys")
        stacked = torch.stack([self._prepare_view(image) for image in images], dim=1).contiguous()
        valid = torch.stack(image_masks, dim=1).to(device=stacked.device, dtype=torch.bool)
        batch, views = stacked.shape[:2]
        if valid.shape != (batch, views):
            raise ValueError("Each image mask must have shape [B]")
        if self.config.use_camera_pose:
            if camera_to_world is None or camera_to_world.shape != (batch, views, 4, 4):
                raise ValueError("Camera-conditioned CVGE requires a 4x4 camera_to_world pose for every camera slot")
            if not torch.isfinite(camera_to_world[valid]).all():
                raise ValueError("Valid camera poses must be finite")

        token_count = self.tokens_per_view
        use_autocast = stacked.device.type == "cuda"
        dtype = torch.float32
        if use_autocast:
            with torch.cuda.device(stacked.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        features = torch.zeros(batch, views, token_count, self.config.feature_dim, device=stacked.device, dtype=dtype)
        # Group samples with the same valid-camera pattern. Missing images never enter the
        # aggregator's cross-view attention, including padded LIBERO wrist cameras. Within a
        # group the first valid camera (in image_keys order) is the reference frame.
        for pattern in valid.unique(dim=0):
            camera_indices = pattern.nonzero(as_tuple=True)[0]
            if camera_indices.numel() == 0:
                continue
            batch_indices = (valid == pattern).all(dim=1).nonzero(as_tuple=True)[0]
            group = stacked.index_select(0, batch_indices).index_select(1, camera_indices).contiguous()
            with torch.autocast(device_type=stacked.device.type, dtype=dtype, enabled=use_autocast):
                outputs, patch_start_idx = self.aggregator(group)
            # Omega returns None for uncached intermediate layers. Do not select an
            # earlier cached layer if the required final feature is missing.
            if not outputs or outputs[-1] is None:
                raise ValueError(f"{self.config.backbone} aggregator did not cache its final layer")
            last = outputs[-1].detach()
            expected = (len(batch_indices), len(camera_indices), token_count, self.config.feature_dim)
            if patch_start_idx != self.patch_token_start or last.shape != expected:
                raise ValueError(
                    f"Unexpected {self.config.backbone} feature shape: {tuple(last.shape)}; expected {expected}"
                )
            features[batch_indices[:, None], camera_indices[None, :]] = last.to(dtype)
            del outputs, last
        return GeometryContext(tokens=features, valid_views=valid, camera_to_world=camera_to_world)
