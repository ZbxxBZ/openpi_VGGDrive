"""Configuration for the PyTorch pi0 / pi0.5 Cross-View Geometric Enabler."""

import dataclasses
from typing import Literal

Backbone = Literal["vggt", "vggt_omega"]

# Released aggregators: VGGT (CVPR 2025) uses DINOv2 14px patches and 1 camera + 4
# register tokens; VGGT-Omega (CVPR 2026) uses DINOv3 16px patches and 1 camera + 16
# register tokens. Both output 2048-dim (frame ‖ global) tokens per view.
# Default square resolutions: 224 is VGGT's LIBERO setting; 416 matches the
# VGGT-Omega-1B-416 reproduction weights and keeps the geometry-token count moderate.
# Override `image_size` to 512 or 256 for the corresponding Omega checkpoints.
_BACKBONES: dict[str, dict] = {
    "vggt": {"patch_size": 14, "image_size": 224, "package": "vggt", "siblings": ("VGGDrive", "vggt")},
    "vggt_omega": {"patch_size": 16, "image_size": 416, "package": "vggt_omega", "siblings": ("vggt-omega",)},
}


@dataclasses.dataclass(frozen=True)
class GeometryConfig:
    enabled: bool = False
    backbone: Backbone = "vggt"
    # Directory containing the `vggt` or `vggt_omega` package. None searches the sibling
    # checkouts listed in `sibling_dirs` next to the openpi repo, then the installed package.
    vggt_source_path: str | None = None
    # Required for initialization, but not when restoring a complete CVGE checkpoint.
    vggt_weights_path: str | None = None
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    # Square VGGT input side. None selects the backbone default (224 VGGT, 416 VGGT-Omega).
    image_size: int | None = None
    feature_dim: int = 2048
    fusion_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    use_camera_pose: bool = False
    train_policy: Literal["adapter_only", "adapter_action", "full"] = "adapter_action"

    def __post_init__(self):
        if self.backbone not in _BACKBONES:
            raise ValueError(f"Unknown geometry.backbone: {self.backbone}")
        if self.image_size is None:
            # Frozen dataclass: resolve the backbone default once so checkpoints store a number.
            object.__setattr__(self, "image_size", _BACKBONES[self.backbone]["image_size"])
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("geometry.image_keys must contain unique camera names")
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError(
                f"geometry.image_size must be a positive multiple of {self.backbone}'s patch size ({self.patch_size})"
            )
        if self.feature_dim != 2048:
            raise ValueError("The supported VGGT and VGGT-Omega aggregators produce 2048-dimensional features")
        if self.num_heads <= 0 or self.fusion_dim <= 0 or self.fusion_dim % self.num_heads:
            raise ValueError("geometry.fusion_dim must be positive and divisible by geometry.num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("geometry.dropout must be in [0, 1)")
        if self.train_policy not in ("adapter_only", "adapter_action", "full"):
            raise ValueError(f"Unknown geometry.train_policy: {self.train_policy}")

    @property
    def patch_size(self) -> int:
        return _BACKBONES[self.backbone]["patch_size"]

    @property
    def package(self) -> str:
        """Importable package name of the backbone (`vggt` or `vggt_omega`)."""
        return _BACKBONES[self.backbone]["package"]

    @property
    def sibling_dirs(self) -> tuple[str, ...]:
        """Checkout directory names searched next to the openpi repo when no source path is set."""
        return _BACKBONES[self.backbone]["siblings"]

    @property
    def patch_tokens_per_view(self) -> int:
        return (self.image_size // self.patch_size) ** 2
