"""Omega interface regression cases, prepared for execution on the server.

The small stand-in follows the official aggregator's sparse output list, 16px
patches and 17 special tokens. It does not substitute for testing real weights.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models.geometry_config import GeometryConfig
from openpi.models_pytorch import vggt_encoder
from openpi.models_pytorch.cvge import CVGE


class _OmegaAggregator(nn.Module):
    patch_size = 16
    patch_token_start = 17
    depth = 24

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.cached_layer_indices = {4, 11, 17, 23}
        self.rope_embed = SimpleNamespace(normalize_coords="max")
        self.patch_embed = nn.Module()
        self.patch_embed.rope_embed = SimpleNamespace(normalize_coords="max")
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        batch, views, _, height, width = images.shape
        count = height // self.patch_size * (width // self.patch_size) + self.patch_token_start
        shared = images.mean(dim=(1, 2, 3, 4))[:, None, None, None] * self.weight
        positions = torch.arange(count, device=images.device)[None, None, :, None]
        tokens = (shared + positions).expand(batch, views, count, 2048)
        return [
            tokens if index in self.cached_layer_indices else None for index in range(self.depth)
        ], self.patch_token_start


def _config(**kwargs):
    return GeometryConfig(enabled=True, backbone="vggt_omega", **kwargs)


@pytest.mark.parametrize("resolution", [256, 416, 512])
def test_omega_resolution_special_tokens_and_cvge(resolution):
    config = _config(image_size=resolution, image_keys=("base_0_rgb",), fusion_dim=16, num_heads=4, dropout=0.0)
    encoder = vggt_encoder.VGGTEncoder(config, load_weights=False, aggregator=_OmegaAggregator())
    context = encoder([torch.zeros(1, 3, 28, 28)], [torch.ones(1, dtype=torch.bool)])
    assert context.tokens.shape == (1, 1, (resolution // 16) ** 2 + 17, 2048)
    assert not context.tokens.requires_grad
    assert context.tokens[0, 0, 0, 0] == 0.5  # camera
    assert context.tokens[0, 0, 16, 0] == 16.5  # final register
    assert context.tokens[0, 0, 17, 0] == 17.5  # first patch
    adapter = CVGE(24, config).eval()
    nn.init.normal_(adapter.output_proj[-1].weight, std=0.02)
    prefix = torch.randn(1, 5, 24)
    output = adapter(prefix, context, torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, 3:], prefix[:, 3:], atol=0, rtol=0)
    assert not torch.allclose(output[:, :3], prefix[:, :3])


def test_omega_excludes_invalid_views_and_handles_empty_sample():
    config = _config(image_size=32)
    aggregator = _OmegaAggregator()
    encoder = vggt_encoder.VGGTEncoder(config, load_weights=False, aggregator=aggregator)
    images = [torch.full((3, 3, 32, 32), value) for value in (-0.6, 0.0, 0.8)]
    masks = [torch.tensor([True, True, False]), torch.tensor([False, True, False]), torch.tensor([True, False, False])]
    for image, mask in zip(images, masks, strict=True):
        image[~mask] = float("nan")
    context = encoder(images, masks)
    assert context.tokens.shape == (3, 3, 21, 2048)
    assert len(aggregator.calls) == 2
    assert all(torch.isfinite(call).all() and call.shape[1] == 2 for call in aggregator.calls)
    torch.testing.assert_close(context.tokens[0, 0, 0, 0], torch.tensor(0.55))
    torch.testing.assert_close(context.tokens[1, 0, 0, 0], torch.tensor(0.35))
    assert torch.count_nonzero(context.tokens[2]) == 0


def test_omega_missing_final_cache_and_wrong_backbone_fail():
    aggregator = _OmegaAggregator()
    config = _config(image_size=32, image_keys=("base_0_rgb",))
    encoder = vggt_encoder.VGGTEncoder(config, load_weights=False, aggregator=aggregator)
    # Injected aggregators are configured like imported ones: only the final layer is cached.
    assert aggregator.cached_layer_indices == {23}
    aggregator.cached_layer_indices = {4, 11, 17}
    with pytest.raises(ValueError, match="final layer"):
        encoder([torch.zeros(1, 3, 32, 32)], [torch.ones(1, dtype=torch.bool)])
    with pytest.raises(ValueError, match="patch size 14"):
        vggt_encoder.VGGTEncoder(GeometryConfig(), load_weights=False, aggregator=_OmegaAggregator())


@pytest.mark.parametrize("format_name", ["full_pt", "aggregator_pt", "wrapped_pt", "ddp_pt", "safetensors"])
def test_omega_checkpoint_formats_and_strict_loading(tmp_path, format_name):
    source = _OmegaAggregator()
    with torch.no_grad():
        source.weight.fill_(2)
    weights = {"aggregator." + key: value for key, value in source.state_dict().items()}
    weights["dense_head.unused"] = torch.zeros(1)
    if format_name == "aggregator_pt":
        weights = source.state_dict()
    elif format_name == "wrapped_pt":
        weights = {"model": weights, "epoch": 12}
    elif format_name == "ddp_pt":
        weights = {"state_dict": {"module." + key: value for key, value in weights.items()}}
    path = tmp_path / ("model.safetensors" if format_name == "safetensors" else "model.pt")
    if format_name == "safetensors":
        safetensors.torch.save_file(weights, path)
    else:
        torch.save(weights, path)
    encoder = vggt_encoder.VGGTEncoder(_config(vggt_weights_path=str(path)), aggregator=_OmegaAggregator())
    torch.testing.assert_close(encoder.aggregator.weight, source.weight)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert not encoder.aggregator.training
    broken_path = tmp_path / "broken.pt"
    torch.save({"aggregator.unexpected": torch.zeros(1)}, broken_path)
    with pytest.raises(RuntimeError):
        vggt_encoder.VGGTEncoder(_config(vggt_weights_path=str(broken_path)), aggregator=_OmegaAggregator())


def test_omega_source_import_namespace_final_cache_and_rope(tmp_path, monkeypatch):
    source_file = tmp_path / "vggt_omega" / "models" / "aggregator.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# import marker\n", encoding="utf-8")
    aggregator = _OmegaAggregator()
    imported = []

    def import_module(name):
        imported.append(name)
        return SimpleNamespace(__file__=str(source_file), Aggregator=lambda: aggregator)

    monkeypatch.setattr(vggt_encoder.importlib, "import_module", import_module)
    monkeypatch.setattr(vggt_encoder.sys, "path", vggt_encoder.sys.path.copy())
    config = _config(vggt_source_path=str(tmp_path))
    encoder = vggt_encoder.VGGTEncoder(config, load_weights=False)
    assert imported == ["vggt_omega.models.aggregator"]
    assert encoder.aggregator.cached_layer_indices == {23}
    aggregator.patch_embed.rope_embed.normalize_coords = "separate"
    with pytest.warns(UserWarning, match="normalize_coords"):
        vggt_encoder.VGGTEncoder(config, load_weights=False)
    with pytest.raises(FileNotFoundError, match=r"vggt/models/aggregator\.py"):
        vggt_encoder.VGGTEncoder(dataclasses.replace(config, backbone="vggt", image_size=None), load_weights=False)


def test_backbone_specific_patch_size_validation():
    assert GeometryConfig().patch_size == 14
    assert _config().patch_size == 16
    # None resolves to the backbone default and is stored as a number for checkpoint metadata.
    assert GeometryConfig().image_size == 224
    assert _config().image_size == 416
    assert _config(image_size=512).image_size == 512
    assert dataclasses.asdict(_config())["image_size"] == 416
    assert GeometryConfig().sibling_dirs == ("VGGDrive", "vggt")
    assert _config().package == "vggt_omega"
    assert _config().patch_tokens_per_view == 26 * 26
    with pytest.raises(ValueError, match="patch size \\(16\\)"):
        _config(image_size=518)
    with pytest.raises(ValueError, match="patch size \\(14\\)"):
        GeometryConfig(image_size=416)
