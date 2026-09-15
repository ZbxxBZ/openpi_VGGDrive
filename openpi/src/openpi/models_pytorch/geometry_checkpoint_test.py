import dataclasses
import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models.geometry_config import GeometryConfig
from openpi.models_pytorch.geometry_checkpoint import load_pi0_weights
from openpi.models_pytorch.geometry_checkpoint import save_geometry_metadata


class _SmallPolicy(nn.Module):
    def __init__(self, geometry, *, pi05=False, discrete_state_input=False):
        super().__init__()
        self.config = SimpleNamespace(
            geometry=geometry,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            action_dim=6,
            action_horizon=3,
            max_token_len=4,
            pi05=pi05,
            discrete_state_input=discrete_state_input,
        )
        self.base = nn.Linear(4, 4)
        if geometry.enabled:
            self.paligemma_with_expert = nn.Module()
            self.paligemma_with_expert.cvge = nn.ModuleList([nn.Linear(4, 4)])
            self.vggt_encoder = nn.Linear(4, 4)


@pytest.mark.parametrize("pi05", [False, True])
def test_base_initialization_only_allows_new_geometry_keys(tmp_path, pi05):
    original = _SmallPolicy(GeometryConfig(), pi05=pi05)
    target = _SmallPolicy(GeometryConfig(enabled=True), pi05=pi05)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(original, path)
    geometry_before = target.vggt_encoder.weight.detach().clone()
    with pytest.raises(ValueError, match="complete CVGE"):
        load_pi0_weights(target, path)
    load_pi0_weights(target, path, allow_base=True)
    torch.testing.assert_close(target.base.weight, original.base.weight)
    torch.testing.assert_close(target.vggt_encoder.weight, geometry_before)
    broken = original.state_dict()
    del broken["base.bias"]
    safetensors.torch.save_file(broken, path)
    with pytest.raises(RuntimeError, match="missing base keys"):
        load_pi0_weights(target, path, allow_base=True)


def test_complete_checkpoint_roundtrip_and_configuration_validation(tmp_path):
    config = GeometryConfig(enabled=True, vggt_weights_path="/old/machine/model.pt")
    model = _SmallPolicy(config)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(model, path)
    save_geometry_metadata(model, tmp_path)
    restored = _SmallPolicy(dataclasses.replace(config, vggt_weights_path=None, vggt_source_path="/new/source"))
    load_pi0_weights(restored, path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    wrong_resolution = _SmallPolicy(dataclasses.replace(config, image_size=518))
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_pi0_weights(wrong_resolution, path)
    next_stage = _SmallPolicy(dataclasses.replace(config, train_policy="full"))
    load_pi0_weights(next_stage, path)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_pi0_weights(next_stage, path, resume=True)
    with pytest.raises(ValueError, match="contains CVGE"):
        load_pi0_weights(_SmallPolicy(GeometryConfig()), path)


def test_partial_geometry_checkpoint_cannot_be_treated_as_base(tmp_path):
    model = _SmallPolicy(GeometryConfig(enabled=True))
    state = model.state_dict()
    del state["paligemma_with_expert.cvge.0.bias"]
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(state, path)
    save_geometry_metadata(model, tmp_path)
    with pytest.raises(RuntimeError):
        load_pi0_weights(model, path, allow_base=True)
    (tmp_path / "geometry_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="geometry_config"):
        load_pi0_weights(model, path, allow_base=True)


@pytest.mark.parametrize("backbone", ["vggt", "vggt_omega"])
def test_geometry_checkpoint_backbone_is_verified(tmp_path, backbone):
    config = GeometryConfig(enabled=True, backbone=backbone)
    model = _SmallPolicy(config)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(model, path)
    save_geometry_metadata(model, tmp_path)
    load_pi0_weights(_SmallPolicy(config), path)
    other = "vggt" if backbone == "vggt_omega" else "vggt_omega"
    # Fresh config: each backbone resolves its own default resolution and patch size.
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_pi0_weights(_SmallPolicy(GeometryConfig(enabled=True, backbone=other)), path)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_pi0_geometry_checkpoint_remains_vggt(tmp_path, version):
    model = _SmallPolicy(GeometryConfig(enabled=True))
    path = tmp_path / "model.safetensors"
    metadata_path = tmp_path / "geometry_config.json"
    safetensors.torch.save_model(model, path)
    save_geometry_metadata(model, tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = version
    metadata["model"].pop("discrete_state_input")
    if version == 1:
        metadata["geometry"].pop("backbone")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    load_pi0_weights(_SmallPolicy(model.config.geometry), path, resume=True)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_pi0_weights(_SmallPolicy(GeometryConfig(enabled=True, backbone="vggt_omega")), path)


@pytest.mark.parametrize("backbone", ["vggt", "vggt_omega"])
@pytest.mark.parametrize("discrete_state_input", [False, True])
def test_pi05_checkpoint_preserves_model_and_state_semantics(tmp_path, backbone, discrete_state_input):
    geometry = GeometryConfig(enabled=True, backbone=backbone)
    model = _SmallPolicy(geometry, pi05=True, discrete_state_input=discrete_state_input)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(model, path)
    save_geometry_metadata(model, tmp_path)
    restored = _SmallPolicy(geometry, pi05=True, discrete_state_input=discrete_state_input)
    load_pi0_weights(restored, path, resume=True)
    torch.testing.assert_close(restored.base.weight, model.base.weight)
    # Identical tensor shapes cannot reveal these two semantic mismatches.
    for wrong in (
        _SmallPolicy(geometry, pi05=False, discrete_state_input=discrete_state_input),
        _SmallPolicy(geometry, pi05=True, discrete_state_input=not discrete_state_input),
    ):
        with pytest.raises(ValueError, match="configuration mismatch"):
            load_pi0_weights(wrong, path)


def test_legacy_pi05_metadata_requires_explicit_state_setting(tmp_path):
    model = _SmallPolicy(GeometryConfig(enabled=True), pi05=True, discrete_state_input=True)
    path = tmp_path / "model.safetensors"
    metadata_path = tmp_path / "geometry_config.json"
    safetensors.torch.save_model(model, path)
    save_geometry_metadata(model, tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = 2
    metadata["model"].pop("discrete_state_input")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="missing discrete_state_input"):
        load_pi0_weights(model, path)
    metadata["model"]["discrete_state_input"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    load_pi0_weights(model, path, resume=True)
