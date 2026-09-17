"""pi0.5 data/config regression cases for later server validation, without downloads."""

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.geometry_config import GeometryConfig
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as training_config


@pytest.mark.parametrize("backbone", ["vggt", "vggt_omega"])
@pytest.mark.parametrize("discrete_state_input", [False, True])
def test_pi05_state_reaches_tokenizer_and_preserves_camera_data(monkeypatch, backbone, discrete_state_input):
    calls = []

    class RecordingTokenizer:
        def __init__(self, max_length):
            self.max_length = max_length

        def tokenize(self, prompt, state=None):
            calls.append((prompt, state))
            tokens = np.zeros(self.max_length, dtype=np.int32)
            tokens[0] = 1
            if state is not None:
                tokens[1] = 2
            return tokens, tokens != 0

    monkeypatch.setattr(training_config._tokenizer, "PaligemmaTokenizer", RecordingTokenizer)  # noqa: SLF001
    config = Pi0Config(
        pi05=True,
        discrete_state_input=discrete_state_input,
        geometry=GeometryConfig(enabled=True, backbone=backbone),
    )
    assert config.model_type == model_lib.ModelType.PI05
    assert config.max_token_len == 200
    pipeline = transforms.compose(training_config.ModelTransformFactory()(config).inputs)
    state = np.array([0.1, -0.4, 0.7], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    image = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    result = pipeline(
        {
            "prompt": "pick up the cube",
            "state": state.copy(),
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": True},
            "camera_to_world": {"base_0_rgb": pose},
        }
    )
    assert calls[0][0] == "pick up the cube"
    if discrete_state_input:
        # Tokenization sees the normalized state before padding, like original pi0.5.
        np.testing.assert_array_equal(calls[0][1], state)
        assert result["tokenized_prompt"][1] == 2
    else:
        assert calls[0][1] is None
        assert result["tokenized_prompt"][1] == 0
    assert result["state"].shape == (config.action_dim,)
    np.testing.assert_array_equal(result["state"][:3], state)
    np.testing.assert_array_equal(result["camera_to_world"]["base_0_rgb"], pose)
    assert result["image_mask"]["base_0_rgb"]
    assert result["tokenized_prompt"].shape == (200,)
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["geometry_image"]["base_0_rgb"].shape == image.shape
    np.testing.assert_array_equal(result["geometry_image"]["base_0_rgb"], image)


def test_observation_normalizes_policy_and_geometry_images_independently():
    policy_image = torch.full((1, 224, 224, 3), 255, dtype=torch.uint8)
    geometry_image = torch.zeros((1, 360, 640, 3), dtype=torch.uint8)
    observation = model_lib.Observation.from_dict(
        {
            "image": {"base_0_rgb": policy_image},
            "geometry_image": {"base_0_rgb": geometry_image},
            "image_mask": {"base_0_rgb": torch.ones(1, dtype=torch.bool)},
            "state": torch.zeros(1, 32),
        }
    )

    assert observation.images["base_0_rgb"].shape == (1, 3, 224, 224)
    assert observation.geometry_images["base_0_rgb"].shape == (1, 3, 360, 640)
    assert torch.all(observation.images["base_0_rgb"] == 1)
    assert torch.all(observation.geometry_images["base_0_rgb"] == -1)
    restored = observation.to_dict()
    assert "geometry_image" in restored
    assert "geometry_images" not in restored


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("backbone", ["vggt", "vggt_omega"])
def test_cvge_libero_presets_preserve_original_data_semantics(pi05, backbone):
    family = "pi05" if pi05 else "pi0"
    suffix = "_omega" if backbone == "vggt_omega" else ""
    original = training_config.get_config(f"{family}_libero")
    enhanced = training_config.get_config(f"{family}_cvge{suffix}_libero")
    for name in ("model_type", "action_dim", "action_horizon", "max_token_len", "discrete_state_input"):
        assert getattr(enhanced.model, name) == getattr(original.model, name)
    assert enhanced.data.extra_delta_transform == original.data.extra_delta_transform
    assert enhanced.data.repo_id == original.data.repo_id
    assert enhanced.model.geometry.enabled
    assert enhanced.model.geometry.backbone == backbone
    assert enhanced.model.pytorch_compile_mode is None
    # Generic pi0.5 still defaults to including discrete state; LIBERO opts out explicitly.
    assert Pi0Config(pi05=True, geometry=enhanced.model.geometry).discrete_state_input
