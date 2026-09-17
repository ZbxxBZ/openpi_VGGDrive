"""CVGE invariants; run on the server with the OpenPI PyTorch environment."""

import dataclasses

import pytest
import torch
from torch import nn

from openpi.models.geometry_config import GeometryConfig
from openpi.models_pytorch.cvge import CVGE
from openpi.models_pytorch.cvge import GeometryContext
from openpi.models_pytorch.vggt_encoder import VGGTEncoder


def _config(**kwargs):
    return GeometryConfig(enabled=True, fusion_dim=16, num_heads=4, dropout=0.0, **kwargs)


def _activate(adapter):
    nn.init.normal_(adapter.output_proj[-1].weight, std=0.02)


def test_zero_initialization_and_visual_only_updates():
    torch.manual_seed(7)
    adapter = CVGE(24, _config()).eval()
    prefix = torch.randn(2, 9, 24)
    visual_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    geometry = GeometryContext(torch.randn(2, 3, 6, 2048), torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(adapter(prefix, geometry, visual_mask), prefix, rtol=0, atol=0)
    _activate(adapter)
    enhanced = adapter(prefix, geometry, visual_mask)
    torch.testing.assert_close(enhanced[:, 4:], prefix[:, 4:], rtol=0, atol=0)
    torch.testing.assert_close(enhanced[0, 2:4], prefix[0, 2:4], rtol=0, atol=0)
    assert not torch.allclose(enhanced[0, :2], prefix[0, :2])


def test_invalid_geometry_and_empty_samples_are_ignored():
    torch.manual_seed(3)
    adapter = CVGE(24, _config()).eval()
    _activate(adapter)
    prefix = torch.randn(2, 7, 24)
    masks = torch.ones(2, 4, dtype=torch.bool)
    valid = torch.tensor([[True, False, True], [False, False, False]])
    tokens = torch.randn(2, 3, 6, 2048)
    clean = adapter(prefix, GeometryContext(tokens, valid), masks)
    tokens[~valid] = float("nan")
    dirty = adapter(prefix, GeometryContext(tokens, valid), masks)
    assert torch.isfinite(dirty).all()
    torch.testing.assert_close(clean, dirty, rtol=0, atol=0)
    torch.testing.assert_close(dirty[1], prefix[1], rtol=0, atol=0)


def test_cross_view_memory_changes_visual_output():
    torch.manual_seed(11)
    adapter = CVGE(24, _config()).eval()
    _activate(adapter)
    prefix = torch.randn(1, 7, 24)
    mask = torch.ones(1, 4, dtype=torch.bool)
    tokens = torch.randn(1, 2, 6, 2048)
    geometry = GeometryContext(tokens, torch.ones(1, 2, dtype=torch.bool))
    before = adapter(prefix, geometry, mask)
    changed = tokens.clone()
    changed[:, 1] += 3
    after = adapter(prefix, dataclasses.replace(geometry, tokens=changed), mask)
    assert not torch.allclose(before[:, :4], after[:, :4])


def test_camera_conditioning_requires_pose_and_ignores_masked_pose():
    adapter = CVGE(24, _config(use_camera_pose=True)).eval()
    _activate(adapter)
    prefix = torch.randn(1, 6, 24)
    mask = torch.ones(1, 4, dtype=torch.bool)
    geometry = GeometryContext(torch.randn(1, 2, 6, 2048), torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="camera_to_world"):
        adapter(prefix, geometry, mask)
    poses = torch.eye(4).expand(1, 2, 4, 4).clone()
    clean = adapter(prefix, dataclasses.replace(geometry, camera_to_world=poses), mask)
    poses[:, 1] = float("nan")
    dirty = adapter(prefix, dataclasses.replace(geometry, camera_to_world=poses), mask)
    torch.testing.assert_close(clean, dirty)


class _MixingAggregator(nn.Module):
    """Mix every supplied view so including a padded view is observable."""

    patch_start_idx = 5
    patch_size = 14

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        batch, views = images.shape[:2]
        shared = images.mean(dim=(1, 2, 3, 4))[:, None, None, None] * self.weight
        return [shared.expand(batch, views, 6, 2048)], self.patch_start_idx


def test_encoder_excludes_missing_views_before_aggregation():
    config = _config(image_size=14)
    aggregator = _MixingAggregator()
    encoder = VGGTEncoder(config, load_weights=False, aggregator=aggregator)
    encoder.train()
    assert not encoder.training
    assert not aggregator.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    images = [torch.full((3, 3, 14, 14), value) for value in (-0.6, 0.0, 0.8)]
    masks = [torch.tensor([True, True, False]), torch.tensor([False, True, False]), torch.tensor([True, False, False])]
    # Invalid images are NaN: merely masking the final memory would not suffice.
    for image, mask in zip(images, masks, strict=True):
        image[~mask] = float("nan")
    context = encoder(images, masks)
    assert len(aggregator.calls) == 2
    assert all(torch.isfinite(call).all() and call.shape[1] == 2 for call in aggregator.calls)
    torch.testing.assert_close(context.tokens[0, 0], torch.full((6, 2048), 0.55))
    torch.testing.assert_close(context.tokens[1, 0], torch.full((6, 2048), 0.35))
    assert torch.count_nonzero(context.tokens[2]) == 0
    assert not context.tokens.requires_grad


def test_encoder_all_views_missing_never_calls_vggt():
    aggregator = _MixingAggregator()
    encoder = VGGTEncoder(_config(image_size=14), load_weights=False, aggregator=aggregator)
    context = encoder([torch.zeros(2, 3, 14, 14)] * 3, [torch.zeros(2, dtype=torch.bool)] * 3)
    assert not aggregator.calls
    assert context.tokens.shape == (2, 3, 6, 2048)
    assert torch.isfinite(context.tokens).all()


def test_encoder_letterboxes_raw_images_without_aspect_ratio_distortion():
    aggregator = _MixingAggregator()
    encoder = VGGTEncoder(_config(image_size=14, image_keys=("base_0_rgb",)), load_weights=False, aggregator=aggregator)
    image = torch.ones(1, 3, 7, 14)
    mask = torch.ones(1, dtype=torch.bool)

    encoder([image], [mask])
    encoder([image], [mask])

    assert len(aggregator.calls) == 2
    first, second = aggregator.calls
    torch.testing.assert_close(first, second)
    assert first.shape == (1, 1, 3, 14, 14)
    assert torch.count_nonzero(first[:, :, :, :3]) == 0
    assert torch.count_nonzero(first[:, :, :, 10:]) == 0
    assert torch.all(first[:, :, :, 3:10] == 1)
