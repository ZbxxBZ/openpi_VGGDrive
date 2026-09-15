"""Real tiny Gemma layers: CVGE gradients and train/prefill/cache equivalence.

Requires OpenPI's transformers_replace installed, just like production pi0/pi0.5.
These tests do not download models or instantiate the large SigLIP/VGGT towers.
"""

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.configuration_gemma import GemmaConfig

from openpi.models.geometry_config import GeometryConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.cvge import CVGE
from openpi.models_pytorch.cvge import GeometryContext
from openpi.models_pytorch.cvge_test import _MixingAggregator
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.vggt_encoder import VGGTEncoder
from openpi.models_pytorch.vggt_omega_test import _OmegaAggregator


class _TinyPrefix(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        self.config = SimpleNamespace(text_config=language_model.config)
        self.vision_tower = nn.Module()
        self.image_proj = nn.Linear(3, language_model.config.hidden_size)

    @property
    def model(self):
        return self

    def get_image_features(self, image):
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        patches = F.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        return self.image_proj(patches)


def _tiny_model(*, layers=3, dropout=0.0, enabled=True, pi05=False, geometry_config=None):
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)

    def make_gemma(width, *, adaptive=False):
        config = GemmaConfig(
            hidden_size=width,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            use_adarms=adaptive,
            adarms_cond_dim=width if adaptive else None,
        )
        config._attn_implementation = "eager"  # noqa: SLF001
        gemma = modeling_gemma.GemmaModel(config).float()
        if adaptive:
            # Stand in for pretrained modulation: nonzero gates let action loss
            # reach visual K/V, and scale/shift depend on the timestep condition.
            for module in gemma.modules():
                if isinstance(module, modeling_gemma.GemmaRMSNorm) and module.dense is not None:
                    nn.init.normal_(module.dense.weight, std=0.02)
                    nn.init.zeros_(module.dense.bias)
                    with torch.no_grad():
                        module.dense.bias[2 * width :].fill_(0.5)
        return gemma

    model.paligemma = _TinyPrefix(make_gemma(32))
    model.gemma_expert = nn.Module()
    model.gemma_expert.model = make_gemma(24, adaptive=pi05)
    model.gemma_expert.model.embed_tokens = None
    if geometry_config is None:
        geometry_config = GeometryConfig(enabled=enabled, fusion_dim=16, num_heads=4, dropout=dropout)
    model.cvge = nn.ModuleList([CVGE(32, geometry_config) for _ in range(layers)] if geometry_config.enabled else [])
    return model.eval()


def _inputs():
    prefix = torch.randn(2, 7, 32)
    suffix = torch.randn(2, 3, 24)
    prefix_valid = torch.tensor([[True, True, False, False, True, True, False], [True] * 7])
    valid = torch.cat([prefix_valid, torch.ones(2, 3, dtype=torch.bool)], dim=1)
    blocks = torch.tensor([0] * 7 + [1] * 3)
    visible = (blocks[None, :] <= blocks[:, None])[None] & valid[:, :, None] & valid[:, None, :]
    mask = torch.where(visible[:, None], 0.0, -2.3819763e38)
    positions = valid.long().cumsum(-1) - 1
    geometry = GeometryContext(torch.randn(2, 2, 6, 2048), torch.tensor([[True, False], [True, True]]))
    return prefix, suffix, mask, positions, geometry, prefix_valid[:, :4]


def _conditions(model, inputs):
    suffix = inputs[1]
    if model.gemma_expert.model.norm.dense is None:
        return [None, None]
    condition = torch.linspace(-1, 1, suffix.shape[0] * suffix.shape[-1]).reshape(suffix.shape[0], suffix.shape[-1])
    return [None, condition.to(suffix.device)]


def _joint(model, inputs):
    prefix, suffix, mask, positions, geometry, visual_mask = inputs
    outputs, _ = model(
        inputs_embeds=[prefix, suffix],
        attention_mask=mask,
        position_ids=positions,
        geometry=geometry,
        visual_mask=visual_mask,
        adarms_cond=_conditions(model, inputs),
    )
    return outputs[1]


def _activate(model):
    for adapter in model.cvge:
        nn.init.normal_(adapter.output_proj[-1].weight, std=0.03)


@pytest.mark.parametrize("layers", [3, 18])
@pytest.mark.parametrize("pi05", [False, True])
def test_joint_equals_cached_actions_and_cache_is_read_only(layers, pi05):
    torch.manual_seed(41)
    model = _tiny_model(layers=layers, pi05=pi05)
    _activate(model)
    inputs = _inputs()
    prefix, suffix, mask, positions, geometry, visual_mask = inputs
    with torch.no_grad():
        joint = _joint(model, inputs)
        _, cache = model(
            inputs_embeds=[prefix, None],
            attention_mask=mask[:, :, :7, :7],
            position_ids=positions[:, :7],
            geometry=geometry,
            visual_mask=visual_mask,
            use_cache=True,
        )
        saved = [(key.clone(), value.clone()) for key, value in cache]
        for _ in range(2):
            (_, result), _ = model(
                inputs_embeds=[None, suffix],
                attention_mask=mask[:, :, 7:],
                position_ids=positions[:, 7:],
                past_key_values=cache,
                use_cache=False,
                adarms_cond=_conditions(model, inputs),
            )
            torch.testing.assert_close(joint, result, atol=2e-5, rtol=2e-5)
        assert cache.get_seq_length() == 7
        for (key, value), (saved_key, saved_value) in zip(cache, saved, strict=True):
            torch.testing.assert_close(key, saved_key, atol=0, rtol=0)
            torch.testing.assert_close(value, saved_value, atol=0, rtol=0)


@pytest.mark.parametrize("pi05", [False, True])
def test_zero_cvge_matches_original_joint_forward(pi05):
    torch.manual_seed(43)
    enhanced = _tiny_model(pi05=pi05)
    original = _tiny_model(enabled=False, pi05=pi05)
    original.load_state_dict(
        {key: value for key, value in enhanced.state_dict().items() if not key.startswith("cvge.")}
    )
    inputs = _inputs()
    torch.testing.assert_close(_joint(enhanced, inputs), _joint(original, inputs), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("pi05", [False, True])
def test_every_layer_including_last_receives_action_gradients_after_warmup(pi05):
    torch.manual_seed(9)
    model = _tiny_model(layers=18, pi05=pi05)
    model.requires_grad_(requires_grad=False)
    model.cvge.requires_grad_(requires_grad=True)
    inputs = _inputs()
    optimizer = torch.optim.SGD(model.cvge.parameters(), lr=0.2)
    target = torch.randn(2, 3, 24)
    (_joint(model, inputs) - target).square().mean().backward()
    for adapter in model.cvge:
        assert adapter.output_proj[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (_joint(model, inputs) - target).square().mean().backward()
    for adapter in model.cvge:
        assert adapter.vision_proj[0].weight.grad.abs().sum() > 0
        assert adapter.geometry_proj[0].weight.grad.abs().sum() > 0
        assert adapter.fusion.q_proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.paligemma.parameters())


@pytest.mark.parametrize("pi05", [False, True])
def test_geometry_reaches_actions(pi05):
    torch.manual_seed(19)
    model = _tiny_model(pi05=pi05)
    _activate(model)
    inputs = _inputs()
    changed = list(inputs)
    changed[4] = dataclasses.replace(inputs[4], tokens=inputs[4].tokens + 2)
    assert not torch.allclose(_joint(model, inputs), _joint(model, changed))


@pytest.mark.parametrize("pi05", [False, True])
def test_checkpoint_recomputation_preserves_dropout_gradients(pi05):
    torch.manual_seed(23)
    direct = _tiny_model(dropout=0.2, pi05=pi05).train()
    _activate(direct)
    recomputed = copy.deepcopy(direct)
    recomputed.gradient_checkpointing = True
    inputs = _inputs()
    target = torch.randn(2, 3, 24)
    for model in (direct, recomputed):
        torch.manual_seed(29)
        (_joint(model, inputs) - target).square().mean().backward()
    for (name, parameter), (_, other) in zip(direct.named_parameters(), recomputed.named_parameters(), strict=True):
        if name.startswith("cvge.") or (pi05 and ".dense." in name):
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5)


def test_geometry_prefix_cannot_silently_omit_context():
    model = _tiny_model()
    prefix, _, mask, positions, _, _ = _inputs()
    with pytest.raises(ValueError, match="requires geometry"):
        model(inputs_embeds=[prefix, None], attention_mask=mask[:, :, :7, :7], position_ids=positions[:, :7])


def test_pi05_time_condition_changes_actions_with_the_same_prefix_cache():
    torch.manual_seed(47)
    model = _tiny_model(pi05=True)
    _activate(model)
    prefix, suffix, mask, positions, geometry, visual_mask = _inputs()
    with torch.no_grad():
        _, cache = model(
            inputs_embeds=[prefix, None],
            attention_mask=mask[:, :, :7, :7],
            position_ids=positions[:, :7],
            geometry=geometry,
            visual_mask=visual_mask,
            use_cache=True,
        )
        saved = [(key.clone(), value.clone()) for key, value in cache]
        outputs = []
        for condition in (torch.zeros(2, 24), torch.ones(2, 24)):
            joint, _ = model(
                inputs_embeds=[prefix, suffix],
                attention_mask=mask,
                position_ids=positions,
                geometry=geometry,
                visual_mask=visual_mask,
                adarms_cond=[None, condition],
            )
            (_, cached), _ = model(
                inputs_embeds=[None, suffix],
                attention_mask=mask[:, :, 7:],
                position_ids=positions[:, 7:],
                past_key_values=cache,
                use_cache=False,
                adarms_cond=[None, condition],
            )
            torch.testing.assert_close(joint[1], cached, atol=2e-5, rtol=2e-5)
            outputs.append(joint)
        # Padding queries are not meaningful; compare all valid visual/text/state slots.
        prefix_valid = (mask[:, 0, :7, :7] == 0).any(-1)
        torch.testing.assert_close(outputs[0][0][prefix_valid], outputs[1][0][prefix_valid], atol=0, rtol=0)
        assert not torch.allclose(outputs[0][1], outputs[1][1])
        for (key, value), (saved_key, saved_value) in zip(cache, saved, strict=True):
            torch.testing.assert_close(key, saved_key, atol=0, rtol=0)
            torch.testing.assert_close(value, saved_value, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pi05_condition_precision_and_gradient(dtype):
    model = _tiny_model(pi05=True).train()
    model.to_bfloat16_for_selected_params("bfloat16" if dtype == torch.bfloat16 else "float32")
    prefix, suffix, mask, positions, geometry, visual_mask = _inputs()
    # Simulate a BF16 time MLP feeding the norm projections that are kept in FP32.
    condition = torch.randn(2, 24, dtype=torch.bfloat16, requires_grad=True)
    outputs, _ = model(
        inputs_embeds=[prefix, suffix],
        attention_mask=mask,
        position_ids=positions,
        geometry=geometry,
        visual_mask=visual_mask,
        adarms_cond=[None, condition],
    )
    assert outputs[1].dtype == dtype
    assert model.gemma_expert.model.norm.dense.weight.dtype == torch.float32
    outputs[1].float().square().mean().backward()
    assert torch.isfinite(condition.grad).all()
    assert condition.grad.abs().sum() > 0


def test_pi05_rejects_missing_or_misrouted_time_condition():
    model = _tiny_model(pi05=True)
    prefix, suffix, mask, positions, geometry, visual_mask = _inputs()
    for condition in (None, torch.zeros(2, 23), torch.zeros(1, 24)):
        with pytest.raises(ValueError, match="require adarms_cond"):
            model(
                inputs_embeds=[prefix, suffix],
                attention_mask=mask,
                position_ids=positions,
                geometry=geometry,
                visual_mask=visual_mask,
                adarms_cond=[None, condition],
            )
    with pytest.raises(ValueError, match="visual prefix"):
        model(
            inputs_embeds=[prefix, suffix],
            attention_mask=mask,
            position_ids=positions,
            geometry=geometry,
            visual_mask=visual_mask,
            adarms_cond=[torch.zeros(2, 32), torch.zeros(2, 24)],
        )
    # The cached suffix path must validate the condition before entering Gemma too.
    with pytest.raises(ValueError, match="require adarms_cond"):
        model(inputs_embeds=[None, suffix], attention_mask=mask[:, :, 7:], position_ids=positions[:, 7:])


def _tiny_policy(
    monkeypatch, *, pi05=False, backbone="vggt", discrete_state_input=False, train_policy="adapter_action"
):
    monkeypatch.setattr(
        pi0_pytorch._gemma,  # noqa: SLF001
        "get_config",
        lambda variant: SimpleNamespace(width=32 if variant == "gemma_2b" else 24),
    )

    def make_wrapper(*args, **kwargs):
        assert kwargs["use_adarms"] == [False, pi05]
        return _tiny_model(pi05=pi05, geometry_config=kwargs["geometry_config"])

    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", make_wrapper)
    monkeypatch.setattr(
        pi0_pytorch,
        "VGGTEncoder",
        lambda config, **kwargs: VGGTEncoder(
            config,
            load_weights=False,
            aggregator=_OmegaAggregator() if config.backbone == "vggt_omega" else _MixingAggregator(),
        ),
    )
    config = Pi0Config(
        dtype="float32",
        action_dim=6,
        action_horizon=3,
        max_token_len=4,
        pi05=pi05,
        discrete_state_input=discrete_state_input,
        pytorch_compile_mode=None,
        geometry=GeometryConfig(
            enabled=True,
            backbone=backbone,
            image_size=16 if backbone == "vggt_omega" else 14,
            fusion_dim=16,
            num_heads=4,
            dropout=0.0,
            train_policy=train_policy,
        ),
    )
    return pi0_pytorch.PI0Pytorch(config, load_geometry_weights=False)


@pytest.mark.parametrize("train_policy", ["adapter_only", "adapter_action", "full"])
def test_pi05_preserves_suffix_layout_and_training_policy(monkeypatch, train_policy):
    model = _tiny_policy(monkeypatch, pi05=True, backbone="vggt_omega", train_policy=train_policy).train()
    state = torch.randn(2, 6)
    noisy_actions = torch.randn(2, 3, 6)
    suffix, valid, blocks, cond = model.embed_suffix(state, noisy_actions, torch.zeros(2))
    changed, _, _, later_cond = model.embed_suffix(state + 10, noisy_actions, torch.ones(2))
    assert suffix.shape == (2, model.config.action_horizon, 24)
    assert valid.all()
    torch.testing.assert_close(blocks, torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=blocks.dtype))
    torch.testing.assert_close(suffix, changed, atol=0, rtol=0)
    assert cond.shape == (2, 24)
    assert not torch.allclose(cond, later_cond)
    assert not hasattr(model, "state_proj")
    assert not hasattr(model, "action_time_mlp_in")
    for name, parameter in model.named_parameters():
        if name.startswith("vggt_encoder."):
            assert not parameter.requires_grad
        elif name.startswith("paligemma_with_expert.cvge."):
            assert parameter.requires_grad
        elif name.startswith("paligemma_with_expert.paligemma."):
            assert parameter.requires_grad == (train_policy == "full")
        else:
            assert parameter.requires_grad == (train_policy != "adapter_only")
    assert not model.vggt_encoder.training


@pytest.mark.parametrize(("pi05", "discrete_state_input"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("backbone", ["vggt", "vggt_omega"])
def test_action_flow_and_single_geometry_prefill(monkeypatch, pi05, discrete_state_input, backbone):
    torch.manual_seed(53)
    model = _tiny_policy(monkeypatch, pi05=pi05, backbone=backbone, discrete_state_input=discrete_state_input)
    config = model.config
    observation = SimpleNamespace(
        images={key: torch.rand(2, 3, 224, 224) * 2 - 1 for key in config.geometry.image_keys},
        image_masks={key: torch.tensor([True, key != "right_wrist_0_rgb"]) for key in config.geometry.image_keys},
        state=torch.randn(2, 6),
        tokenized_prompt=torch.randint(0, 32, (2, 4)),
        tokenized_prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    model.train()
    model.gradient_checkpointing_enable()
    loss = model(observation, torch.randn(2, 3, 6)).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert not model.vggt_encoder.training
    assert all(parameter.grad is None for parameter in model.vggt_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.paligemma_with_expert.paligemma.parameters())
    assert model.paligemma_with_expert.cvge[-1].output_proj[-1].weight.grad.abs().sum() > 0
    if pi05:
        assert not hasattr(model, "state_proj")
        for projection in (model.time_mlp_in, model.time_mlp_out):
            assert projection.weight.grad.abs().sum() > 0
        expert = model.paligemma_with_expert.gemma_expert.model
        for layer in expert.layers:
            assert layer.input_layernorm.dense.weight.grad.abs().sum() > 0
            assert layer.post_attention_layernorm.dense.weight.grad.abs().sum() > 0
    model.eval()
    calls = [0] * (len(model.paligemma_with_expert.cvge) + 1)

    def count(index):
        def hook(*args):
            calls[index] += 1

        return hook

    handles = [model.vggt_encoder.register_forward_hook(count(0))]
    handles.extend(
        adapter.register_forward_hook(count(index + 1))
        for index, adapter in enumerate(model.paligemma_with_expert.cvge)
    )
    noise = torch.randn(2, 3, 6)
    try:
        actions = model.sample_actions(torch.device("cpu"), observation, noise=noise, num_steps=3)
        assert actions.shape == (2, 3, 6)
        assert torch.isfinite(actions).all()
        assert calls == [1] * len(calls)
        again = model.sample_actions(torch.device("cpu"), observation, noise=noise, num_steps=3)
        torch.testing.assert_close(again, actions)
        assert calls == [2] * len(calls)
    finally:
        for handle in handles:
            handle.remove()
