from typing import Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from openpi.models.geometry_config import GeometryConfig
from openpi.models_pytorch.cvge import CVGE
from openpi.models_pytorch.cvge import GeometryContext


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        geometry_config: GeometryConfig | None = None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        if vlm_config.depth != action_expert_config.depth:
            raise ValueError("PaliGemma and the action expert must have the same number of layers")
        self.cvge = nn.ModuleList(
            [CVGE(vlm_config.width, geometry_config) for _ in range(vlm_config.depth)]
            if geometry_config is not None and geometry_config.enabled
            else []
        )

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
        geometry: GeometryContext | None = None,
        visual_mask: torch.Tensor | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if self.cvge:
            adarms_cond = self._prepare_geometry_conditions(inputs_embeds, adarms_cond)
        if self.cvge and inputs_embeds[0] is not None:
            if geometry is None or visual_mask is None:
                raise ValueError("CVGE prefix processing requires geometry and visual_mask in training and inference")
            return self._forward_with_geometry(
                inputs_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                geometry,
                visual_mask,
                past_key_values=past_key_values,
                use_cache=bool(use_cache),
            )
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

    def _prepare_geometry_conditions(self, inputs_embeds, adarms_cond):
        """Keep pi0.5 timestep conditioning on the action expert in every CVGE path.

        This also runs for suffix-only denoising, which uses the original Gemma
        forward. Adaptive norm projections stay FP32 when attention uses BF16.
        Casting the condition keeps its gradient connected to the time MLP.
        """
        if inputs_embeds is None or len(inputs_embeds) != 2 or all(value is None for value in inputs_embeds):
            raise ValueError("CVGE expects prefix and/or action embeddings in a two-element sequence")
        if len(adarms_cond) != 2:
            raise ValueError("adarms_cond must contain separate prefix and action conditions")
        models = [self.paligemma.language_model, self.gemma_expert.model]
        conditions = []
        for model, states, cond in zip(models, inputs_embeds, adarms_cond, strict=True):
            if states is None:
                conditions.append(None)
                continue
            projection = model.norm.dense
            if projection is None:
                if cond is not None:
                    raise ValueError("Unconditional Gemma streams, including the visual prefix, cannot take adarms_cond")
                conditions.append(None)
                continue
            if cond is None or cond.shape != (states.shape[0], projection.in_features):
                raise ValueError("pi0.5 action embeddings require adarms_cond with shape [batch, expert_width]")
            conditions.append(cond.to(device=states.device, dtype=projection.weight.dtype))
        return conditions

    def _forward_with_geometry(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        adarms_cond,
        geometry,
        visual_mask,
        *,
        past_key_values=None,
        use_cache=False,
    ):
        """Shared per-layer computation for joint training and prefix prefill.

        Suffix-only denoising deliberately uses the original action expert above.
        Its read-only cache access is implemented by OpenPI's Gemma replacement.
        """
        if attention_mask is None or attention_mask.ndim != 4 or position_ids is None:
            raise ValueError("CVGE requires the explicit pi0 attention mask and position_ids")
        if past_key_values is not None:
            raise ValueError("A new geometry observation must start with an empty prefix cache")
        if use_cache and (self.training or inputs_embeds[1] is not None):
            raise ValueError("CVGE cache creation is supported for evaluation prefix prefill only")
        models = [self.paligemma.language_model, self.gemma_expert.model]
        cache = DynamicCache() if use_cache else None
        use_checkpoint = self.training and (
            getattr(self, "gradient_checkpointing", False)
            or any(getattr(model, "gradient_checkpointing", False) for model in models)
        )
        hidden = [
            value.to(model.layers[0].self_attn.q_proj.weight.dtype) if value is not None else None
            for value, model in zip(inputs_embeds, models, strict=True)
        ]
        for layer_idx in range(len(self.cvge)):
            args = (layer_idx, hidden, attention_mask, position_ids, adarms_cond, geometry, visual_mask, cache)
            if use_checkpoint:
                # CVGE attention has dropout; recomputation MUST reuse its RNG state.
                hidden = checkpoint(self._geometry_layer, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                hidden = self._geometry_layer(*args)
        output = [
            model.norm(value, cond=cond)[0] if value is not None else None
            for model, value, cond in zip(models, hidden, adarms_cond, strict=True)
        ]
        return output, cache

    def _geometry_layer(
        self, layer_idx, hidden, attention_mask, position_ids, adarms_cond, geometry, visual_mask, cache
    ):
        models = [self.paligemma.language_model, self.gemma_expert.model]
        # Inject before normalization and Q/K/V construction, including the last layer.
        hidden = [self.cvge[layer_idx](hidden[0], geometry, visual_mask), hidden[1]]
        queries, keys, values, gates = [], [], [], []
        for model, states, cond in zip(models, hidden, adarms_cond, strict=True):
            if states is None:
                gates.append(None)
                continue
            layer = model.layers[layer_idx]
            normalized, gate = layer.input_layernorm(states, cond=cond)
            gates.append(gate)
            shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
            queries.append(layer.self_attn.q_proj(normalized).view(shape).transpose(1, 2))
            keys.append(layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2))
            values.append(layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2))
        query = torch.cat(queries, dim=2)
        key = torch.cat(keys, dim=2)
        value = torch.cat(values, dim=2)
        cos, sin = models[0].rotary_emb(hidden[0], position_ids)
        query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
        if cache is not None:
            key, value = cache.update(key, value, layer_idx)
        attention = models[0].layers[layer_idx].self_attn
        attended, _ = modeling_gemma.eager_attention_forward(
            attention, query, key, value, attention_mask, attention.scaling
        )
        attended = attended.reshape(query.shape[0], query.shape[2], -1)
        outputs = []
        offset = 0
        for model, states, cond, gate in zip(models, hidden, adarms_cond, gates, strict=True):
            if states is None:
                outputs.append(None)
                continue
            layer = model.layers[layer_idx]
            end = offset + states.shape[1]
            update = layer.self_attn.o_proj(attended[:, offset:end].to(layer.self_attn.o_proj.weight.dtype))
            residual = modeling_gemma._gated_residual(states, update, gate)  # noqa: SLF001
            normalized, mlp_gate = layer.post_attention_layernorm(residual, cond=cond)
            update = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
            outputs.append(modeling_gemma._gated_residual(residual, update, mlp_gate))  # noqa: SLF001
            offset = end
        return outputs
