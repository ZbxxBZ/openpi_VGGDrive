"""Layer-wise CVGE: visual queries attend to a frozen, multi-view geometry memory.

The bottleneck MLPs, attention residual, LayerNorm and outer residual follow
VGGDrive/inject_utils/Qwen2_5_vggt_fusion_inject_cam.py. For pi0 the module runs
BEFORE each decoder layer, so even the last adapter affects the action loss.
"""

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models.geometry_config import GeometryConfig


@dataclasses.dataclass(frozen=True)
class GeometryContext:
    tokens: torch.Tensor  # [B, cameras, geometry_tokens, feature_dim]
    valid_views: torch.Tensor  # bool[B, cameras]
    camera_to_world: torch.Tensor | None = None  # [B, cameras, 4, 4], same camera order


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, valid_keys: torch.Tensor) -> torch.Tensor:
        batch, query_count, dim = queries.shape

        def split_heads(x):
            return x.reshape(batch, -1, self.num_heads, dim // self.num_heads).transpose(1, 2)

        # Give empty samples a zero dummy key. The outer adapter masks their entire
        # residual; this avoids all-masked softmax rows on every attention backend.
        has_memory = valid_keys.any(dim=-1, keepdim=True)
        first_key = torch.arange(memory.shape[1], device=memory.device)[None, :] == 0
        safe_keys = valid_keys | (~has_memory & first_key)
        memory = torch.where(valid_keys[..., None], memory, torch.zeros_like(memory))
        output = F.scaled_dot_product_attention(
            split_heads(self.q_proj(queries)),
            split_heads(self.k_proj(memory)),
            split_heads(self.v_proj(memory)),
            attn_mask=safe_keys[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, query_count, dim)
        residual = queries + self.out_proj(output)
        # Keep normalization numerically stable even if the surrounding model is BF16.
        return F.layer_norm(
            residual.float(),
            self.layernorm.normalized_shape,
            self.layernorm.weight.float(),
            self.layernorm.bias.float(),
            self.layernorm.eps,
        ).to(residual.dtype)


class CVGE(nn.Module):
    def __init__(self, hidden_dim: int, config: GeometryConfig):
        super().__init__()
        self.vision_proj = _mlp(hidden_dim, config.fusion_dim, config.fusion_dim)
        self.geometry_proj = _mlp(config.feature_dim, config.fusion_dim, config.fusion_dim)
        self.camera_proj = _mlp(16, config.fusion_dim, config.fusion_dim) if config.use_camera_pose else None
        self.fusion = CrossAttentionFusion(config.fusion_dim, config.num_heads, config.dropout)
        self.output_proj = _mlp(config.fusion_dim, config.fusion_dim, hidden_dim)
        # Start from the pretrained pi0 function. Only this projection learns on
        # the first backward pass; gradients reach the inner adapter after it moves.
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, prefix: torch.Tensor, geometry: GeometryContext, visual_mask: torch.Tensor) -> torch.Tensor:
        # pi0 packs all camera slots before language. Retain the fixed slots even
        # for absent cameras; boolean gather/stack would break heterogeneous batches.
        if visual_mask.ndim != 2 or visual_mask.shape[0] != prefix.shape[0]:
            raise ValueError("visual_mask must describe the leading image-token slots of the prefix")
        visual_count = visual_mask.shape[1]
        if not 0 < visual_count <= prefix.shape[1]:
            raise ValueError("visual_mask must describe the leading image-token slots of the prefix")
        tokens = geometry.tokens
        if tokens.ndim != 4 or tokens.shape[0] != prefix.shape[0] or tokens.shape[2] == 0:
            raise ValueError("Geometry tokens must have shape [B, cameras, nonempty_tokens, feature_dim]")
        if geometry.valid_views.shape != tokens.shape[:2] or tokens.shape[1] == 0:
            raise ValueError("Geometry view mask does not match geometry tokens")
        dtype = self.vision_proj[0].weight.dtype
        valid_views = geometry.valid_views.to(device=prefix.device, dtype=torch.bool)
        tokens = tokens.to(device=prefix.device, dtype=dtype)
        tokens = torch.where(valid_views[:, :, None, None], tokens, torch.zeros_like(tokens))
        queries = self.vision_proj(prefix[:, :visual_count].to(dtype))
        memory = self.geometry_proj(tokens)
        if self.camera_proj is not None:
            poses = geometry.camera_to_world
            if poses is None or poses.shape != (*tokens.shape[:2], 4, 4):
                raise ValueError("use_camera_pose requires camera_to_world with shape [B, cameras, 4, 4]")
            poses = poses.to(device=prefix.device, dtype=dtype)
            poses = torch.where(valid_views[:, :, None, None], poses, torch.zeros_like(poses))
            camera = self.camera_proj(poses.flatten(-2))
            # VGGT's first token in each view is the camera token.
            memory = torch.cat([memory[:, :, :1] + camera[:, :, None], memory[:, :, 1:]], dim=2)
        valid_keys = valid_views[:, :, None].expand(tokens.shape[:3]).flatten(1, 2)
        fused = self.fusion(queries, memory.flatten(1, 2), valid_keys)
        delta = self.output_proj(fused).to(prefix.dtype)
        update_mask = visual_mask.to(device=prefix.device, dtype=torch.bool) & valid_views.any(-1, keepdim=True)
        delta = torch.where(update_mask[..., None], delta, torch.zeros_like(delta))
        return torch.cat([prefix[:, :visual_count] + delta, prefix[:, visual_count:]], dim=1)
