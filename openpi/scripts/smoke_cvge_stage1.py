"""Stage-1 GPU smoke test for pi0.5 with the frozen VGGT-Omega CVGE branch.

This uses synthetic observations and local checkpoints. It does not download
models, update weights, or evaluate task performance.
"""

import argparse
import gc
import json
import pathlib
import time
from types import SimpleNamespace

import torch

from openpi.models.geometry_config import GeometryConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.geometry_checkpoint import load_pi0_weights
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.vggt_encoder import VGGTEncoder


def _peak_memory_gib() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def _geometry_config(args: argparse.Namespace) -> GeometryConfig:
    return GeometryConfig(
        enabled=True,
        backbone="vggt_omega",
        vggt_source_path=str(args.omega_source),
        vggt_weights_path=str(args.omega_weights),
        image_keys=("base_0_rgb",),
        image_size=args.image_size,
        dropout=0.0,
        train_policy="adapter_only",
    )


def geometry_smoke(args: argparse.Namespace) -> dict:
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(json.dumps({"event": "geometry_encoder_load_started"}), flush=True)
    encoder = VGGTEncoder(_geometry_config(args)).to(args.device).eval()
    loaded = time.perf_counter()
    print(json.dumps({"event": "geometry_forward_started"}), flush=True)
    image = torch.zeros(1, 3, 224, 224, device=args.device)
    context = encoder([image], [torch.ones(1, dtype=torch.bool, device=args.device)])
    finished = time.perf_counter()
    expected_tokens = (args.image_size // 16) ** 2 + 17
    if context.tokens.shape != (1, 1, expected_tokens, 2048):
        raise RuntimeError(f"Unexpected Omega geometry shape: {tuple(context.tokens.shape)}")
    if not torch.isfinite(context.tokens).all():
        raise RuntimeError("Omega geometry output contains non-finite values")
    result = {
        "stage": "geometry",
        "shape": list(context.tokens.shape),
        "dtype": str(context.tokens.dtype),
        "load_seconds": round(loaded - start, 3),
        "forward_seconds": round(finished - loaded, 3),
        "peak_cuda_gib": round(_peak_memory_gib(), 3),
    }
    del context, image, encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_policy(args: argparse.Namespace) -> tuple[PI0Pytorch, float]:
    config = Pi0Config(
        dtype="bfloat16",
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        pi05=True,
        discrete_state_input=True,
        pytorch_compile_mode=None,
        geometry=_geometry_config(args),
    )
    start = time.perf_counter()
    print(json.dumps({"event": "policy_model_load_started"}), flush=True)
    model = PI0Pytorch(config, load_geometry_weights=True)
    print(json.dumps({"event": "policy_base_weights_load_started"}), flush=True)
    load_pi0_weights(model, args.pi05_weights, allow_base=True, device="cpu")
    print(json.dumps({"event": "policy_cuda_transfer_started"}), flush=True)
    return model.to(args.device), time.perf_counter() - start


def _observation(args: argparse.Namespace) -> SimpleNamespace:
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return SimpleNamespace(
        images={key: torch.zeros(1, 3, 224, 224, device=args.device) for key in keys},
        image_masks={key: torch.tensor([key == "base_0_rgb"], dtype=torch.bool, device=args.device) for key in keys},
        state=torch.zeros(1, 32, device=args.device),
        tokenized_prompt=torch.zeros(1, args.prompt_tokens, dtype=torch.long, device=args.device),
        tokenized_prompt_mask=torch.ones(1, args.prompt_tokens, dtype=torch.bool, device=args.device),
        token_ar_mask=None,
        token_loss_mask=None,
        camera_to_world=None,
    )


def policy_smoke(args: argparse.Namespace) -> dict:
    torch.cuda.reset_peak_memory_stats()
    model, load_seconds = _load_policy(args)
    model.eval()
    loaded = time.perf_counter()
    observation = _observation(args)
    noise = torch.zeros(1, 32, 32, device=args.device)
    print(json.dumps({"event": "policy_sample_started"}), flush=True)
    actions = model.sample_actions(args.device, observation, noise=noise, num_steps=1)
    finished = time.perf_counter()
    if actions.shape != (1, 32, 32):
        raise RuntimeError(f"Unexpected pi0.5 action shape: {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("pi0.5 actions contain non-finite values")
    return {
        "stage": "policy",
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "load_seconds": round(load_seconds, 3),
        "sample_seconds": round(finished - loaded, 3),
        "peak_cuda_gib": round(_peak_memory_gib(), 3),
    }


def training_smoke(args: argparse.Namespace) -> dict:
    torch.cuda.reset_peak_memory_stats()
    model, load_seconds = _load_policy(args)
    model.train()
    model.gradient_checkpointing_enable()
    observation = _observation(args)
    torch.manual_seed(7)
    actions = torch.randn(1, 32, 32, device=args.device)
    noise = torch.randn_like(actions)
    timestep = torch.full((1,), 0.5, device=args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4, weight_decay=0.0
    )
    started = time.perf_counter()
    step_results = []
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"event": "policy_train_forward_started", "step": step}), flush=True)
        loss = model(observation, actions, noise=noise, time=timestep).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"pi0.5 training loss is non-finite at step {step}")
        print(json.dumps({"event": "policy_train_backward_started", "step": step, "loss": loss.item()}), flush=True)
        loss.backward()
        terminal_grad = model.paligemma_with_expert.cvge[-1].output_proj[-1].weight.grad
        if terminal_grad is None or not torch.isfinite(terminal_grad).all() or terminal_grad.abs().sum() == 0:
            raise RuntimeError(f"Final CVGE projection has no finite, nonzero gradient at step {step}")
        layers_with_inner_grad = 0
        inner_grad_l1 = 0.0
        for adapter in model.paligemma_with_expert.cvge:
            gradients = (
                adapter.vision_proj[0].weight.grad,
                adapter.geometry_proj[0].weight.grad,
                adapter.fusion.q_proj.weight.grad,
            )
            if all(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients):
                layers_with_inner_grad += 1
                inner_grad_l1 += sum(value.abs().sum().item() for value in gradients)
        if step > 0 and layers_with_inner_grad != len(model.paligemma_with_expert.cvge):
            raise RuntimeError(
                f"Only {layers_with_inner_grad}/{len(model.paligemma_with_expert.cvge)} CVGE layers have inner gradients"
            )
        step_results.append(
            {
                "step": step,
                "loss": round(loss.item(), 6),
                "final_cvge_grad_l1": round(terminal_grad.abs().sum().item(), 6),
                "layers_with_inner_grad": layers_with_inner_grad,
                "inner_grad_l1": round(inner_grad_l1, 6),
            }
        )
        optimizer.step()
    finished = time.perf_counter()
    if any(parameter.grad is not None for parameter in model.vggt_encoder.parameters()):
        raise RuntimeError("Frozen Omega received gradients")
    if any(parameter.grad is not None for parameter in model.paligemma_with_expert.paligemma.parameters()):
        raise RuntimeError("Frozen PaliGemma received gradients in adapter_only mode")
    if any(parameter.grad is not None for parameter in model.paligemma_with_expert.gemma_expert.parameters()):
        raise RuntimeError("Frozen action expert received gradients in adapter_only mode")
    return {
        "stage": "training",
        "steps": step_results,
        "load_seconds": round(load_seconds, 3),
        "forward_backward_seconds": round(finished - started, 3),
        "peak_cuda_gib": round(_peak_memory_gib(), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega-source", type=pathlib.Path, required=True)
    parser.add_argument("--omega-weights", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-weights", type=pathlib.Path)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("geometry", "policy", "training", "both"), default="both")
    args = parser.parse_args()
    if args.train_steps < 2:
        parser.error("--train-steps must be at least 2 to verify gradients after zero initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 smoke requires CUDA")
    if args.mode in ("policy", "training", "both") and args.pi05_weights is None:
        parser.error("--pi05-weights is required for policy smoke")
    if args.mode in ("geometry", "both"):
        print(json.dumps(geometry_smoke(args), sort_keys=True), flush=True)
    if args.mode in ("policy", "both"):
        print(json.dumps(policy_smoke(args), sort_keys=True), flush=True)
    if args.mode == "training":
        print(json.dumps(training_smoke(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
