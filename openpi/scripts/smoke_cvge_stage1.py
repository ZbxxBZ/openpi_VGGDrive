"""Stage-1 GPU smoke test for pi0.5 with a frozen VGGT CVGE branch.

This uses synthetic observations and local checkpoints. It does not download
models, update weights, or evaluate task performance.
"""

import argparse
import dataclasses
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

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _peak_memory_gib() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def _geometry_config(args: argparse.Namespace) -> GeometryConfig:
    return GeometryConfig(
        enabled=True,
        backbone=args.backbone,
        vggt_source_path=str(args.vggt_source),
        vggt_weights_path=str(args.vggt_weights),
        image_keys=IMAGE_KEYS[: args.num_cameras],
        image_size=args.image_size,
        dropout=0.0,
        train_policy=args.train_policy,
    )


def geometry_smoke(args: argparse.Namespace) -> dict:
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(json.dumps({"event": "geometry_encoder_load_started"}), flush=True)
    geometry = _geometry_config(args)
    encoder = VGGTEncoder(geometry).to(args.device).eval()
    loaded = time.perf_counter()
    print(json.dumps({"event": "geometry_forward_started"}), flush=True)
    images = [torch.zeros(args.batch_size, 3, 224, 224, device=args.device) for _ in geometry.image_keys]
    masks = [torch.ones(args.batch_size, dtype=torch.bool, device=args.device) for _ in geometry.image_keys]
    context = encoder(images, masks)
    finished = time.perf_counter()
    special_tokens = 5 if args.backbone == "vggt" else 17
    expected_tokens = (geometry.image_size // geometry.patch_size) ** 2 + special_tokens
    if context.tokens.shape != (args.batch_size, args.num_cameras, expected_tokens, 2048):
        raise RuntimeError(f"Unexpected {args.backbone} geometry shape: {tuple(context.tokens.shape)}")
    if not torch.isfinite(context.tokens).all():
        raise RuntimeError(f"{args.backbone} geometry output contains non-finite values")
    result = {
        "stage": "geometry",
        "shape": list(context.tokens.shape),
        "dtype": str(context.tokens.dtype),
        "load_seconds": round(loaded - start, 3),
        "forward_seconds": round(finished - loaded, 3),
        "peak_cuda_gib": round(_peak_memory_gib(), 3),
    }
    del context, images, masks, encoder
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
    valid_keys = set(IMAGE_KEYS[: args.num_cameras])
    return SimpleNamespace(
        images={key: torch.zeros(args.batch_size, 3, 224, 224, device=args.device) for key in IMAGE_KEYS},
        image_masks={
            key: torch.full(
                (args.batch_size,), key in valid_keys, dtype=torch.bool, device=args.device
            )
            for key in IMAGE_KEYS
        },
        state=torch.zeros(args.batch_size, 32, device=args.device),
        tokenized_prompt=torch.zeros(
            args.batch_size, args.prompt_tokens, dtype=torch.long, device=args.device
        ),
        tokenized_prompt_mask=torch.ones(
            args.batch_size, args.prompt_tokens, dtype=torch.bool, device=args.device
        ),
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
    noise = torch.zeros(args.batch_size, 32, 32, device=args.device)
    print(json.dumps({"event": "policy_sample_started"}), flush=True)
    actions = model.sample_actions(args.device, observation, noise=noise, num_steps=1)
    finished = time.perf_counter()
    if actions.shape != (args.batch_size, 32, 32):
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


def _run_training_stage(
    model: PI0Pytorch,
    args: argparse.Namespace,
    train_policy: str,
    observation: SimpleNamespace,
    actions: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> dict:
    model.config = dataclasses.replace(
        model.config,
        geometry=dataclasses.replace(model.config.geometry, train_policy=train_policy),
    )
    model.configure_geometry_training()
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4, weight_decay=0.0
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    torch.cuda.synchronize()
    initial_cuda_gib = torch.cuda.memory_allocated() / 1024**3
    started = time.perf_counter()
    step_results = []
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        stage = "stage1" if train_policy == "adapter_only" else "stage2"
        print(json.dumps({"event": "policy_train_forward_started", "stage": stage, "step": step}), flush=True)
        loss = model(observation, actions, noise=noise, time=timestep).mean()
        torch.cuda.synchronize()
        forward_finished = time.perf_counter()
        if not torch.isfinite(loss):
            raise RuntimeError(f"pi0.5 training loss is non-finite at step {step}")
        print(
            json.dumps({"event": "policy_train_backward_started", "stage": stage, "step": step, "loss": loss.item()}),
            flush=True,
        )
        loss.backward()
        torch.cuda.synchronize()
        backward_finished = time.perf_counter()
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
        optimizer.step()
        torch.cuda.synchronize()
        step_finished = time.perf_counter()
        step_result = {
            "step": step,
            "loss": round(loss.item(), 6),
            "forward_seconds": round(forward_finished - step_started, 4),
            "backward_seconds": round(backward_finished - forward_finished, 4),
            "optimizer_seconds": round(step_finished - backward_finished, 4),
            "step_seconds": round(step_finished - step_started, 4),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "final_cvge_grad_l1": round(terminal_grad.abs().sum().item(), 6),
            "layers_with_inner_grad": layers_with_inner_grad,
            "inner_grad_l1": round(inner_grad_l1, 6),
        }
        step_results.append(step_result)
        print(json.dumps({"event": "policy_train_step_finished", "stage": stage, **step_result}), flush=True)
    finished = time.perf_counter()
    if any(parameter.grad is not None for parameter in model.vggt_encoder.parameters()):
        raise RuntimeError("Frozen geometry encoder received gradients")
    if train_policy != "full" and any(
        parameter.grad is not None for parameter in model.paligemma_with_expert.paligemma.parameters()
    ):
        raise RuntimeError(f"Frozen PaliGemma received gradients in {train_policy} mode")
    if train_policy == "adapter_only" and any(
        parameter.grad is not None for parameter in model.paligemma_with_expert.gemma_expert.parameters()
    ):
        raise RuntimeError("Frozen action expert received gradients in adapter_only mode")
    if train_policy in ("adapter_action", "full") and not any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in model.paligemma_with_expert.gemma_expert.parameters()
    ):
        raise RuntimeError(f"Action expert has no finite, nonzero gradient in {train_policy} mode")
    if train_policy == "full" and not any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in model.paligemma_with_expert.paligemma.parameters()
    ):
        raise RuntimeError("PaliGemma has no finite, nonzero gradient in full mode")
    result = {
        "stage": "stage1" if train_policy == "adapter_only" else "stage2",
        "train_policy": train_policy,
        "batch_size": args.batch_size,
        "num_cameras": args.num_cameras,
        "trainable_parameters": trainable_parameters,
        "initial_cuda_gib": round(initial_cuda_gib, 3),
        "steps": step_results,
        "training_seconds": round(finished - started, 3),
        "peak_allocated_gib": max(item["peak_allocated_gib"] for item in step_results),
        "peak_reserved_gib": max(item["peak_reserved_gib"] for item in step_results),
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def training_smoke(args: argparse.Namespace) -> dict:
    torch.cuda.reset_peak_memory_stats()
    model, load_seconds = _load_policy(args)
    model.gradient_checkpointing_enable()
    observation = _observation(args)
    torch.manual_seed(7)
    actions = torch.randn(args.batch_size, 32, 32, device=args.device)
    noise = torch.randn_like(actions)
    timestep = torch.full((args.batch_size,), 0.5, device=args.device)
    policies = ("adapter_only", "full") if args.two_stage else (args.train_policy,)
    stages = [
        _run_training_stage(model, args, policy, observation, actions, noise, timestep) for policy in policies
    ]
    return {
        "stage": "two_stage_training" if args.two_stage else stages[0]["stage"],
        "load_seconds": round(load_seconds, 3),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("vggt", "vggt_omega"), default="vggt")
    parser.add_argument(
        "--vggt-source", "--omega-source", dest="vggt_source", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--vggt-weights", "--omega-weights", dest="vggt_weights", type=pathlib.Path, required=True
    )
    parser.add_argument("--pi05-weights", type=pathlib.Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--num-cameras", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument(
        "--train-policy", choices=("adapter_only", "adapter_action", "full"), default="adapter_only"
    )
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("geometry", "policy", "training", "both"), default="both")
    args = parser.parse_args()
    if args.train_steps < 2:
        parser.error("--train-steps must be at least 2 to verify gradients after zero initialization")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
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
