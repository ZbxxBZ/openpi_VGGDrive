"""Real-data Stage-1 smoke for pi0.5, CVGE, and a local LeRobot v3 dataset.

The script reads genuine image/state/action/task samples, applies OpenPI's full
data transform pipeline, and performs in-memory adapter-only optimizer steps.
It never modifies the dataset or writes a training checkpoint.
"""

import argparse
import dataclasses
import json
import pathlib
import time

import jax
import torch

from openpi.models.geometry_config import GeometryConfig
from openpi.models_pytorch.geometry_checkpoint import load_pi0_weights
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as training_config
from openpi.training import data_loader

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _make_config(args: argparse.Namespace) -> training_config.TrainConfig:
    config = training_config.get_config("pi05_cvge_omega_robotwin")
    geometry = GeometryConfig(
        enabled=True,
        backbone="vggt_omega",
        vggt_source_path=str(args.omega_source),
        vggt_weights_path=str(args.omega_weights),
        image_keys=IMAGE_KEYS,
        image_size=args.image_size,
        dropout=0.0,
        train_policy="adapter_only",
    )
    model = dataclasses.replace(config.model, geometry=geometry, pytorch_compile_mode=None)
    base_data = dataclasses.replace(
        config.data.base_config,
        lerobot_root=str(args.lerobot_root),
        lerobot_video_backend=args.video_backend,
        lerobot_episodes=tuple(args.episodes),
    )
    data = dataclasses.replace(config.data, base_config=base_data)
    return dataclasses.replace(
        config,
        model=model,
        data=data,
        batch_size=1,
        num_workers=0,
        num_train_steps=args.train_steps,
        wandb_enabled=False,
    )


def _validate_batch(observation, actions: torch.Tensor) -> dict:
    expected_image_shape = (1, 3, 224, 224)
    if set(observation.images) != set(IMAGE_KEYS):
        raise RuntimeError(f"Unexpected camera keys: {tuple(observation.images)}")
    for key in IMAGE_KEYS:
        image = observation.images[key]
        if tuple(image.shape) != expected_image_shape or not torch.isfinite(image).all():
            raise RuntimeError(f"Invalid transformed image for {key}: {tuple(image.shape)}")
        if not bool(observation.image_masks[key].all()):
            raise RuntimeError(f"Real camera {key} was marked invalid")
    if tuple(actions.shape) != (1, 32, 32) or not torch.isfinite(actions).all():
        raise RuntimeError(f"Invalid transformed action batch: {tuple(actions.shape)}")
    if tuple(observation.state.shape) != (1, 32) or not torch.isfinite(observation.state).all():
        raise RuntimeError(f"Invalid transformed state batch: {tuple(observation.state.shape)}")
    if not bool(observation.tokenized_prompt_mask.any()):
        raise RuntimeError("The real LeRobot task produced an empty prompt")
    return {
        "image_shapes": {key: list(observation.images[key].shape) for key in IMAGE_KEYS},
        "valid_cameras": sum(bool(observation.image_masks[key].all()) for key in IMAGE_KEYS),
        "state_shape": list(observation.state.shape),
        "action_shape": list(actions.shape),
        "prompt_tokens": int(observation.tokenized_prompt_mask.sum()),
    }


def _inner_gradient_stats(model: PI0Pytorch) -> tuple[int, float]:
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
    return layers_with_inner_grad, inner_grad_l1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=pathlib.Path, required=True)
    parser.add_argument("--omega-source", type=pathlib.Path, required=True)
    parser.add_argument("--omega-weights", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-weights", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.train_steps < 2:
        parser.error("--train-steps must be at least 2 to check gradients after zero initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("The real-data Stage-1 smoke requires CUDA")

    config = _make_config(args)
    started = time.perf_counter()
    print(json.dumps({"event": "real_data_loader_started"}), flush=True)
    loader = data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=args.train_steps,
    )
    batches = []
    batch_metadata = None
    for step, (observation, actions) in enumerate(loader):
        metadata = _validate_batch(observation, actions)
        batch_metadata = batch_metadata or metadata
        batches.append((observation, actions))
        print(json.dumps({"event": "real_batch_loaded", "step": step}), flush=True)
    data_seconds = time.perf_counter() - started

    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"event": "policy_model_load_started"}), flush=True)
    load_started = time.perf_counter()
    model = PI0Pytorch(config.model, load_geometry_weights=True)
    load_pi0_weights(model, args.pi05_weights, allow_base=True, device="cpu")
    model = model.to(args.device).train()
    model.gradient_checkpointing_enable()
    load_seconds = time.perf_counter() - load_started
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.0,
    )

    torch.manual_seed(7)
    train_started = time.perf_counter()
    step_results = []
    for step, (cpu_observation, cpu_actions) in enumerate(batches):
        observation = jax.tree.map(lambda value: value.to(args.device), cpu_observation)
        actions = cpu_actions.to(device=args.device, dtype=torch.float32)
        noise = torch.randn_like(actions)
        timestep = torch.full((actions.shape[0],), 0.5, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(observation, actions, noise=noise, time=timestep).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite real-data loss at step {step}")
        loss.backward()

        terminal_grad = model.paligemma_with_expert.cvge[-1].output_proj[-1].weight.grad
        if terminal_grad is None or not torch.isfinite(terminal_grad).all() or terminal_grad.abs().sum() == 0:
            raise RuntimeError(f"Final CVGE projection has no finite, nonzero gradient at step {step}")
        layers_with_inner_grad, inner_grad_l1 = _inner_gradient_stats(model)
        if step > 0 and layers_with_inner_grad != len(model.paligemma_with_expert.cvge):
            raise RuntimeError(
                f"Only {layers_with_inner_grad}/{len(model.paligemma_with_expert.cvge)} "
                "CVGE layers have inner gradients"
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
        print(json.dumps({"event": "optimizer_step_finished", **step_results[-1]}), flush=True)

    frozen_modules = {
        "omega": model.vggt_encoder,
        "paligemma": model.paligemma_with_expert.paligemma,
        "action_expert": model.paligemma_with_expert.gemma_expert,
    }
    for name, module in frozen_modules.items():
        if any(parameter.grad is not None for parameter in module.parameters()):
            raise RuntimeError(f"Frozen {name} received gradients")

    result = {
        "stage": "lerobot_training",
        "dataset_root": str(args.lerobot_root),
        "episodes": args.episodes,
        "batch": batch_metadata,
        "steps": step_results,
        "data_seconds": round(data_seconds, 3),
        "model_load_seconds": round(load_seconds, 3),
        "training_seconds": round(time.perf_counter() - train_started, 3),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
