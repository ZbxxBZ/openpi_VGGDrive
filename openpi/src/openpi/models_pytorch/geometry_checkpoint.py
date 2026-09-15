"""Strict CVGE checkpoint loading and original pi0 / pi0.5 initialization."""

import dataclasses
import json
import pathlib

from safetensors import safe_open
import safetensors.torch

_GEOMETRY_PREFIXES = ("vggt_encoder.", "paligemma_with_expert.cvge.")
_METADATA_FILE = "geometry_config.json"


def checkpoint_has_geometry(weight_path: str | pathlib.Path) -> bool:
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        return any(key.startswith(_GEOMETRY_PREFIXES) for key in keys)


def _metadata(config) -> dict:
    return {
        "version": 3,
        "geometry": dataclasses.asdict(config.geometry),
        "model": {
            name: getattr(config, name)
            for name in (
                "paligemma_variant",
                "action_expert_variant",
                "action_dim",
                "action_horizon",
                "max_token_len",
                "pi05",
                "discrete_state_input",
            )
        },
    }


def save_geometry_metadata(model, directory: str | pathlib.Path) -> None:
    if model.config.geometry.enabled:
        path = pathlib.Path(directory) / _METADATA_FILE
        path.write_text(json.dumps(_metadata(model.config), indent=2) + "\n", encoding="utf-8")


def _validate_metadata(config, directory: pathlib.Path, *, resume: bool) -> None:
    path = directory / _METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"CVGE checkpoint is missing {path}; geometry settings cannot be verified")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("version") == 1:
        # Version 1 predates the backbone selector and only supported VGGT.
        saved["geometry"].setdefault("backbone", "vggt")
        if saved["geometry"]["backbone"] != "vggt":
            raise ValueError("Version 1 geometry checkpoints cannot contain a VGGT-Omega backbone")
        saved["version"] = 2
    if saved.get("version") == 2:
        if "discrete_state_input" not in saved["model"]:
            if saved["model"].get("pi05"):
                raise ValueError(
                    "Legacy pi0.5 CVGE metadata is missing discrete_state_input. Recover this setting from "
                    "the original training config and add it to geometry_config.json under model before loading."
                )
            # Legacy pi0 presets used continuous state tokens; pi0.5 may opt in or
            # out of discrete state, which is impossible to infer from tensor shapes.
            saved["model"]["discrete_state_input"] = False
        saved["version"] = 3
    expected = json.loads(json.dumps(_metadata(config)))
    # Complete checkpoints embed VGGT weights; machine-local paths may change.
    ignored = {"vggt_source_path", "vggt_weights_path"}
    if not resume:
        ignored.add("train_policy")
    for metadata in (saved, expected):
        for name in ignored:
            metadata.get("geometry", {}).pop(name, None)
    if saved != expected:
        raise ValueError(
            f"CVGE checkpoint configuration mismatch in {path}. "
            f"Saved: {saved}; requested: {expected}. Use matching geometry/model settings."
        )


def load_pi0_weights(
    model, weight_path: str | pathlib.Path, *, allow_base: bool = False, resume: bool = False, device: str = "cpu"
) -> None:
    """Only explicit base initialization may omit the entire geometry branch.

    Partially saved CVGE, missing base-model keys, and unexpected keys fail.
    Tied weights are handled by safetensors.load_model.
    """
    weight_path = pathlib.Path(weight_path)
    has_geometry = checkpoint_has_geometry(weight_path)
    enabled = model.config.geometry.enabled
    if has_geometry:
        if not enabled:
            raise ValueError("This checkpoint contains CVGE; select a configuration with geometry.enabled=True")
        _validate_metadata(model.config, weight_path.parent, resume=resume)
    if enabled and not has_geometry:
        if not allow_base or resume:
            raise ValueError(
                "Expected a complete CVGE checkpoint; original pi0/pi0.5 weights are only valid for initialization"
            )
        missing, unexpected = safetensors.torch.load_model(model, str(weight_path), strict=False, device=device)
        expected_missing = {key for key in model.state_dict() if key.startswith(_GEOMETRY_PREFIXES)}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"Invalid original pi0/pi0.5 initialization: missing base keys={set(missing) - expected_missing}, "
                f"unexpected keys={unexpected}"
            )
        return
    safetensors.torch.load_model(model, str(weight_path), strict=True, device=device)
