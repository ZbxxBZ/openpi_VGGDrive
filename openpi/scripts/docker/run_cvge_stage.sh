#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
if [[ "$stage" != "stage1" && "$stage" != "stage2" ]]; then
  echo "Usage: $0 stage1|stage2" >&2
  exit 2
fi

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

require_dir /root/zbx/data/robotwin_unified
require_dir /root/zbx/openpi_VGGDrive/VGGDrive
require_file /root/zbx/openpi_VGGDrive/VGGDrive/vggt/models/aggregator.py
require_dir /root/zbx/weights/pi05_robotwin2/assets
require_dir /root/zbx/checkpoints

nproc_per_node="${NPROC_PER_NODE:-1}"
batch_size="${BATCH_SIZE:-1}"
if ! [[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got: $nproc_per_node" >&2
  exit 2
fi
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || (( batch_size % nproc_per_node != 0 )); then
  echo "BATCH_SIZE must be a positive multiple of NPROC_PER_NODE" >&2
  exit 2
fi

common_args=(
  pi05_cvge_robotwin
  --checkpoint-base-dir /root/zbx/checkpoints
  --batch-size "$batch_size"
  --num-workers "${NUM_WORKERS:-0}"
  --save-interval "${SAVE_INTERVAL:-1000}"
  --keep-period "${KEEP_PERIOD:-5000}"
  --model.geometry.vggt-source-path /root/zbx/openpi_VGGDrive/VGGDrive
  --model.geometry.image-size "${IMAGE_SIZE:-224}"
)

if is_true "${WANDB_ENABLED:-0}"; then
  common_args+=(--wandb-enabled)
fi

if [[ "$stage" == "stage1" ]]; then
  exp_name="${STAGE1_EXP_NAME:-robotwin_stage1}"
  train_steps="${STAGE1_STEPS:-5000}"
  resume="${STAGE1_RESUME:-0}"
  overwrite="${STAGE1_OVERWRITE:-0}"
  common_args+=(
    --exp-name "$exp_name"
    --num-train-steps "$train_steps"
    --model.geometry.train-policy adapter_only
  )
  if ! is_true "$resume"; then
    require_file /root/zbx/weights/pi05_robotwin2/model.safetensors
    require_file /root/zbx/weights/vggt-1b/model.pt
    common_args+=(
      --pytorch-weight-path /root/zbx/weights/pi05_robotwin2
      --model.geometry.vggt-weights-path /root/zbx/weights/vggt-1b/model.pt
    )
  fi
else
  exp_name="${STAGE2_EXP_NAME:-robotwin_stage2}"
  train_steps="${STAGE2_STEPS:-20000}"
  resume="${STAGE2_RESUME:-0}"
  overwrite="${STAGE2_OVERWRITE:-0}"
  common_args+=(
    --exp-name "$exp_name"
    --num-train-steps "$train_steps"
    --model.geometry.train-policy full
  )
  if ! is_true "$resume"; then
    stage1_checkpoint="/root/zbx/checkpoints/pi05_cvge_robotwin/${STAGE1_EXP_NAME:-robotwin_stage1}/${STAGE1_CHECKPOINT_STEP:-5000}"
    require_file "$stage1_checkpoint/model.safetensors"
    require_file "$stage1_checkpoint/geometry_config.json"
    common_args+=(--pytorch-weight-path "$stage1_checkpoint")
  fi
fi

if is_true "$resume" && is_true "$overwrite"; then
  echo "Resume and overwrite cannot both be enabled for $stage" >&2
  exit 2
elif is_true "$resume"; then
  common_args+=(--resume)
elif is_true "$overwrite"; then
  common_args+=(--overwrite)
fi

python_bin="${PYTHON_BIN:-/opt/openpi-venv/bin/python}"
torchrun_bin="${TORCHRUN_BIN:-/opt/openpi-venv/bin/torchrun}"
require_file "$python_bin"
available_gpus="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if (( available_gpus < nproc_per_node )); then
  echo "Requested $nproc_per_node GPU process(es), but PyTorch sees $available_gpus GPU(s)" >&2
  exit 1
fi

echo "Starting $stage: exp=$exp_name steps=$train_steps batch=$batch_size gpus=$nproc_per_node"
if (( nproc_per_node > 1 )); then
  exec "$torchrun_bin" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$nproc_per_node" \
    scripts/train_pytorch.py \
    "${common_args[@]}"
fi

exec "$python_bin" scripts/train_pytorch.py "${common_args[@]}"
