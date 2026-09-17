#!/usr/bin/env bash
set -euo pipefail
LOG=/tmp/setup_stage1_env.log
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is)"

export PATH="$HOME/.local/bin:$PATH"
WORK=/root/work
OPENPI="$WORK/openpi_VGGDrive/openpi"

if ! command -v uv >/dev/null 2>&1; then
  echo "=== install uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if [[ ! -d $WORK/vggt-omega/.git ]]; then
  echo "=== clone vggt-omega ==="
  if ! git clone --depth 1 https://github.com/facebookresearch/vggt-omega.git "$WORK/vggt-omega"; then
    echo "github clone failed, trying ghproxy"
    git clone --depth 1 https://ghproxy.net/https://github.com/facebookresearch/vggt-omega.git "$WORK/vggt-omega"
  fi
fi
# Pin if the commit exists in this shallow clone; otherwise keep HEAD.
git -C "$WORK/vggt-omega" fetch --depth 1 origin b2c61f6631d9f344a2d914bfba5d9529d6fc1d35 2>/dev/null || true
git -C "$WORK/vggt-omega" checkout b2c61f6631d9f344a2d914bfba5d9529d6fc1d35 2>/dev/null || git -C "$WORK/vggt-omega" log -1 --oneline

echo "=== uv sync ==="
cd "$OPENPI"
export GIT_LFS_SKIP_SMUDGE=1
# Prefer Tsinghua for generic wheels; torch CUDA wheels stay on the lockfile index.
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
uv python install 3.11 || true
uv sync
echo "=== transformers_replace ==="
uv run python - <<'PY'
from pathlib import Path
from shutil import copytree
import transformers
src = Path("src/openpi/models_pytorch/transformers_replace")
dst = Path(transformers.__file__).parent
print("copy", src, "->", dst)
copytree(src, dst, dirs_exist_ok=True)
print("transformers", transformers.__version__, transformers.__file__)
PY

echo "=== torch cuda ==="
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "mem", round(torch.cuda.get_device_properties(0).total_mem/1024**3,1), "GB")
PY

echo "DONE $(date -Is)"
