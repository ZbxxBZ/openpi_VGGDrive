#!/usr/bin/env bash
set -euo pipefail
LOG=/tmp/copy_mnt_assets.log
exec > >(tee -a "$LOG") 2>&1

echo "START $(date -Is)"
df -h / /mnt

WORK=/root/work
mkdir -p "$WORK/weights/vggt_omega" "$WORK/openpi_VGGDrive"

copy_or_link() {
  local src=$1 dst=$2
  mkdir -p "$(dirname "$dst")"
  if [[ -f $dst && $(stat -c%s "$dst") -gt 1000000000 ]]; then
    echo "SKIP $dst already $(stat -c%s "$dst") bytes"
    return
  fi
  if [[ -f $src ]]; then
    ln -f "$src" "$dst" 2>/dev/null || cp -v "$src" "$dst"
  else
    echo "MISSING $src" >&2
    return 1
  fi
  ls -lh "$dst"
}

echo "=== Omega weights ==="
CACHE=/root/weights/models/facebook--VGGT-Omega/snapshots/master
for f in vggt_omega_1b_416_reproduce.pt vggt_omega_1b_512.pt vggt_omega_1b_256_text.pt; do
  if [[ -f $CACHE/$f ]]; then
    copy_or_link "$CACHE/$f" "$WORK/weights/vggt_omega/$f"
  else
    copy_or_link "/mnt/ckpts/$f" "$WORK/weights/vggt_omega/$f"
  fi
done

echo "=== openpi_VGGDrive.tar ==="
if [[ ! -d $WORK/openpi_VGGDrive/openpi ]]; then
  tar -xf /mnt/openpi_VGGDrive.tar -C "$WORK/openpi_VGGDrive"
fi
ls -la "$WORK/openpi_VGGDrive"

echo "=== RoboTwin.tar ==="
if [[ ! -e $WORK/RoboTwin && ! -d $WORK/RoboTwin ]]; then
  tar -xf /mnt/RoboTwin.tar -C "$WORK"
fi
ls -ld "$WORK"/RoboTwin* 2>/dev/null || true

echo "=== pi05_robotwin2 (no optimizer.pt) ==="
if [[ ! -f $WORK/weights/pi05_robotwin2/model.safetensors ]]; then
  tar -xf /mnt/ckpts/pi05_robotwin2.tar -C "$WORK/weights" \
    pi05_robotwin2/model.safetensors \
    pi05_robotwin2/assets \
    pi05_robotwin2/metadata.pt \
    pi05_robotwin2/README.md
fi
ls -lh "$WORK/weights/pi05_robotwin2/"
find "$WORK/weights/pi05_robotwin2" -type f -printf '%s %p\n'

echo "DONE $(date -Is)"
df -h /
echo "WORK tree:"
du -sh "$WORK"/* 2>/dev/null || true
