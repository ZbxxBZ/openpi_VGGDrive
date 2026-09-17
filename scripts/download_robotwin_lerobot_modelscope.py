#!/usr/bin/env python3
"""Download lerobot/robotwin_unified from ModelScope.

Default destination is on the 200G overlay disk, not /mnt (only ~26G free).
Usage:
  /root/miniconda3/bin/python /tmp/download_robotwin_lerobot_modelscope.py
  /root/miniconda3/bin/python /tmp/download_robotwin_lerobot_modelscope.py --local-dir /root/data/robotwin_unified
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_ID = "lerobot/robotwin_unified"
DEFAULT_DIR = Path("/root/data/robotwin_unified")
EXPECTED_BYTES = 79_475_205_476
MIN_FREE_BYTES = 90 * 1024**3  # 90 GiB headroom for download + metadata


def _free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _looks_complete(dest: Path) -> bool:
    info = dest / "meta" / "info.json"
    videos = dest / "videos"
    data = dest / "data"
    if not (info.is_file() and videos.is_dir() and data.is_dir()):
        return False
    size = _dir_size(dest)
    # Allow a little slack vs advertised 79.5GB.
    return size >= int(EXPECTED_BYTES * 0.98)


def _download_with_sdk(dest: Path) -> None:
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    print(f"SDK download {DATASET_ID} -> {dest}", flush=True)
    dataset_snapshot_download(
        dataset_id=DATASET_ID,
        local_dir=str(dest),
        revision="master",
    )


def _download_with_cli(dest: Path) -> None:
    python = sys.executable
    cmd = [
        python,
        "-m",
        "modelscope",
        "download",
        "--dataset",
        DATASET_ID,
        "--local_dir",
        str(dest),
    ]
    print("CLI", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if meta/info.json already exists.",
    )
    args = parser.parse_args()
    dest: Path = args.local_dir
    dest.mkdir(parents=True, exist_ok=True)

    free = _free_bytes(dest)
    print(f"dest={dest}", flush=True)
    print(f"free={free / 1024**3:.1f} GiB  need>={MIN_FREE_BYTES / 1024**3:.0f} GiB", flush=True)
    if free < MIN_FREE_BYTES and not _looks_complete(dest):
        print(
            "Not enough free space. Use overlay disk, e.g. --local-dir /root/data/robotwin_unified. "
            "/mnt is too small for the full 79.5GB dataset.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if _looks_complete(dest) and not args.force:
        print(f"SKIP already complete {dest} ({_dir_size(dest)} bytes)", flush=True)
        return 0

    try:
        _download_with_sdk(dest)
    except Exception as exc:
        print(f"SDK failed: {exc!r}; trying CLI", flush=True)
        _download_with_cli(dest)

    size = _dir_size(dest)
    print(f"downloaded size={size} bytes ({size / 1e9:.2f} GB)", flush=True)
    info = dest / "meta" / "info.json"
    if not info.is_file():
        print(f"MISSING {info}", file=sys.stderr, flush=True)
        return 1
    print(info.read_text(encoding="utf-8")[:800], flush=True)
    if size < int(EXPECTED_BYTES * 0.98):
        print("WARNING: size looks incomplete", file=sys.stderr, flush=True)
        return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
