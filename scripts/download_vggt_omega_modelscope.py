#!/usr/bin/env python3
"""Download VGGT-Omega checkpoints from ModelScope into /mnt/ckpts."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEST = Path("/mnt/ckpts")
WORKDIR = Path("/root/weights/VGGT-Omega")
FILES = [
    "vggt_omega_1b_416_reproduce.pt",  # default for this repo
    "vggt_omega_1b_512.pt",
    "vggt_omega_1b_256_text.pt",
]
MODEL_ID = "facebook/VGGT-Omega"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.file_download import model_file_download

    for name in FILES:
        dest_path = DEST / name
        if dest_path.exists() and dest_path.stat().st_size > 1_000_000_000:
            print(f"SKIP already present {dest_path} ({dest_path.stat().st_size} bytes)", flush=True)
            continue
        print(f"DOWNLOAD {name} -> {WORKDIR}", flush=True)
        local = model_file_download(
            model_id=MODEL_ID,
            file_path=name,
            cache_dir=str(WORKDIR.parent),
            revision="master",
        )
        print(f"GOT {local}", flush=True)
        src = Path(local)
        if not src.exists():
            print(f"MISSING {local}", file=sys.stderr, flush=True)
            return 1
        size = src.stat().st_size
        print(f"SIZE {name} {size} bytes", flush=True)
        if size < 1_000_000_000:
            print(f"TOO SMALL {name}", file=sys.stderr, flush=True)
            return 1
        tmp = dest_path.with_suffix(dest_path.suffix + ".partial")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest_path)
        print(f"COPIED {dest_path} ({dest_path.stat().st_size} bytes)", flush=True)

    print("DONE", flush=True)
    for name in FILES:
        path = DEST / name
        print(f"  {path} {path.stat().st_size if path.exists() else 'MISSING'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
