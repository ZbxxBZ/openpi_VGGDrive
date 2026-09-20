# syntax=docker/dockerfile:1.7

# Blackwell GPUs require a CUDA 12.8 build of PyTorch. The project lock currently
# resolves the PyPI CUDA 12.6 build, so install the matching cu128 wheels last and
# run the environment without a runtime uv re-sync.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    UV_NO_SYNC=1 \
    PATH=/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN UV_PYTHON_INSTALL_MIRROR=https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download \
    uv venv --python 3.11.9 "$UV_PROJECT_ENVIRONMENT"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    uv sync --frozen --no-install-project --no-dev

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_NO_SYNC=0 uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --reinstall \
        torch==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)" \
    | xargs -I{} cp -r /tmp/transformers_replace/. {} \
    && rm -rf /tmp/transformers_replace \
    && python -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda; print(torch.__version__, torch.version.cuda)"

CMD ["sleep", "infinity"]
