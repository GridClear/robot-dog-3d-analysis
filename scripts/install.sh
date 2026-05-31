#!/usr/bin/env bash
# Base + optional GPU deps for the splat service on DGX Spark (GB10, aarch64, CUDA 13.0).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt

if [ "${SPLAT_INSTALL_GPU:-0}" = "1" ]; then
  echo ">>> Installing GPU stack (torch cu130, vggt, gsplat)..."
  ./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
  ./.venv/bin/pip install "git+https://github.com/facebookresearch/vggt.git"
  # pycolmap has no aarch64 wheel on PyPI; COLMAP text fallback is used instead
  ./.venv/bin/pip install pycolmap 2>/dev/null || echo "WARN: pycolmap skipped (no aarch64 wheel); using text COLMAP export"
  ./.venv/bin/pip install 'numpy>=2.0' plyfile
  # gsplat builds CUDA kernels; expect 20-60 min on aarch64
  ./.venv/bin/pip install gsplat
  if [ ! -d vendor/gsplat ]; then
    git clone --depth=1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat vendor/gsplat
  fi
  if [ -f vendor/gsplat/examples/requirements.txt ]; then
    ./.venv/bin/pip install -r vendor/gsplat/examples/requirements.txt || true
  fi
  ./.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
fi

command -v ffmpeg >/dev/null || echo "WARN: install ffmpeg for video ingest (sudo apt install ffmpeg)"
