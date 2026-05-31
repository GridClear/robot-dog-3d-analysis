#!/usr/bin/env bash
# SANA-WM conda env install with aarch64 (DGX Spark) workarounds.
# Upstream environment_setup.sh pins xformers==0.0.33.post2 which has no aarch64 wheel;
# SANA-WM inference sets DISABLE_XFORMERS=1 anyway.
set -euo pipefail

CONDA_ENV="${1:-sana}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SANA_DIR="$ROOT/third_party/Sana"
ARCH="$(uname -m)"

if [ ! -d "$SANA_DIR" ]; then
  echo "ERROR: $SANA_DIR missing. Run ./scripts/setup_sana_wm.sh first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH. Install Miniconda and: eval \"\$(conda shell.bash hook)\""
  exit 1
fi

eval "$(conda shell.bash hook)"

# conda cuda-nvcc activate.d scripts reference these; set -u treats unset as fatal.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-}"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[sana] reusing conda env '$CONDA_ENV'"
else
  echo "[sana] creating conda env '$CONDA_ENV' (python 3.11)"
  set +u
  conda create -n "$CONDA_ENV" python=3.11 -y
  set -u
fi
set +u
conda activate "$CONDA_ENV"
set -u

# CUDA toolkit for building flash-attn / mmcv when needed
set +u
if conda install -c nvidia cuda-toolkit=12.8 -y 2>/dev/null; then
  echo "[sana] cuda-toolkit 12.8 installed"
else
  echo "[sana] WARN: cuda-toolkit 12.8 install failed; continuing (torch bundles CUDA libs)"
fi
set -u

pip install -U pip wheel
pip install "setuptools<80"

TORCH_INDEX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
echo "[sana] installing torch from $TORCH_INDEX"
pip install --upgrade --index-url "$TORCH_INDEX" \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

if [ "$ARCH" = "aarch64" ]; then
  echo "[sana] aarch64: skipping xformers (no wheel; inference uses DISABLE_XFORMERS=1)"
else
  pip install --upgrade --index-url "$TORCH_INDEX" xformers==0.0.33.post2 || {
    echo "[sana] WARN: xformers install failed; continuing with SDPA fallback"
  }
fi

echo "[sana] installing mmcv"
pip install --no-build-isolation mmcv==1.7.2 || {
  echo "[sana] ERROR: mmcv build failed"
  exit 1
}

echo "[sana] editable install (pyproject without xformers pin on aarch64)"
pushd "$SANA_DIR" >/dev/null
PYPROJECT_BACKUP=""
if [ "$ARCH" = "aarch64" ]; then
  PYPROJECT_BACKUP="$(mktemp)"
  cp pyproject.toml "$PYPROJECT_BACKUP"
  grep -v 'xformers==' pyproject.toml > pyproject.toml.tmp
  mv pyproject.toml.tmp pyproject.toml
fi

if ! pip install -e .; then
  [ -n "$PYPROJECT_BACKUP" ] && mv "$PYPROJECT_BACKUP" pyproject.toml
  popd >/dev/null
  echo "[sana] ERROR: pip install -e . failed"
  exit 1
fi

if [ -n "$PYPROJECT_BACKUP" ]; then
  mv "$PYPROJECT_BACKUP" pyproject.toml
fi
popd >/dev/null

echo "[sana] Pi3X (intrinsics)"
pip install git+https://github.com/yyfz/Pi3.git --no-deps || {
  echo "[sana] WARN: Pi3 install failed; intrinsics estimation may not work"
}

if [ "$ARCH" = "aarch64" ]; then
  echo "[sana] aarch64: skipping flash-attn compile (optional for SANA-WM bidirectional)"
else
  echo "[sana] building flash-attn (may take a while)"
  MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-2} \
    pip install --no-build-isolation "flash-attn>=2.7.0" || {
      echo "[sana] WARN: flash-attn build failed; continuing"
    }
fi

echo "[sana] verifying imports"
(
  cd "$SANA_DIR"
  python - <<'PY'
import os
os.environ.setdefault("DISABLE_XFORMERS", "1")
import imageio.v3  # noqa: F401
import diffusion.model.nets  # noqa: F401
print("OK: imageio + diffusion.model.nets")
PY
)

echo ""
echo "[sana] Done. Activate: conda activate $CONDA_ENV"
