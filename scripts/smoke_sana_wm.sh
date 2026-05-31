#!/usr/bin/env bash
# Quick SANA-WM smoke (~17 frames, no refiner). Use to verify GPU + weights before full benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SANA_DIR="$ROOT/third_party/Sana"
OUT="${1:-/tmp/sana_wm_smoke}"

eval "$(conda shell.bash hook)"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-}"
set +u && conda activate sana && set -u

export PYTHONUNBUFFERED=1 DISABLE_XFORMERS=1

IMAGE="$SANA_DIR/asset/sana_wm/demo_0.png"
PROMPT="$SANA_DIR/asset/sana_wm/demo_0.txt"
INTRINSICS="$SANA_DIR/asset/sana_wm/demo_0_intrinsics.npy"

echo ">>> smoke → $OUT (watch HF cache: du -sh ~/.cache/huggingface/hub)"
mkdir -p "$OUT"
cd "$SANA_DIR"
exec python -u inference_video_scripts/inference_sana_wm.py \
  --image "$IMAGE" --prompt "$PROMPT" --intrinsics "$INTRINSICS" \
  --action "w-30,w-30" --num_frames 17 --no_refiner --step 10 \
  --offload_vae --offload_refiner --output_dir "$OUT"
