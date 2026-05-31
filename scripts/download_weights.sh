#!/usr/bin/env bash
# Optional: prefetch SANA-WM weights via a minimal conda inference (requires setup first).
set -euo pipefail
cd "$(dirname "$0")/.."

SANA_DIR="third_party/Sana"
if [ ! -f "$SANA_DIR/inference_video_scripts/inference_sana_wm.py" ]; then
  echo "Run ./scripts/setup_sana_wm.sh first"
  exit 1
fi

echo "Weights download on first inference from HuggingFace (~96GB)."
echo "Running minimal smoke to trigger cache..."
conda run -n sana python "$SANA_DIR/inference_video_scripts/inference_sana_wm.py" \
  --image "$SANA_DIR/asset/sana_wm/demo_0.png" \
  --prompt "$SANA_DIR/asset/sana_wm/demo_0.txt" \
  --action "none-20" \
  --num_frames 21 \
  --no_refiner \
  --step 10 \
  --output_dir /tmp/sana_wm_prefetch
