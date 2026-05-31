#!/usr/bin/env bash
# Clone NVlabs/Sana and create conda env for SANA-WM inference.
set -euo pipefail
cd "$(dirname "$0")/.."

SANA_DIR="third_party/Sana"
if [ ! -d "$SANA_DIR/.git" ]; then
  mkdir -p third_party
  git clone --depth=1 https://github.com/NVlabs/Sana.git "$SANA_DIR"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda required."
  echo "  Install Miniconda: bash miniconda.sh -b -p \$HOME/miniconda3"
  echo "  Then: eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\""
  exit 1
fi

bash ./scripts/sana_env_install.sh sana

echo ""
echo ">>> Smoke test (fast, no refiner):"
echo "conda activate sana"
echo "cd third_party/Sana"
echo "DISABLE_XFORMERS=1 python inference_video_scripts/inference_sana_wm.py \\"
echo "  --image asset/sana_wm/demo_0.png \\"
echo "  --prompt asset/sana_wm/demo_0.txt \\"
echo "  --action 'w-60,w-60' --num_frames 21 --no_refiner --step 10 \\"
echo "  --output_dir /tmp/sana_wm_smoke"

if [ -f data/benchmark.json.example ] && [ ! -f data/benchmark.json ]; then
  cp data/benchmark.json.example data/benchmark.json
fi
