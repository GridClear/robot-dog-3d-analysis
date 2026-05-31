#!/usr/bin/env bash
# Benchmark SANA-WM on this machine; writes data/benchmark.json with recommended WM_* settings.
#
# First run downloads large HF weights (DiT ~10GB; +refiner ~85GB if BENCHMARK_FULL=1).
# Default order runs fast configs first. Use BENCHMARK_FULL=1 to include refiner.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SANA_DIR="$ROOT/third_party/Sana"
OUT="data/benchmark.json"
mkdir -p data

BENCH_FRAMES="${BENCH_FRAMES:-41}"
INCLUDE_FULL="${BENCHMARK_FULL:-0}"

if [ ! -f "$SANA_DIR/inference_video_scripts/inference_sana_wm.py" ]; then
  echo "Run ./scripts/setup_sana_wm.sh first"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH"
  exit 1
fi

eval "$(conda shell.bash hook)"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-}"
set +u
conda activate sana 2>/dev/null || {
  echo "ERROR: conda env 'sana' missing. Run ./scripts/repair_sana_env.sh"
  exit 1
}
set -u

export PYTHONUNBUFFERED=1
export DISABLE_XFORMERS=1
# Show HuggingFace download progress when supported
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
WM_CACHE=$(find "$HF_CACHE" -maxdepth 1 -name 'models--Efficient-Large-Model--SANA-WM*' 2>/dev/null | head -1)
echo ">>> HuggingFace cache: ${WM_CACHE:-not downloaded yet}"
if [ -n "$WM_CACHE" ]; then
  du -sh "$WM_CACHE" 2>/dev/null || true
else
  echo ">>> First run will download weights (DiT ~10GB; refiner+Gemma ~85GB extra if BENCHMARK_FULL=1)"
  echo ">>> This can take a long time with little GPU use. Watch: du -sh ~/.cache/huggingface/hub"
fi
echo ">>> Benchmark frames: $BENCH_FRAMES (override with BENCH_FRAMES=161)"
echo ""

if ! (cd "$SANA_DIR" && python - <<'PY'
import os
os.environ.setdefault("DISABLE_XFORMERS", "1")
import imageio.v3
import diffusion.model.nets
print("deps OK")
PY
); then
  echo "ERROR: sana env incomplete. Run: ./scripts/repair_sana_env.sh"
  exit 1
fi

IMAGE="$SANA_DIR/asset/sana_wm/demo_0.png"
PROMPT="$SANA_DIR/asset/sana_wm/demo_0.txt"
INTRINSICS="$SANA_DIR/asset/sana_wm/demo_0_intrinsics.npy"
if [ ! -f "$IMAGE" ]; then
  echo "ERROR: missing $IMAGE"
  exit 1
fi

ACTION="w-40,w-40"
BASE="/tmp/sana_wm_bench_$$"
echo ">>> Logs and outputs: $BASE"
mkdir -p "$BASE"

run_case() {
  local name="$1"
  shift
  local dir="${BASE}/${name}"
  mkdir -p "$dir"
  echo ""
  echo "=== $name ($(date -Iseconds)) ==="
  echo "    args: $*"
  local t0
  t0=$(date +%s.%N)
  if (
    cd "$SANA_DIR"
    stdbuf -oL -eL python -u inference_video_scripts/inference_sana_wm.py \
      --image "$IMAGE" --prompt "$PROMPT" --action "$ACTION" \
      --intrinsics "$INTRINSICS" \
      --output_dir "$dir" "$@" 2>&1 | stdbuf -oL tee "${BASE}/${name}.log"
  ); then
    local t1 elapsed mp4
    t1=$(date +%s.%N)
    elapsed=$(python3 -c "print(round(float('$t1') - float('$t0'), 2))")
    mp4=$(find "$dir" -name '*.mp4' 2>/dev/null | head -1)
    echo ">>> $name done: ${elapsed}s mp4=${mp4:-none}"
    python3 - "$name" "$elapsed" "${mp4:-}" <<'PY'
import json, sys
from pathlib import Path
name, elapsed, mp4 = sys.argv[1], float(sys.argv[2]), sys.argv[3]
path = Path("data/benchmark.json")
data = json.loads(path.read_text()) if path.is_file() else {"runs": []}
data.setdefault("runs", []).append({"name": name, "elapsed_sec": elapsed, "mp4": mp4 or None})
path.write_text(json.dumps(data, indent=2))
PY
  else
    echo ">>> $name FAILED (see ${BASE}/${name}.log)"
  fi
}

python3 -c "import json; json.dump({'runs':[]}, open('$OUT','w'))"

# Fast configs first (--intrinsics skips slow Pi3X download; no refiner skips ~85GB)
run_case smoke --num_frames 17 --no_refiner --step 10 --offload_vae --offload_refiner
run_case fast --num_frames "$BENCH_FRAMES" --no_refiner --step 20 --offload_vae --offload_refiner
run_case no_refiner --num_frames "$BENCH_FRAMES" --no_refiner --step 60 --offload_vae --offload_refiner

if [ "$INCLUDE_FULL" = "1" ]; then
  echo ">>> BENCHMARK_FULL=1: running with LTX-2 refiner (large download + slow)"
  run_case full --num_frames "$BENCH_FRAMES" --step 60 --offload_vae --offload_refiner
else
  echo ">>> Skipping 'full' refiner case (set BENCHMARK_FULL=1 to include)"
fi

python3 - <<'PY'
import json
import math
from pathlib import Path

path = Path("data/benchmark.json")
data = json.loads(path.read_text())
ok = [r for r in data.get("runs", []) if r.get("mp4")]
if not ok:
    data["recommended"] = {
        "note": "all runs failed — see /tmp/sana_wm_bench_*/*.log; check HF download and GB10/pytorch"
    }
    path.write_text(json.dumps(data, indent=2))
    raise SystemExit(1)

best = min(ok, key=lambda r: r["elapsed_sec"])
p95 = sorted(r["elapsed_sec"] for r in ok)[-1]
interval = max(5, math.ceil(p95 * 1.05))
data["recommended"] = {
    "profile": best["name"],
    "interval_sec": interval,
    "use_refiner": best["name"] == "full",
    "inference_step": 20 if best["name"] == "fast" else 60,
    "num_frames": int(__import__("os").environ.get("BENCH_FRAMES", "41")),
    "note": f"Set WM_INTERVAL_SEC={interval} in .env (p95={p95}s)",
}
path.write_text(json.dumps(data, indent=2))
print(json.dumps(data["recommended"], indent=2))
PY

echo "Wrote $OUT"
