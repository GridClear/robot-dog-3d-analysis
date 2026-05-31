#!/usr/bin/env bash
# Quick VGGT pose smoke test (needs SPLAT_INSTALL_GPU=1 and frames).
set -euo pipefail
cd "$(dirname "$0")/.."
FRAMES="${SPLAT_RECON_IMAGES:-}"
if [ -z "$FRAMES" ]; then
  echo "Set SPLAT_RECON_IMAGES to comma-separated image paths"
  exit 1
fi
export SPLAT_DATA_DIR="${SPLAT_DATA_DIR:-/tmp/splat_smoke}"
./.venv/bin/python - <<'PY'
import os
from pathlib import Path
from app.services.pose.vggt import VGGTPoseBackend

frames = [Path(p) for p in os.environ["SPLAT_RECON_IMAGES"].split(",") if Path(p).exists()]
work = Path(os.environ["SPLAT_DATA_DIR"]) / "smoke"
r = VGGTPoseBackend().estimate(frames, work)
print(r)
assert r.available, r.note
PY
