#!/usr/bin/env bash
# API venv for the world-model streaming service (SANA-WM runs in separate conda env).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt

echo ">>> API venv ready. Install SANA-WM with: ./scripts/setup_sana_wm.sh"
