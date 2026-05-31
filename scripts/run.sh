#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/uvicorn app.main:app \
  --host "${WM_HOST:-0.0.0.0}" \
  --port "${WM_PORT:-8090}"
