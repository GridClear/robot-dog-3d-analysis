#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/uvicorn app.main:app \
  --host "${SPLAT_HOST:-0.0.0.0}" \
  --port "${SPLAT_PORT:-8090}"
