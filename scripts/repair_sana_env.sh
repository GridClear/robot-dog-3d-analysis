#!/usr/bin/env bash
# Fix a partial upstream install (xformers failed → editable install never finished).
set -euo pipefail
exec "$(dirname "$0")/sana_env_install.sh" "${1:-sana}"
