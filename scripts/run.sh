#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [[ -x "$HOME/miniconda3/envs/local-rag-app/bin/python" ]]; then
    PYTHON="$HOME/miniconda3/envs/local-rag-app/bin/python"
fi

exec "$PYTHON" -m app.main "$@"
