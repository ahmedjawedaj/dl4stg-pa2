#!/usr/bin/env bash
# Create a local virtual environment with the pinned dependencies (macOS/Linux).
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
for c in "$HOME/miniforge3/bin/python3" "$HOME/anaconda3/bin/python3" python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
done
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
mkdir -p logs && .venv/bin/pip freeze > logs/pip_freeze.txt  # exact resolved versions
