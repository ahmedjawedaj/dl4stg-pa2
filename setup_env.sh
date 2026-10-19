#!/usr/bin/env bash
# Create a local virtual environment and record its resolved dependencies.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
for c in "$HOME/miniforge3/bin/python3" "$HOME/anaconda3/bin/python3" python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
done
"$PY" -m venv .venv
# Ignore machine-wide indexes, including obsolete CUDA/NVIDIA mirrors on this Mac.
# This affects these installer commands only, not the user's global pip settings.
PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pip install --index-url https://pypi.org/simple --upgrade pip -q
PIP_CONFIG_FILE=/dev/null .venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
mkdir -p logs && .venv/bin/pip freeze > logs/pip_freeze.txt  # exact resolved versions
