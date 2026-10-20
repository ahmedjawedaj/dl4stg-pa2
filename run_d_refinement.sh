#!/usr/bin/env bash
# Additional D candidate. Stops after validation selection, before test evaluation.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
for file in answers/Q2_11.md experiments/d_refinement_plan.md; do
  [ -n "$(git log -1 --format=%H -- "$file")" ] || { echo "Commit $file first"; exit 1; }
  git diff --quiet HEAD -- "$file" || { echo "$file has uncommitted edits"; exit 1; }
done
mkdir -p logs
if [ ! -f logs/d_refinement_gate.txt ]; then
  echo "Additional experiment started at $(date +%Y-%m-%dT%H:%M:%S%z), commit $(git rev-parse HEAD)" \
    | tee logs/d_refinement_gate.txt
fi
$PY -m scripts.reach_check D_dil64 | tee logs/reach_D_dil64.txt
for seed in 0 1; do
  if [ ! -f "checkpoints/D_D_dil64_s${seed}.pt" ]; then
    $PY -m pa2.train --model D --arch D_dil64 --seed "$seed" --epochs 30 --val-speeds \
      2>&1 | tee -a "logs/train_D_D_dil64_s${seed}.txt"
  fi
done
$PY -m scripts.select_d | tee logs/select_d_refinement.txt
echo "D REFINEMENT VALIDATION COMPLETE: inspect selection before test evaluation"
