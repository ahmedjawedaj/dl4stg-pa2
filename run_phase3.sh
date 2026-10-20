#!/usr/bin/env bash
# Phase 3: train B, C and the D candidates (validation only). Tables 5-6, Figure 6.
# Refuses to run unless answers/Q2_11.md is committed unchanged.
# Optional D+B (only if Table 5 supports it):  ./run_phase3.sh --db
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -n "$(git log -1 --format=%H -- answers/Q2_11.md)" ] || { echo "Commit answers/Q2_11.md first"; exit 1; }
git diff --quiet HEAD -- answers/Q2_11.md || { echo "answers/Q2_11.md has uncommitted edits"; exit 1; }
mkdir -p logs
[ -f logs/phase3_gate.txt ] || echo "Q2.11 committed at $(git log -1 --format=%cI -- answers/Q2_11.md) in $(git log -1 --format=%H -- answers/Q2_11.md), phase 3 started at $(date +%Y-%m-%dT%H:%M:%S%z)" \
  | tee logs/phase3_gate.txt

train() {  # train MODEL SEED [ARCH]
  local tag=$1_s$2 extra=()
  if [ -n "${3:-}" ]; then tag=$1_$3_s$2; extra=(--arch "$3"); fi
  [ -f checkpoints/$tag.pt ] || $PY -m pa2.train --model "$1" --seed "$2" ${extra[@]+"${extra[@]}"} 2>&1 | tee logs/train_$tag.txt
}

if [ "${1:-}" != "--db" ]; then
  for s in 0 1; do train B $s; train C $s; done
  # model D design: every candidate is trained exactly like a final model (30 epochs, seeds 0
  # and 1) and compared on the validation split only. The winner's two runs are model D.
  for a in D_k5 D_dil D_p8 D_dil64; do
    for s in 0 1; do
      [ -f checkpoints/D_${a}_s$s.pt ] || $PY -m pa2.train --model D --arch $a --seed $s --val-speeds \
          2>&1 | tee logs/train_D_${a}_s$s.txt
    done
  done
  $PY -m scripts.select_d | tee logs/select_d.txt
fi
DA=$($PY -c "import json; print(json.load(open('results/d_selection.json'))['chosen'])")
$PY -m scripts.reach_check A $DA | tee logs/reach_D.txt
if [ "${1:-}" = "--db" ]; then
  for s in 0 1; do train DB $s $DA; done
  $PY -m scripts.phase3_final --db 2>&1 | tee logs/phase3_db.txt
else
  $PY -m scripts.phase3_final 2>&1 | tee logs/phase3.txt
fi
echo "PHASE 3 DONE"
