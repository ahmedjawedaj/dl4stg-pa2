#!/usr/bin/env bash
# Phase 1: everything allowed BEFORE Q2.5 is written: Table 1, Figure 1, reach, train model A
# (seeds 0 and 1), Table 2 and Figure 2. No mirror, rotation or speed test of model A runs here.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs results
$PY -m scripts.hardware | tee logs/hardware.txt
$PY -m scripts.table1_checks | tee logs/table1.txt
$PY -m scripts.fig1_flow > logs/fig1.txt
$PY -m scripts.reach_check A | tee logs/reach_A.txt
for s in 0 1; do
  [ -f checkpoints/A_s$s.pt ] || $PY -m pa2.train --model A --seed $s 2>&1 | tee logs/train_A_s$s.txt
done
$PY -m scripts.phase1_baseline 2>&1 | tee logs/phase1.txt
echo "PHASE 1 DONE"
