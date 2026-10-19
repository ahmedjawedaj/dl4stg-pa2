#!/usr/bin/env bash
# Runs the whole experiment in the order the handout requires.
# Phase 2 starts only after answers/Q2_5.md is committed, phase 3 only after answers/Q2_11.md.
# Waits for the Table 5 decision in results/db_decision.json before optional D+B.
set -euo pipefail
cd "$(dirname "$0")"
committed() { [ -n "$(git log -1 --format=%H -- "$1" 2>/dev/null)" ] && git diff --quiet HEAD -- "$1"; }
./run_phase1.sh
echo "waiting for answers/Q2_5.md to be committed"
until committed answers/Q2_5.md; do sleep 60; done
./run_phase2.sh
echo "waiting for answers/Q2_11.md to be committed"
until committed answers/Q2_11.md; do sleep 60; done
./run_phase3.sh
echo "waiting for results/db_decision.json (decision based on Table 5)"
until [ -f results/db_decision.json ]; do sleep 60; done
DB=$(.venv/bin/python - <<'PY'
import json
with open('results/db_decision.json') as f:
    decision = json.load(f)['train_optional_DB']
if type(decision) is not bool:
    raise ValueError('train_optional_DB must be true or false')
print('yes' if decision else 'no')
PY
)
if [ "$DB" = yes ]; then ./run_phase3.sh --db; fi
echo "ALL PHASES DONE"
