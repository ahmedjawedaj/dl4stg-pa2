#!/usr/bin/env bash
# Runs the whole experiment in the order the handout requires, unattended.
# Phase 2 starts only after answers/Q2_5.md is committed, phase 3 only after answers/Q2_11.md.
# The optional D+B run starts only if logs/db_decision.txt says "yes" (written after Table 5).
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
echo "waiting for logs/db_decision.txt"
until [ -f logs/db_decision.txt ]; do sleep 60; done
if grep -qi '^yes' logs/db_decision.txt; then ./run_phase3.sh --db; fi
echo "ALL PHASES DONE"
