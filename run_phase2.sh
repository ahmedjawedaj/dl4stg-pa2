#!/usr/bin/env bash
# Phase 2: stress tests of model A. Refuses to run unless answers/Q2_5.md is committed unchanged.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -n "$(git log -1 --format=%H -- answers/Q2_5.md)" ] || { echo "Commit answers/Q2_5.md first"; exit 1; }
git diff --quiet HEAD -- answers/Q2_5.md || { echo "answers/Q2_5.md has uncommitted edits"; exit 1; }
mkdir -p logs
echo "Q2.5 committed at $(git log -1 --format=%cI -- answers/Q2_5.md) in $(git log -1 --format=%H -- answers/Q2_5.md), phase 2 started at $(date +%Y-%m-%dT%H:%M:%S%z)" \
  | tee logs/phase2_gate.txt
$PY -m scripts.phase2_stress 2>&1 | tee logs/phase2.txt
echo "PHASE 2 DONE"
