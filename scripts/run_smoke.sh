#!/usr/bin/env bash
# The first thing you should run. Tiny model, tiny corpus, tiny splits.
# It proves the three stages are wired together. The numbers are meaningless.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

CONFIG=configs/smoke.yaml

echo "== stage 1: utility labelling =="
python -m precog.label_utility --config "$CONFIG"

echo "== stage 2: fit and calibrate the gate =="
python -m precog.train_gate --config "$CONFIG"

echo "== stage 3: routed evaluation =="
python -m precog.evaluate --config "$CONFIG"

echo
echo "Results table: artifacts/smoke/results.md"
