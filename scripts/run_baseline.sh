#!/usr/bin/env bash
# The real baseline run: 1.5B backbone, BM25 over 200k Wikipedia passages,
# NQ-Open plus TriviaQA. Budget a few hours the first time, most of it spent
# in stage 1 building the utility labels. Stages 2 and 3 take seconds after.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

CONFIG="${1:-configs/default.yaml}"

python -m precog.label_utility --config "$CONFIG"
python -m precog.train_gate     --config "$CONFIG"
python -m precog.evaluate       --config "$CONFIG"
