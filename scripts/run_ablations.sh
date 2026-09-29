#!/usr/bin/env bash
# Ablations. Each one writes its own config into configs/ and its own run
# directory under artifacts/, so nothing overwrites your baseline.
#
# Signal ablations rerun stage 1 because the feature matrix is built there.
# If you want them cheap, extend signals.stack_features to always save all
# three blocks and slice columns in train_gate instead. That is a good first
# task to hand your coding agent.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

BASE=configs/default.yaml
BASE_RUN=$(python - "$BASE" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["run_name"])
PY
)

# ---- signal ablations --------------------------------------------------------
for FEATURES in "hidden" "p_yes" "entropy" "hidden,p_yes"; do
  NAME="${BASE_RUN}-feat-${FEATURES//,/+}"
  echo "== signal ablation: $FEATURES =="
  python - "$BASE" "$NAME" "$FEATURES" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["run_name"] = sys.argv[2]
cfg["gate"]["features"] = sys.argv[3].split(",")
yaml.safe_dump(cfg, open(f"configs/{sys.argv[2]}.yaml", "w"), sort_keys=False)
PY
  bash scripts/run_baseline.sh "configs/$NAME.yaml"
done

# ---- expensive ablations: new backbone, new stage 1 --------------------------
for MODEL in "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" \
             "HuggingFaceTB/SmolLM2-1.7B-Instruct"; do
  SHORT=$(basename "$MODEL")
  NAME="scale-${SHORT}"
  echo "== backbone ablation: $MODEL =="
  python - "$BASE" "$NAME" "$MODEL" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["run_name"] = sys.argv[2]
cfg["model"]["name"] = sys.argv[3]
yaml.safe_dump(cfg, open(f"configs/{sys.argv[2]}.yaml", "w"), sort_keys=False)
PY
  bash scripts/run_baseline.sh "configs/$NAME.yaml"
done

echo "done. collect artifacts/*/results.md into experiments/log.md"
