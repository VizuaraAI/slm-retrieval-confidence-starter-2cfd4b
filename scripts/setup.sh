#!/usr/bin/env bash
# One-time setup. Creates a virtualenv, installs dependencies, and checks that
# the package imports.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH="$PWD/src"
python -c "import precog; print('precog', precog.__version__, 'imports cleanly')"

echo
echo "Setup done. Activate with:  source .venv/bin/activate"
echo "Then run:                   bash scripts/run_smoke.sh"
