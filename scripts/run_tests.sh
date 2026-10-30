#!/usr/bin/env bash
# Two separate pytest processes: running the TF and PyTorch suites in one process crashed
# natively in our test environment (TF 2.21 + torch 2.14), so they are kept apart.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests/test_port.py
if python -c "import tensorflow, tf_slim" 2>/dev/null; then
  python -m pytest -q tests/test_parity_tf.py
else
  echo "[skip] TF parity tests: pip install -r requirements-dev.txt"
fi
