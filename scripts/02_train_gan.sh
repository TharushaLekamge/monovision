#!/usr/bin/env bash
# Usage: scripts/02_train_gan.sh configs/gan/pt_augan_bdd100k.yaml [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m monovision.enhance.train --config "${1:-configs/gan/pt_augan_bdd100k.yaml}" "${@:2}"
