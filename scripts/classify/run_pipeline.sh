#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Project root: $ROOT"

# Train all supervised classifiers (linear probe, LoRA, EfficientNet, fusion)
# bash scripts/classify/run_train_all.sh

# Infer trained checkpoints -> classification_results/
# python scripts/classify/infer_trained.py --preset all

# Evaluate existing JSON outputs only
python scripts/classify/run_evaluate.py --preset all
