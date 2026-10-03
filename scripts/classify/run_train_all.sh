#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Project root: $ROOT"
echo "Training all supervised classifiers..."

python scripts/classify/train.py --method linear_probe --backbone clip --epochs 5 --batch-size 32
python scripts/classify/train.py --method lora --backbone clip --epochs 5 --batch-size 32
python scripts/classify/train.py --method linear_probe --backbone siglip --epochs 5 --batch-size 32
python scripts/classify/train.py --method lora --backbone siglip --epochs 5 --batch-size 32
python scripts/classify/train.py --method efficientnet --epochs 5 --batch-size 32
python scripts/classify/train.py --method fusion

echo "Running inference for all trained presets..."
python scripts/classify/infer_trained.py --preset all

echo "Updating classification summary..."
python scripts/classify/run_evaluate.py --preset all

echo "Done."
