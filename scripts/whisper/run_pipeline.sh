#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Project root: $ROOT"

# 1) Optional preprocessing: extract audio from video clips
# python scripts/whisper/extract_audio.py

# 2) Build LoRA training metadata
python scripts/whisper/build_training_metadata.py

# 3) Train LoRA adapter (requires GPU + HF dependencies)
# python scripts/whisper/train_lora.py --max-steps 50

# 4) Inference
# python scripts/whisper/transcribe.py --version v1
# python scripts/whisper/transcribe.py --version v2
# python scripts/whisper/transcribe.py --version lora --checkpoint-dir lora_whisper_checkpoints

# 5) Evaluate all available transcription outputs
python scripts/whisper/evaluate.py --version all --save-details
