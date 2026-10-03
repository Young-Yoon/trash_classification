#!/usr/bin/env bash
# High-level repro entrypoints for v1–v16.
# Heavy steps (SAM3 detect, LoRA train, classifier train) need GPU + local data.
# Summary rebuild works from shipped CSVs + optional meta CSVs only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/scripts/shared:${ROOT}/scripts/classify:${ROOT}/scripts/segment:${ROOT}/configs:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage: ./run_repro.sh <command>

Commands:
  summaries     Rebuild summaries/*.csv from meta + shipped tables (CPU)
  whisper-eval  Evaluate existing Whisper manifests (needs EGOWASTE_ROOT)
  classify-eval Evaluate existing classification JSON (needs EGOWASTE_ROOT)
  segment-iou   Evaluate existing SAM detection pickles (needs EGOWASTE_ROOT)
  help          Show this message

GPU / training (run manually after setting EGOWASTE_ROOT and SAM3_ROOT):
  bash scripts/segment/run_pipeline.sh
  bash scripts/whisper/run_pipeline.sh
  bash scripts/classify/run_pipeline.sh
  python scripts/classify/run_segment_classify_eval.py --help
EOF
}

cmd="${1:-help}"
case "$cmd" in
  summaries)
    python3 analysis/build_summaries.py
    ;;
  whisper-eval)
    python3 scripts/whisper/evaluate.py --version all --save-details
    ;;
  classify-eval)
    python3 scripts/classify/run_evaluate.py --preset all
    ;;
  segment-iou)
    python3 scripts/segment/evaluate_iou.py --version both
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
