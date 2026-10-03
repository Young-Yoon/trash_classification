"""Shared path configuration for EgoWaste / trash_classification pipelines.

Paths resolve from environment variables when set:

  EGOWASTE_ROOT   Dataset + optional sibling assets root
  CSIRE_CODE_ROOT Optional fallback for an existing csire_code checkout
  SAM3_ROOT       SAM3 install directory (weights + BPE)

Raw detection masks, candidate crops, audio, and frame JSONs are *not*
shipped in this repo; point the env vars at a local data tree.
"""

from __future__ import annotations

import os
from pathlib import Path

# trash_classification/ (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_CSIRE = PROJECT_ROOT.parent / "csire_code"
CSIRE_CODE_ROOT = Path(os.environ.get("CSIRE_CODE_ROOT", str(_DEFAULT_CSIRE))).expanduser()

_EGOWASTE_ENV = os.environ.get("EGOWASTE_ROOT") or os.environ.get("CSIRE_DATA_ROOT")
if _EGOWASTE_ENV:
    EGOWASTE_ROOT = Path(_EGOWASTE_ENV).expanduser()
elif (CSIRE_CODE_ROOT / "camera1_plastics_shareable_dataset").is_dir():
    EGOWASTE_ROOT = CSIRE_CODE_ROOT
elif (PROJECT_ROOT.parent / "camera1_plastics_shareable_dataset").is_dir():
    EGOWASTE_ROOT = PROJECT_ROOT.parent
else:
    EGOWASTE_ROOT = PROJECT_ROOT

DATASET_META_ROOT = EGOWASTE_ROOT / "camera1_plastics_shareable_dataset"
# Frame/mask assets may live next to the workspace even when meta CSVs are under csire_code.
_ASSET_CANDIDATES = [
    Path(os.environ["EGOWASTE_ASSET_ROOT"]).expanduser()
    if os.environ.get("EGOWASTE_ASSET_ROOT")
    else None,
    PROJECT_ROOT.parent / "camera1_plastics_shareable_dataset",
    DATASET_META_ROOT,
]
DATASET_ASSET_ROOT = next(
    (p for p in _ASSET_CANDIDATES if p is not None and p.is_dir()),
    DATASET_META_ROOT,
)
DATASET_ROOT = DATASET_META_ROOT

AUDIO_DIR = EGOWASTE_ROOT / "audio"
AUDIO_CLEANED_DIR = AUDIO_DIR / "cleaned"
TRANSCRIBE_DIR = EGOWASTE_ROOT / "transcribe"
LORA_CHECKPOINT_DIR = EGOWASTE_ROOT / "lora_whisper_checkpoints"
LORA_DENOISED_CHECKPOINT_DIR = EGOWASTE_ROOT / "lora_whisper_checkpoints_denoised"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARIES_DIR = PROJECT_ROOT / "summaries"

META2_CSV = DATASET_ROOT / "camera1_plastics_meta2.csv"
TIMESTAMP_GROUPS_CSV = DATASET_ROOT / "camera1_plastics_timestamp_groups.csv"
COMMON_PHRASES_CSV = DATASET_ROOT / "camera1_plastics_common_phrases.csv"
TAXONOMY_JSON = DATASET_ROOT / "camera1_plastics_taxonomy.json"
VIDEO_CLIPS_DIR = DATASET_ROOT / "audio_video_clips"

TRANSCRIBE_V1_DIR = TRANSCRIBE_DIR / "v1"
TRANSCRIBE_V2_DIR = TRANSCRIBE_DIR / "v2"
TRANSCRIBE_LORA_DIR = TRANSCRIBE_DIR / "lora"
TRANSCRIBE_V1_DENOISED_DIR = TRANSCRIBE_DIR / "v1_denoised"
TRANSCRIBE_V2_DENOISED_DIR = TRANSCRIBE_DIR / "v2_denoised"
TRANSCRIBE_LORA_DENOISED_DIR = TRANSCRIBE_DIR / "lora_denoised"
TRANSCRIBE_LORA_DENOISED_TRAINED_DIR = TRANSCRIBE_DIR / "lora_denoised_trained"

MANIFEST_V1 = TRANSCRIBE_DIR / "manifest_v1.json"
MANIFEST_V2 = TRANSCRIBE_DIR / "manifest_v2.json"
MANIFEST_LORA = TRANSCRIBE_DIR / "manifest_lora.json"
MANIFEST_V1_DENOISED = TRANSCRIBE_DIR / "manifest_v1_denoised.json"
MANIFEST_V2_DENOISED = TRANSCRIBE_DIR / "manifest_v2_denoised.json"
MANIFEST_LORA_DENOISED = TRANSCRIBE_DIR / "manifest_lora_denoised.json"
MANIFEST_LORA_DENOISED_TRAINED = TRANSCRIBE_DIR / "manifest_lora_denoised_trained.json"

TRAINING_METADATA_CSV = EGOWASTE_ROOT / "training_metadata.csv"
TRAINING_METADATA_DENOISED_CSV = EGOWASTE_ROOT / "training_metadata_denoised.csv"
EVAL_SUMMARY_CSV = RESULTS_DIR / "evaluation_summary.csv"

# SAM3 segmentation outputs (local / gitignored)
SEGMENT_DIR = EGOWASTE_ROOT / "segment" / "sam3_whole"
SEGMENT_V1_DIR = SEGMENT_DIR / "v1_individual_results"
SEGMENT_V1_HAND_GLOVE_DIR = SEGMENT_DIR / "v1_hand_glove_individual_results"
SEGMENT_V2_DIR = SEGMENT_DIR / "v2_individual_results"
SEGMENT_IOU_V1_CSV = SEGMENT_DIR / "iou_metrics_v1.csv"
SEGMENT_IOU_V1_HAND_GLOVE_CSV = SEGMENT_DIR / "iou_metrics_v1_hand_glove.csv"
SEGMENT_IOU_V2_CSV = SEGMENT_DIR / "iou_metrics_v2.csv"
SEGMENT_EVAL_SUMMARY_CSV = SEGMENT_DIR / "segment_evaluation_summary.csv"
SEGMENT_RESULTS_V1_PKL = SEGMENT_DIR / "detection_results_v1.pkl"
SEGMENT_RESULTS_V2_PKL = SEGMENT_DIR / "detection_results_v2.pkl"

CLASSIFICATION_RESULTS_DIR = EGOWASTE_ROOT / "classification_results"
CLASSIFICATION_SUMMARY_CSV = RESULTS_DIR / "classification_summary.csv"
CLASSIFICATION_SEGMENT_SUMMARY_CSV = RESULTS_DIR / "classification_segment_summary.csv"
CLASSIFICATION_SEGMENT_ZEROSHOT_SUMMARY_CSV = RESULTS_DIR / "classification_segment_zeroshot_summary.csv"
CLASSIFICATION_COMPARISON_CSV = RESULTS_DIR / "classification_comparison.csv"
CLASSIFY_CHECKPOINT_DIR = EGOWASTE_ROOT / "classify_checkpoints"
SEGMENT_PREDICTED_MASKS_V1_DIR = SEGMENT_DIR / "predicted_masks_v1"
SEGMENT_PREDICTED_MASKS_V2_DIR = SEGMENT_DIR / "predicted_masks_v2"
SEGMENT_PREDICTED_MASKS_V3_DIR = SEGMENT_DIR / "predicted_masks_v3"
SEGMENT_PREDICTED_MASKS_V4_DIR = SEGMENT_DIR / "predicted_masks_v4"
SEGMENT_PREDICTED_MASKS_V5_DIR = SEGMENT_DIR / "predicted_masks_v5"
SEGMENT_PREDICTED_MASKS_V6_DIR = SEGMENT_DIR / "predicted_masks_v6"

WHISPER_MODEL_V1 = "base"
WHISPER_MODEL_V2 = "large"
WHISPER_MODEL_LORA = "openai/whisper-large-v2"

# Legacy absolute prefixes stored in CSV metadata
LEGACY_EXTERNAL_ROOT = "/media/jisha/E/trash_dataset/"
LEGACY_COLAB_PREFIX = "drive/MyDrive/"

# SAM3 install (override with SAM3_ROOT env var or --sam3-root)
SAM3_DEFAULT_ROOT = Path(os.environ.get("SAM3_ROOT", str(Path.home() / "sam3" / "sam3"))).expanduser()
SAM3_BPE_FILENAME = "assets/bpe_simple_vocab_16e6.txt.gz"
