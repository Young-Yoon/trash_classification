"""Canonical segment / rerank version registry (v1–v16).

Single source of truth for documentation and analysis. Runtime selection
logic lives in scripts/classify/segment_masks.py and must stay in sync.
"""

from __future__ import annotations

from typing import Literal

SegmentVersion = Literal[
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
]

ALL_SEGMENT_VERSIONS: tuple[SegmentVersion, ...] = tuple(  # type: ignore[assignment]
    f"v{i}" for i in range(1, 17)
)

PHASES: dict[str, tuple[SegmentVersion, ...]] = {
    "Phase1_Backbone": ("v1", "v2"),
    "Phase2_MaskSelect": ("v3", "v4", "v5", "v6", "v8"),
    "Phase3_RerankPool": ("v7", "v9", "v10"),
    "Phase4_RerankStrategy": ("v11", "v12", "v13", "v14", "v15", "v16"),
}

RERANK_STRATEGIES: dict[str, str] = {
    "v7": "confidence",
    "v9": "confidence",
    "v10": "confidence",
    "v11": "margin",
    "v12": "margin",
    "v13": "composite",
    "v14": "composite_margin",
    "v15": "text_aware",
    "v16": "geo_composite",
}

CROP_MODES: dict[str, str] = {
    "v1": "bbox",
    "v2": "bbox",
    "v3": "bbox",
    "v4": "masked",
    "v5": "bbox",
    "v6": "masked",
    "v7": "bbox",
    "v8": "bbox",
    "v9": "bbox",
    "v10": "bbox",
    "v11": "bbox",
    "v12": "bbox",
    "v13": "bbox",
    "v14": "bbox",
    "v15": "bbox",
    "v16": "bbox",
}

BBOX_PADDING: dict[str, float] = {
    "v1": 0.0,
    "v2": 0.0,
    "v3": 0.0,
    "v4": 0.0,
    "v5": 0.0,
    "v6": 0.0,
    "v7": 0.0,
    "v8": 0.15,
    "v9": 0.0,
    "v10": 0.0,
    "v11": 0.0,
    "v12": 0.0,
    "v13": 0.0,
    "v14": 0.0,
    "v15": 0.0,
    "v16": 0.0,
}

SEGMENT_VERSION_DESCRIPTIONS: dict[str, str] = {
    "v1": "Hand-guided ROI; select in-hand max containment, else largest bbox",
    "v2": "Full-image plastic detection; SAM score Top-1",
    "v3": "v1 detections; SAM score Top-1 from all plastic candidates",
    "v4": "Same masks as v3; pixel masking crop at classification time",
    "v5": "v1 in-hand score Top-1, else v2 full-image Top-1 fallback",
    "v6": "Same masks as v5; pixel masking crop at classification time",
    "v7": "Top-5 score candidates from v1; classifier rerank by confidence",
    "v8": "Same masks as v5; bbox crop with 15% padding",
    "v9": "v5 candidate pool (in-hand + v2 + score-ranked); Top-5 rerank",
    "v10": "In-hand Top-5 score candidates; classifier confidence rerank",
    "v11": "Same candidate pool as v7; margin (top1-top2) rerank",
    "v12": "Same candidate pool as v10; margin rerank",
    "v13": "Same pool as v7; composite rerank (0.7 cls conf + 0.3 SAM score)",
    "v14": "Same pool as v7; composite margin rerank (0.7 margin + 0.3 SAM score)",
    "v15": "Same pool as v7; margin rerank + Whisper category match bonus",
    "v16": "Same pool as v7; geo composite (0.5 margin + 0.3 SAM + 0.2 hand IoU)",
}

# Reported paper numbers (group any-frame final-prob Top-1, speech-aligned).
PAPER_GROUP_ANY_FRAME = {
    "v15_fusion_en": 73.4,
    "v15_efficientnet": 69.6,
    "paper_mask_iou_v1_pct": 8.3,
    "paper_mask_iou_v2_pct": 19.8,
}

REPORTED_FINAL_VERSION = "v15"
