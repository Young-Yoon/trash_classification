"""Load SAM3 segment v1-v12 predicted masks for classification."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config import META2_CSV, SEGMENT_DIR, SEGMENT_V1_DIR, SEGMENT_V2_DIR
from dataset_paths import frame_result_id, resolve_meta_asset_path
from paths import clip_id_from_path
from training_data import ClassificationSample

_V1_RESULTS_DIR: Path = SEGMENT_V1_DIR


def set_v1_results_dir(path: Path | None) -> None:
    global _V1_RESULTS_DIR
    _V1_RESULTS_DIR = path or SEGMENT_V1_DIR

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
CropMode = Literal["bbox", "masked"]
RerankStrategy = Literal[
    "confidence",
    "margin",
    "composite",
    "composite_margin",
    "text_aware",
    "geo_composite",
]
ALL_SEGMENT_VERSIONS: tuple[SegmentVersion, ...] = (
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
)
RERANK_SEGMENT_VERSIONS: tuple[SegmentVersion, ...] = (
    "v7",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
)
V7_CANDIDATE_POOL_VERSIONS: tuple[SegmentVersion, ...] = (
    "v7",
    "v11",
    "v13",
    "v14",
    "v15",
    "v16",
)
RERANK_STRATEGIES: dict[SegmentVersion, RerankStrategy] = {
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

PREDICTED_MASK_DIRS = {
    "v1": SEGMENT_DIR / "predicted_masks_v1",
    "v2": SEGMENT_DIR / "predicted_masks_v2",
    "v3": SEGMENT_DIR / "predicted_masks_v3",
    "v4": SEGMENT_DIR / "predicted_masks_v4",
    "v5": SEGMENT_DIR / "predicted_masks_v5",
    "v6": SEGMENT_DIR / "predicted_masks_v6",
    "v7": SEGMENT_DIR / "predicted_masks_v7",
    "v8": SEGMENT_DIR / "predicted_masks_v8",
    "v9": SEGMENT_DIR / "predicted_masks_v9",
    "v10": SEGMENT_DIR / "predicted_masks_v10",
    "v11": SEGMENT_DIR / "predicted_masks_v11",
    "v12": SEGMENT_DIR / "predicted_masks_v12",
    "v13": SEGMENT_DIR / "predicted_masks_v13",
    "v14": SEGMENT_DIR / "predicted_masks_v14",
    "v15": SEGMENT_DIR / "predicted_masks_v15",
    "v16": SEGMENT_DIR / "predicted_masks_v16",
}

SEGMENT_CROP_MODES: dict[SegmentVersion, CropMode] = {
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

SEGMENT_BBOX_PADDING: dict[SegmentVersion, float] = {
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

SEGMENT_VERSION_DESCRIPTIONS = {
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


def _bbox_area(bbox: list | tuple) -> float:
    xmin, ymin, xmax, ymax = bbox
    return max(0.0, float(xmax - xmin)) * max(0.0, float(ymax - ymin))


def _mask_pixel_count(mask: torch.Tensor | None) -> float:
    if mask is None:
        return 0.0
    if isinstance(mask, torch.Tensor):
        return float(mask.sum().item())
    return 0.0


def _detection_score(item: dict) -> float:
    score = item.get("plastic_score")
    if score is not None:
        return float(score)
    mask_score = _mask_pixel_count(item.get("plastic_mask_original_coords"))
    if mask_score > 0:
        return mask_score
    return _bbox_area(item.get("plastic_bbox_original_coords", [0, 0, 0, 0]))


def _tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    if mask_np.max() <= 1:
        mask_np = mask_np * 255
    return Image.fromarray(mask_np, mode="L")


def select_v1_mask(result: dict) -> torch.Tensor | None:
    in_hand = result.get("plastic_in_hand_region_detections") or []
    if in_hand:
        best = max(in_hand, key=lambda item: float(item.get("iou_plastic_with_hand", 0.0)))
        return best.get("plastic_mask_original_coords")

    all_plastic = result.get("all_detected_plastic_for_analysis") or []
    if all_plastic:
        best = max(
            all_plastic,
            key=lambda item: _bbox_area(item.get("plastic_bbox_original_coords", [0, 0, 0, 0])),
        )
        return best.get("plastic_mask_original_coords")
    return None


def select_v2_mask(result: dict) -> torch.Tensor | None:
    masks = result.get("top_k_plastic_masks") or []
    if not masks:
        return None
    return masks[0]


def select_v3_mask(result: dict) -> torch.Tensor | None:
    all_plastic = result.get("all_detected_plastic_for_analysis") or []
    if not all_plastic:
        return None
    best = max(all_plastic, key=_detection_score)
    return best.get("plastic_mask_original_coords")


def select_v4_mask(result: dict) -> torch.Tensor | None:
    return select_v3_mask(result)


def select_v5_mask(v1_result: dict, v2_result: dict | None = None) -> torch.Tensor | None:
    in_hand = v1_result.get("plastic_in_hand_region_detections") or []
    if in_hand:
        best = max(in_hand, key=_detection_score)
        return best.get("plastic_mask_original_coords")

    if v2_result is not None:
        mask = select_v2_mask(v2_result)
        if mask is not None:
            return mask

    return select_v3_mask(v1_result)


def select_v6_mask(v1_result: dict, v2_result: dict | None = None) -> torch.Tensor | None:
    return select_v5_mask(v1_result, v2_result)


def select_v8_mask(v1_result: dict, v2_result: dict | None = None) -> torch.Tensor | None:
    return select_v5_mask(v1_result, v2_result)


def select_segment_mask(
    result: dict,
    version: SegmentVersion,
    *,
    v2_result: dict | None = None,
) -> torch.Tensor | None:
    if version == "v1":
        return select_v1_mask(result)
    if version == "v2":
        return select_v2_mask(result)
    if version == "v3":
        return select_v3_mask(result)
    if version == "v4":
        return select_v4_mask(result)
    if version == "v5":
        return select_v5_mask(result, v2_result)
    if version == "v6":
        return select_v6_mask(result, v2_result)
    if version == "v8":
        return select_v8_mask(result, v2_result)
    raise ValueError(f"Unsupported segment version: {version}")


def segment_result_path(
    row_index: int,
    version: SegmentVersion,
    *,
    v1_dir: Path | None = None,
) -> Path:
    if version in {
        "v1",
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
    }:
        base = v1_dir or _V1_RESULTS_DIR
        return base / f"v1_result_{row_index:06d}.pkl"
    return SEGMENT_V2_DIR / f"v2_result_{row_index:06d}.pkl"


def segment_v2_result_path(row_index: int) -> Path:
    return SEGMENT_V2_DIR / f"v2_result_{row_index:06d}.pkl"


def export_segment_masks(
    version: SegmentVersion,
    *,
    meta_csv: Path = META2_CSV,
    output_dir: Path | None = None,
    overwrite: bool = False,
    v1_results_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or PREDICTED_MASK_DIRS[version]
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_df = pd.read_csv(meta_csv)

    exported = 0
    missing = 0
    for row_index, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"export {version} masks"):
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue

        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        mask_path = output_dir / f"{frame_id}.png"
        if mask_path.is_file() and not overwrite:
            exported += 1
            continue

        result_path = segment_result_path(int(row_index), version, v1_dir=v1_results_dir)
        if not result_path.is_file():
            missing += 1
            continue

        with result_path.open("rb") as handle:
            result = pickle.load(handle)

        v2_result = None
        if version in {"v5", "v6", "v8"}:
            v2_path = segment_v2_result_path(int(row_index))
            if v2_path.is_file():
                with v2_path.open("rb") as handle:
                    v2_result = pickle.load(handle)

        mask_tensor = select_segment_mask(result, version, v2_result=v2_result)
        if mask_tensor is None:
            if mask_path.is_file():
                mask_path.unlink()
            missing += 1
            continue

        _tensor_mask_to_pil(mask_tensor).save(mask_path)
        exported += 1

    print(f"{version}: exported {exported} masks to {output_dir} (missing/no-detection: {missing})")
    return output_dir


def load_segment_classification_samples(
    version: SegmentVersion,
    *,
    meta_csv: Path = META2_CSV,
    mask_dir: Path | None = None,
    export_if_missing: bool = True,
    crop_mode: CropMode | None = None,
    v1_results_dir: Path | None = None,
) -> list[ClassificationSample]:
    mask_dir = mask_dir or PREDICTED_MASK_DIRS[version]
    resolved_crop_mode = crop_mode or SEGMENT_CROP_MODES[version]
    resolved_padding = SEGMENT_BBOX_PADDING[version]

    if export_if_missing and not mask_dir.is_dir():
        export_segment_masks(
            version,
            meta_csv=meta_csv,
            output_dir=mask_dir,
            v1_results_dir=v1_results_dir,
        )

    meta_df = pd.read_csv(meta_csv)
    samples: list[ClassificationSample] = []

    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue

        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        mask_path = mask_dir / f"{frame_id}.png"
        resolved_mask = mask_path if mask_path.is_file() else None
        clip_id = clip_id_from_path(row.get("timestamp_clip_path"))
        samples.append(
            ClassificationSample(
                frame_path=frame_path,
                mask_path=resolved_mask,
                category_name=str(row["category_name"]),
                gx_id=str(row["gx_id"]),
                clip_id=clip_id,
                crop_mode=resolved_crop_mode,
                bbox_padding_ratio=resolved_padding,
            )
        )

    with_mask = sum(1 for sample in samples if sample.mask_path is not None)
    print(
        f"{version}: {with_mask}/{len(samples)} samples have predicted masks "
        f"(crop_mode={resolved_crop_mode})"
    )
    return samples
