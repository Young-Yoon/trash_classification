"""Top-k candidate mask reranking for segment classification."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import META2_CSV, SEGMENT_DIR, SEGMENT_V1_DIR, SEGMENT_V2_DIR
from dataset_paths import frame_result_id, resolve_meta_asset_path
from image_utils import CropMode, load_classification_image
from paths import clip_id_from_path
from segment_masks import (
    SegmentVersion,
    V7_CANDIDATE_POOL_VERSIONS,
    _detection_score,
    _tensor_mask_to_pil,
    segment_result_path,
    segment_v2_result_path,
    select_v2_mask,
    select_v3_mask,
)
from training_data import ClassificationSample


@dataclass
class RerankClassificationSample(ClassificationSample):
    candidate_mask_paths: list[Path] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)
    candidate_hand_ious: list[float] = field(default_factory=list)


def _mask_signature(mask: torch.Tensor) -> tuple[int, int, int, int]:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask_np = mask.detach().cpu().numpy()
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _dedupe_masks(masks: list[torch.Tensor]) -> list[torch.Tensor]:
    unique: list[torch.Tensor] = []
    seen: set[tuple[int, int, int, int]] = set()
    for mask in masks:
        signature = _mask_signature(mask)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mask)
    return unique


def select_top_k_masks(
    result: dict,
    k: int = 5,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    all_plastic = result.get("all_detected_plastic_for_analysis") or []
    if not all_plastic:
        return [], [], []
    ranked = sorted(all_plastic, key=_detection_score, reverse=True)
    masks: list[torch.Tensor] = []
    scores: list[float] = []
    hand_ious: list[float] = []
    for item in ranked:
        mask = item.get("plastic_mask_original_coords")
        if mask is None:
            continue
        masks.append(mask)
        scores.append(_detection_score(item))
        hand_ious.append(float(item.get("iou_plastic_with_hand", 0.0)))
    deduped_masks = _dedupe_masks(masks)
    if not deduped_masks:
        return [], [], []
    deduped_scores: list[float] = []
    deduped_hand_ious: list[float] = []
    seen: set[tuple[int, int, int, int]] = set()
    for mask, score, hand_iou in zip(masks, scores, hand_ious):
        signature = _mask_signature(mask)
        if signature in seen:
            continue
        seen.add(signature)
        deduped_scores.append(score)
        deduped_hand_ious.append(hand_iou)
    return deduped_masks[:k], deduped_scores[:k], deduped_hand_ious[:k]


def select_in_hand_top_k_masks(
    v1_result: dict,
    v2_result: dict | None,
    *,
    k: int = 5,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    in_hand = v1_result.get("plastic_in_hand_region_detections") or []
    if in_hand:
        ranked = sorted(in_hand, key=_detection_score, reverse=True)
        masks: list[torch.Tensor] = []
        scores: list[float] = []
        hand_ious: list[float] = []
        for item in ranked:
            mask = item.get("plastic_mask_original_coords")
            if mask is None:
                continue
            masks.append(mask)
            scores.append(_detection_score(item))
            hand_ious.append(float(item.get("iou_plastic_with_hand", 0.0)))
        deduped_masks = _dedupe_masks(masks)
        if not deduped_masks:
            return [], [], []
        deduped_scores: list[float] = []
        deduped_hand_ious: list[float] = []
        seen: set[tuple[int, int, int, int]] = set()
        for mask, score, hand_iou in zip(masks, scores, hand_ious):
            signature = _mask_signature(mask)
            if signature in seen:
                continue
            seen.add(signature)
            deduped_scores.append(score)
            deduped_hand_ious.append(hand_iou)
        return deduped_masks[:k], deduped_scores[:k], deduped_hand_ious[:k]

    fallback = select_v3_mask(v1_result)
    if fallback is None and v2_result is not None:
        fallback = select_v2_mask(v2_result)
    if fallback is None:
        return [], [], []
    return [fallback], [0.0], [0.0]


def select_v9_candidate_masks(
    v1_result: dict,
    v2_result: dict | None,
    *,
    k: int = 5,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    scored: list[tuple[float, float, torch.Tensor]] = []

    in_hand = v1_result.get("plastic_in_hand_region_detections") or []
    for item in in_hand:
        mask = item.get("plastic_mask_original_coords")
        if mask is not None:
            scored.append(
                (
                    _detection_score(item),
                    float(item.get("iou_plastic_with_hand", 0.0)),
                    mask,
                )
            )

    if not in_hand and v2_result is not None:
        mask = select_v2_mask(v2_result)
        if mask is not None:
            scored.append((10_000.0, 0.0, mask))

    if not scored:
        fallback = select_v3_mask(v1_result)
        if fallback is not None:
            scored.append((0.0, 0.0, fallback))

    all_plastic = v1_result.get("all_detected_plastic_for_analysis") or []
    for item in all_plastic:
        mask = item.get("plastic_mask_original_coords")
        if mask is not None:
            scored.append(
                (
                    _detection_score(item),
                    float(item.get("iou_plastic_with_hand", 0.0)),
                    mask,
                )
            )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    masks = _dedupe_masks([mask for _, _, mask in scored])
    score_map: list[float] = []
    hand_map: list[float] = []
    seen: set[tuple[int, int, int, int]] = set()
    for score, hand_iou, mask in scored:
        signature = _mask_signature(mask)
        if signature in seen:
            continue
        seen.add(signature)
        score_map.append(score)
        hand_map.append(hand_iou)
    return masks[:k], score_map[:k], hand_map[:k]


def select_rerank_masks(
    version: SegmentVersion,
    v1_result: dict,
    v2_result: dict | None,
    *,
    k: int = 5,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    if version in V7_CANDIDATE_POOL_VERSIONS or version == "v11":
        return select_top_k_masks(v1_result, k=k)
    if version in {"v10", "v12"}:
        return select_in_hand_top_k_masks(v1_result, v2_result, k=k)
    if version == "v9":
        return select_v9_candidate_masks(v1_result, v2_result, k=k)
    raise ValueError(f"Unsupported rerank segment version: {version}")


def load_rerank_classification_samples(
    version: SegmentVersion,
    *,
    meta_csv: Path = META2_CSV,
    candidate_k: int = 5,
    crop_mode: CropMode = "bbox",
    bbox_padding_ratio: float = 0.0,
    cache_dir: Path | None = None,
    v1_results_dir: Path | None = None,
) -> list[RerankClassificationSample]:
    meta_df = pd.read_csv(meta_csv)
    cache_key = "v7" if version in {"v13", "v14", "v15", "v16"} else version
    cache_root = cache_dir or (SEGMENT_DIR / f"rerank_candidates_{cache_key}")
    cache_root.mkdir(parents=True, exist_ok=True)

    samples: list[RerankClassificationSample] = []
    missing = 0

    for row_index, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue

        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        v1_path = segment_result_path(int(row_index), version, v1_dir=v1_results_dir)
        if not v1_path.is_file():
            missing += 1
            continue

        with v1_path.open("rb") as handle:
            v1_result = pickle.load(handle)

        v2_result = None
        if version in {"v9", "v10", "v12"}:
            v2_path = segment_v2_result_path(int(row_index))
            if v2_path.is_file():
                with v2_path.open("rb") as handle:
                    v2_result = pickle.load(handle)

        candidate_masks, candidate_scores, candidate_hand_ious = select_rerank_masks(
            version,
            v1_result,
            v2_result,
            k=candidate_k,
        )
        if not candidate_masks:
            missing += 1
            continue

        candidate_paths: list[Path] = []
        for idx, mask_tensor in enumerate(candidate_masks):
            candidate_path = cache_root / f"{frame_id}_cand{idx}.png"
            if not candidate_path.is_file():
                _tensor_mask_to_pil(mask_tensor).save(candidate_path)
            candidate_paths.append(candidate_path)

        clip_id = clip_id_from_path(row.get("timestamp_clip_path"))
        samples.append(
            RerankClassificationSample(
                frame_path=frame_path,
                mask_path=candidate_paths[0],
                category_name=str(row["category_name"]),
                gx_id=str(row["gx_id"]),
                clip_id=clip_id,
                crop_mode=crop_mode,
                bbox_padding_ratio=bbox_padding_ratio,
                candidate_mask_paths=candidate_paths,
                candidate_scores=candidate_scores[: len(candidate_paths)],
                candidate_hand_ious=candidate_hand_ious[: len(candidate_paths)],
            )
        )

    print(
        f"{version} rerank: {len(samples)}/{len(meta_df)} samples with "
        f"<= {candidate_k} candidates (missing/no-detection: {missing})"
    )
    return samples


def load_candidate_crops(
    sample: RerankClassificationSample,
) -> list[Image.Image]:
    crops: list[Image.Image] = []
    for mask_path in sample.candidate_mask_paths:
        crops.append(
            load_classification_image(
                sample.frame_path,
                mask_path,
                crop_mode=sample.crop_mode,
                bbox_padding_ratio=sample.bbox_padding_ratio,
            )
        )
    return crops
