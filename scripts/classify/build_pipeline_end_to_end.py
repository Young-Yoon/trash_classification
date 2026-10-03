#!/usr/bin/env python3
"""Build per-frame end-to-end pipeline metrics (mask/bbox IoU + classification)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
SEGMENT_DIR_PKG = PACKAGE_DIR.parent / "segment"
for path in (SHARED_DIR, PACKAGE_DIR, SEGMENT_DIR_PKG):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from cnn_model import efficientnet_transform, load_cnn_checkpoint
from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, RESULTS_DIR, SEGMENT_DIR, SEGMENT_V1_DIR, SEGMENT_V2_DIR
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import predicted_label_name, result_json_path
from fusion_image import predict_crop_probabilities
from fusion_model import build_fusion_features, load_fusion_checkpoint
from geometry import compute_bbox_iou, compute_mask_iou
from infer_trained import (
    _encode_crops,
    _normalize_scores,
    pick_best_candidate_index,
    resolve_checkpoint,
)
from labels import load_label_sets
from presets import PRESETS, TRAINED_PRESETS
from segment_masks import (
    ALL_SEGMENT_VERSIONS,
    RERANK_SEGMENT_VERSIONS,
    RERANK_STRATEGIES,
    SEGMENT_BBOX_PADDING,
    SEGMENT_CROP_MODES,
    SEGMENT_VERSION_DESCRIPTIONS,
    SegmentVersion,
    segment_result_path,
    segment_v2_result_path,
    select_segment_mask,
)
from segment_rerank import RerankClassificationSample, load_candidate_crops, load_rerank_classification_samples
from training_data import load_text_category_map
from vision_model import load_vision_checkpoint

IOU_ZERO_EPS = 1e-6

VERSION_PHASE: dict[str, tuple[str, int]] = {
    "v1": ("Phase1_Backbone", 1),
    "v2": ("Phase1_Backbone", 1),
    "v3": ("Phase2_MaskSelect", 2),
    "v4": ("Phase2_MaskSelect", 2),
    "v5": ("Phase2_MaskSelect", 2),
    "v6": ("Phase2_MaskSelect", 2),
    "v8": ("Phase2_MaskSelect", 2),
    "v7": ("Phase3_RerankPool", 3),
    "v9": ("Phase3_RerankPool", 3),
    "v10": ("Phase3_RerankPool", 3),
    "v11": ("Phase4_RerankStrategy", 4),
    "v12": ("Phase4_RerankStrategy", 4),
    "v13": ("Phase4_RerankStrategy", 4),
    "v14": ("Phase4_RerankStrategy", 4),
    "v15": ("Phase4_RerankStrategy", 4),
    "v16": ("Phase4_RerankStrategy", 4),
}

NON_RERANK_VERSIONS: tuple[SegmentVersion, ...] = (
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v8",
)

METHOD_LABELS = {
    "clip_full_labels_top5": "CLIP zero-shot",
    "clip_linear_probe_crop": "CLIP linear probe",
    "clip_lora_crop": "CLIP LoRA",
    "siglip_linear_probe_crop": "SigLIP linear probe",
    "siglip_lora_crop": "SigLIP LoRA",
    "efficientnet_b0_crop": "EfficientNet-B0",
    "late_fusion_clip_whisper_crop": "Late fusion (CLIP)",
    "late_fusion_efficientnet_whisper_crop": "Late fusion (EfficientNet)",
}

ZERO_SHOT_E2E_PRESETS = {
    "clip_full_labels_top5": PRESETS["clip_full_labels_top5"],
}
E2E_PRESET_KEYS = list(TRAINED_PRESETS.keys()) + list(ZERO_SHOT_E2E_PRESETS.keys())

_ZEROSHOT_MODELS: dict | None = None
_ZEROSHOT_LABEL_SETS: tuple[list[str], list[str]] | None = None


def resolve_e2e_preset(base_preset: str) -> dict:
    if base_preset in TRAINED_PRESETS:
        return TRAINED_PRESETS[base_preset]
    if base_preset in ZERO_SHOT_E2E_PRESETS:
        return ZERO_SHOT_E2E_PRESETS[base_preset]
    raise KeyError(f"Unknown e2e preset: {base_preset}")


def segment_output_dir(base_output_dir: str, version: SegmentVersion) -> str:
    return f"{base_output_dir}_seg_{version}"


def e2e_segment_output_dir(base_preset: str, version: SegmentVersion) -> str:
    return segment_output_dir(resolve_e2e_preset(base_preset)["output_dir"], version)


def get_zeroshot_resources() -> tuple[dict, list[str], list[str]]:
    global _ZEROSHOT_MODELS, _ZEROSHOT_LABEL_SETS
    if _ZEROSHOT_MODELS is None:
        from transformers import pipeline

        full_labels, concise_labels = load_label_sets()
        _ZEROSHOT_LABEL_SETS = (full_labels, concise_labels)
        _ZEROSHOT_MODELS = {
            "clip": pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32"),
            "siglip": pipeline(
                "zero-shot-image-classification",
                model="google/siglip-base-patch16-224",
            ),
        }
    assert _ZEROSHOT_LABEL_SETS is not None
    return _ZEROSHOT_MODELS, _ZEROSHOT_LABEL_SETS[0], _ZEROSHOT_LABEL_SETS[1]


def load_gt_mask(row: pd.Series) -> torch.Tensor | None:
    frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
    mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
    if frame_path is None or mask_path is None or not frame_path.is_file() or not mask_path.is_file():
        return None
    frame = Image.open(frame_path)
    gt = Image.open(mask_path).convert("L")
    if gt.size != frame.size:
        gt = gt.resize(frame.size, Image.NEAREST)
    return torch.from_numpy(np.array(gt) > 0).bool()


def resize_mask_to_frame(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    if mask.shape == (height, width):
        return mask.bool()
    return (
        torch.nn.functional.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .bool()
    )


def bbox_from_mask(mask: torch.Tensor) -> list[int] | None:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    ys, xs = torch.where(mask.bool())
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou_vs_gt(pred_mask: torch.Tensor | None, gt_mask: torch.Tensor | None) -> float:
    if gt_mask is None or pred_mask is None:
        return 0.0
    pred_mask = resize_mask_to_frame(
        pred_mask.cpu() if isinstance(pred_mask, torch.Tensor) else pred_mask,
        gt_mask.shape[0],
        gt_mask.shape[1],
    )
    pred_bbox = bbox_from_mask(pred_mask)
    gt_bbox = bbox_from_mask(gt_mask)
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    return float(compute_bbox_iou(pred_bbox, gt_bbox))


def mask_iou_vs_gt(selected: torch.Tensor | None, gt_mask: torch.Tensor | None) -> float:
    if gt_mask is None or selected is None:
        return 0.0
    selected = resize_mask_to_frame(
        selected.cpu() if isinstance(selected, torch.Tensor) else selected,
        gt_mask.shape[0],
        gt_mask.shape[1],
    )
    return float(compute_mask_iou(selected, gt_mask))


def load_mask_from_png(path: Path, gt_mask: torch.Tensor) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask_tensor = torch.from_numpy(np.array(mask) > 0).bool()
    return resize_mask_to_frame(mask_tensor, gt_mask.shape[0], gt_mask.shape[1])


def bbox_iou_from_pred_path(pred_mask_path: str, gt_mask: torch.Tensor | None) -> float:
    if not pred_mask_path or gt_mask is None:
        return 0.0
    path = Path(pred_mask_path)
    if not path.is_file():
        return 0.0
    pred_mask = load_mask_from_png(path, gt_mask)
    return bbox_iou_vs_gt(pred_mask, gt_mask)


def compute_non_rerank_ious(version: SegmentVersion, meta_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    ious: dict[str, float] = {}
    mask_paths: dict[str, str] = {}
    mask_dir = SEGMENT_DIR / f"predicted_masks_{version}"
    for row_index, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"{version} mask IoU"):
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        gt_mask = load_gt_mask(row)

        if version == "v2":
            result_path = SEGMENT_V2_DIR / f"v2_result_{int(row_index):06d}.pkl"
            if not result_path.is_file():
                ious[frame_id] = 0.0
                continue
            with result_path.open("rb") as handle:
                result = pickle.load(handle)
            selected = select_segment_mask(result, "v2")
        else:
            result_path = segment_result_path(int(row_index), version)
            if not result_path.is_file():
                ious[frame_id] = 0.0
                continue
            with result_path.open("rb") as handle:
                v1_result = pickle.load(handle)
            v2_result = None
            if version in {"v5", "v6", "v8"}:
                v2_path = segment_v2_result_path(int(row_index))
                if v2_path.is_file():
                    with v2_path.open("rb") as handle:
                        v2_result = pickle.load(handle)
            selected = select_segment_mask(v1_result, version, v2_result=v2_result)

        ious[frame_id] = mask_iou_vs_gt(selected, gt_mask)
        pred_path = mask_dir / f"{frame_id}.png"
        mask_paths[frame_id] = str(pred_path) if pred_path.is_file() else ""
    return ious, mask_paths


@torch.no_grad()
def compute_vision_rerank_indices(
    checkpoint_dir: Path,
    samples: list[RerankClassificationSample],
    *,
    device: torch.device,
    rerank_strategy: str,
) -> dict[str, int]:
    model, processor, label2id, _config = load_vision_checkpoint(checkpoint_dir, device)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    indices: dict[str, int] = {}
    for sample in tqdm(samples, desc=f"{checkpoint_dir.name} vision rerank"):
        crops = load_candidate_crops(sample)
        if not crops:
            continue
        pixel_values = _encode_crops(crops, processor=processor, transform=None).to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        text_category = clip_to_category.get(sample.clip_id or "", "")
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        indices[frame_id] = pick_best_candidate_index(
            probs,
            strategy=rerank_strategy,
            candidate_scores=sample.candidate_scores,
            candidate_hand_ious=sample.candidate_hand_ious,
            id2label=id2label,
            text_category=text_category,
        )
    return indices


@torch.no_grad()
def compute_cnn_rerank_indices(
    checkpoint_dir: Path,
    samples: list[RerankClassificationSample],
    *,
    device: torch.device,
    rerank_strategy: str,
) -> dict[str, int]:
    model, label2id, _config = load_cnn_checkpoint(checkpoint_dir, device)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    transform = efficientnet_transform()
    indices: dict[str, int] = {}
    for sample in tqdm(samples, desc=f"{checkpoint_dir.name} cnn rerank"):
        crops = load_candidate_crops(sample)
        if not crops:
            continue
        pixel_values = _encode_crops(crops, processor=None, transform=transform).to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        text_category = clip_to_category.get(sample.clip_id or "", "")
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        indices[frame_id] = pick_best_candidate_index(
            probs,
            strategy=rerank_strategy,
            candidate_scores=sample.candidate_scores,
            candidate_hand_ious=sample.candidate_hand_ious,
            id2label=id2label,
            text_category=text_category,
        )
    return indices


@torch.no_grad()
def compute_fusion_rerank_indices(
    checkpoint_dir: Path,
    samples: list[RerankClassificationSample],
    *,
    device: torch.device,
    rerank_strategy: str,
) -> dict[str, int]:
    fusion_model, label2id, config = load_fusion_checkpoint(checkpoint_dir)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    image_checkpoint = Path(config["image_checkpoint"])

    indices: dict[str, int] = {}
    for sample in tqdm(samples, desc=f"{checkpoint_dir.name} fusion rerank"):
        crops = load_candidate_crops(sample)
        if not crops:
            continue
        image_probs_batch, image_id2label = predict_crop_probabilities(
            image_checkpoint,
            crops,
            device=device,
        )
        text_category = clip_to_category.get(sample.clip_id or "")
        sam = _normalize_scores(sample.candidate_scores, len(image_probs_batch))
        hand = _normalize_scores(sample.candidate_hand_ious, len(image_probs_batch))
        best_idx = 0
        best_score = -1.0
        for cand_idx, candidate_probs in enumerate(image_probs_batch):
            image_probs = np.zeros(len(label2id), dtype=np.float32)
            for class_idx, score in enumerate(candidate_probs):
                label_name = image_id2label.get(class_idx)
                if label_name and label_name in label2id:
                    image_probs[label2id[label_name]] = float(score)
            features = build_fusion_features(image_probs, text_category, label2id).reshape(1, -1)
            fusion_probs = fusion_model.predict_proba(features)[0]
            sorted_probs = np.sort(fusion_probs)
            margin = (
                float(sorted_probs[-1] - sorted_probs[-2])
                if sorted_probs.size >= 2
                else float(fusion_probs.max())
            )
            fusion_top_label = id2label[int(np.argmax(fusion_probs))]
            if rerank_strategy == "margin":
                score = margin
            elif rerank_strategy == "composite":
                score = 0.7 * float(fusion_probs.max()) + 0.3 * float(sam[cand_idx])
            elif rerank_strategy == "composite_margin":
                score = 0.7 * margin + 0.3 * float(sam[cand_idx])
            elif rerank_strategy == "text_aware":
                score = margin + (0.2 if text_category and fusion_top_label == text_category else 0.0)
            elif rerank_strategy == "geo_composite":
                score = 0.5 * margin + 0.3 * float(sam[cand_idx]) + 0.2 * float(hand[cand_idx])
            else:
                score = float(fusion_probs.max())
            if score > best_score:
                best_score = score
                best_idx = cand_idx
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        indices[frame_id] = best_idx
    return indices


def compute_zeroshot_rerank_indices(
    preset: dict,
    samples: list[RerankClassificationSample],
    *,
    rerank_strategy: str,
    top_k: int = 5,
) -> dict[str, int]:
    from run_segment_zeroshot_eval import classify_sample_candidates

    models, full_labels, concise_labels = get_zeroshot_resources()
    id2label = {idx: label for idx, label in enumerate(concise_labels)}
    clip_to_category = load_text_category_map()
    indices: dict[str, int] = {}
    model_name = preset["model"]
    for sample in tqdm(samples, desc=f"zeroshot {model_name} rerank"):
        if not sample.candidate_mask_paths:
            continue
        probs = classify_sample_candidates(
            sample,
            preset=preset,
            models=models,
            full_labels=full_labels,
            concise_labels=concise_labels,
            top_k=top_k,
        )
        text_category = clip_to_category.get(sample.clip_id or "", "")
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        indices[frame_id] = pick_best_candidate_index(
            probs,
            strategy=rerank_strategy,
            candidate_scores=sample.candidate_scores,
            candidate_hand_ious=sample.candidate_hand_ious,
            id2label=id2label,
            text_category=text_category,
        )
    return indices


def compute_rerank_ious(
    version: SegmentVersion,
    base_preset: str,
    preset: dict,
    meta_df: pd.DataFrame,
    *,
    samples: list[RerankClassificationSample],
    device: torch.device,
    cache_path: Path | None = None,
    from_cache_only: bool = False,
) -> tuple[dict[str, float], dict[str, str]]:
    if cache_path and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        ious = dict(zip(cached["frame_id"], cached["selected_iou"].astype(float)))
        paths = dict(zip(cached["frame_id"], cached.get("pred_mask_path", pd.Series(dtype=str)).fillna("")))
        return ious, paths

    if from_cache_only:
        raise SystemExit(f"Missing rerank cache: {cache_path}")

    rerank_strategy = RERANK_STRATEGIES[version]
    sample_by_frame = {
        frame_result_id(sample.gx_id, sample.frame_path): sample for sample in samples
    }

    preset = resolve_e2e_preset(base_preset)
    model_type = preset.get("model_type")

    if base_preset in ZERO_SHOT_E2E_PRESETS:
        zeroshot_preset = dict(preset)
        zeroshot_preset["crop_mode"] = SEGMENT_CROP_MODES[version]
        selected_indices = compute_zeroshot_rerank_indices(
            zeroshot_preset,
            samples,
            rerank_strategy=rerank_strategy,
        )
    elif model_type == "fusion":
        checkpoint_dir = resolve_checkpoint(preset)
        selected_indices = compute_fusion_rerank_indices(
            checkpoint_dir, samples, device=device, rerank_strategy=rerank_strategy
        )
    elif model_type == "efficientnet":
        checkpoint_dir = resolve_checkpoint(preset)
        selected_indices = compute_cnn_rerank_indices(
            checkpoint_dir, samples, device=device, rerank_strategy=rerank_strategy
        )
    else:
        checkpoint_dir = resolve_checkpoint(preset)
        selected_indices = compute_vision_rerank_indices(
            checkpoint_dir, samples, device=device, rerank_strategy=rerank_strategy
        )

    ious: dict[str, float] = {}
    mask_paths: dict[str, str] = {}
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"{version}/{base_preset} IoU join"):
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        sample = sample_by_frame.get(frame_id)
        if sample is None:
            ious[frame_id] = 0.0
            continue
        gt_mask = load_gt_mask(row)
        if gt_mask is None:
            ious[frame_id] = 0.0
            continue
        cand_idx = selected_indices.get(frame_id, 0)
        cand_idx = min(cand_idx, len(sample.candidate_mask_paths) - 1)
        mask_path = sample.candidate_mask_paths[cand_idx]
        pred_mask = load_mask_from_png(mask_path, gt_mask)
        ious[frame_id] = float(compute_mask_iou(pred_mask, gt_mask))
        mask_paths[frame_id] = str(mask_path)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"frame_id": fid, "selected_iou": iou, "pred_mask_path": mask_paths.get(fid, "")}
                for fid, iou in ious.items()
            ]
        ).to_csv(cache_path, index=False)
    return ious, mask_paths


def load_class_top1_map(
    base_preset: str,
    version: SegmentVersion,
    meta_df: pd.DataFrame,
    *,
    results_dir: Path,
) -> dict[str, bool]:
    output_dir = results_dir / e2e_segment_output_dir(base_preset, version)
    top1_map: dict[str, bool] = {}
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        json_path = result_json_path(row, output_dir)
        if not json_path.is_file():
            top1_map[frame_id] = False
            continue
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            top1_map[frame_id] = False
            continue

        if isinstance(payload, list) and payload:
            pred = predicted_label_name(payload[0]["label"])
        elif isinstance(payload, dict) and "label" in payload:
            pred = predicted_label_name(payload["label"])
        else:
            top1_map[frame_id] = False
            continue
        top1_map[frame_id] = pred == str(row["category_name"])
    return top1_map


def build_per_frame_records(
    meta_df: pd.DataFrame,
    *,
    results_dir: Path,
    device: torch.device,
    cache_dir: Path | None,
    versions: list[SegmentVersion] | None = None,
    presets: list[str] | None = None,
    from_cache_only: bool = False,
) -> pd.DataFrame:
    selected_versions = list(versions or ALL_SEGMENT_VERSIONS)
    selected_presets = list(presets or E2E_PRESET_KEYS)

    non_rerank_ious: dict[str, dict[str, float]] = {}
    non_rerank_mask_paths: dict[str, dict[str, str]] = {}
    for version in selected_versions:
        if version in NON_RERANK_VERSIONS:
            nr_cache = cache_dir / f"mask_iou_{version}.csv" if cache_dir else None
            if nr_cache and nr_cache.is_file():
                cached = pd.read_csv(nr_cache)
                non_rerank_ious[version] = dict(
                    zip(cached["frame_id"], cached["selected_iou"].astype(float))
                )
                non_rerank_mask_paths[version] = dict(
                    zip(cached["frame_id"], cached.get("pred_mask_path", pd.Series(dtype=str)).fillna(""))
                )
                continue
            if from_cache_only:
                raise SystemExit(f"Missing cache for {version}: {nr_cache}")
            ious, paths = compute_non_rerank_ious(version, meta_df)
            non_rerank_ious[version] = ious
            non_rerank_mask_paths[version] = paths
            if nr_cache:
                nr_cache.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [
                        {"frame_id": fid, "selected_iou": iou, "pred_mask_path": paths.get(fid, "")}
                        for fid, iou in ious.items()
                    ]
                ).to_csv(nr_cache, index=False)

    rerank_ious: dict[tuple[str, str], dict[str, float]] = {}
    rerank_mask_paths: dict[tuple[str, str], dict[str, str]] = {}
    rerank_samples_cache: dict[str, list[RerankClassificationSample]] = {}
    for version in selected_versions:
        if version not in RERANK_SEGMENT_VERSIONS:
            continue
        if version not in rerank_samples_cache and not from_cache_only:
            rerank_samples_cache[version] = load_rerank_classification_samples(
                version,
                crop_mode=SEGMENT_CROP_MODES[version],
                bbox_padding_ratio=SEGMENT_BBOX_PADDING[version],
            )
        samples = rerank_samples_cache.get(version, [])
        for base_preset in selected_presets:
            preset = resolve_e2e_preset(base_preset)
            cache_path = None
            if cache_dir:
                cache_path = cache_dir / f"rerank_iou_{version}_{base_preset}.csv"
            if from_cache_only and (not cache_path or not cache_path.is_file()):
                raise SystemExit(f"Missing rerank cache: {cache_path}")
            ious, paths = compute_rerank_ious(
                version,
                base_preset,
                preset,
                meta_df,
                samples=samples,
                device=device,
                cache_path=cache_path,
                from_cache_only=from_cache_only,
            )
            rerank_ious[(version, base_preset)] = ious
            rerank_mask_paths[(version, base_preset)] = paths

    class_maps: dict[tuple[str, str], dict[str, bool]] = {}
    for version in selected_versions:
        for base_preset in selected_presets:
            class_maps[(version, base_preset)] = load_class_top1_map(
                base_preset,
                version,
                meta_df,
                results_dir=results_dir,
            )

    iou_lookup: dict[tuple[str, str | None], dict[str, tuple[float, float, str]]] = {}
    for version in selected_versions:
        preset_keys = selected_presets if version in RERANK_SEGMENT_VERSIONS else [None]
        for base_preset in preset_keys:
            if version in NON_RERANK_VERSIONS:
                path_map = non_rerank_mask_paths[version]
            else:
                path_map = rerank_mask_paths[(version, str(base_preset))]
            frame_metrics: dict[str, tuple[float, float, str]] = {}
            for _, meta_row in meta_df.iterrows():
                frame_path = resolve_meta_asset_path(meta_row.get("copied_frame_path"))
                if frame_path is None or not frame_path.is_file():
                    continue
                frame_id = frame_result_id(str(meta_row["gx_id"]), frame_path)
                gt_mask = load_gt_mask(meta_row)
                pred_mask_path = path_map.get(frame_id, "")
                if gt_mask is None or not pred_mask_path:
                    frame_metrics[frame_id] = (0.0, 0.0, pred_mask_path)
                    continue
                pred_mask = load_mask_from_png(Path(pred_mask_path), gt_mask)
                frame_metrics[frame_id] = (
                    float(compute_mask_iou(pred_mask, gt_mask)),
                    bbox_iou_vs_gt(pred_mask, gt_mask),
                    pred_mask_path,
                )
            lookup_key = (version, base_preset)
            iou_lookup[lookup_key] = frame_metrics

    rows: list[dict] = []
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        gt_label = str(row["category_name"])

        for version in selected_versions:
            phase, phase_order = VERSION_PHASE[version]
            for base_preset in selected_presets:
                lookup_key = (version, base_preset if version in RERANK_SEGMENT_VERSIONS else None)
                metrics = iou_lookup[lookup_key].get(frame_id, (0.0, 0.0, ""))
                selected_mask_iou, selected_bbox_iou, pred_mask_path = metrics
                class_top1_ok = class_maps[(version, base_preset)].get(frame_id, False)
                mask_ok = selected_bbox_iou > IOU_ZERO_EPS
                bbox_iou_zero = selected_bbox_iou <= IOU_ZERO_EPS
                iou_zero_class_ok = bbox_iou_zero and class_top1_ok

                rows.append(
                    {
                        "frame_id": frame_id,
                        "gx_id": str(row["gx_id"]),
                        "category_name": gt_label,
                        "segment_version": version,
                        "phase": phase,
                        "phase_order": phase_order,
                        "base_preset": base_preset,
                        "method": METHOD_LABELS.get(base_preset, base_preset),
                        "selected_mask_iou": round(selected_mask_iou, 6),
                        "selected_bbox_iou": round(selected_bbox_iou, 6),
                        "pred_mask_path": pred_mask_path,
                        "mask_ok": mask_ok,
                        "class_top1_ok": class_top1_ok,
                        "iou_zero_class_ok": iou_zero_class_ok,
                        "description": SEGMENT_VERSION_DESCRIPTIONS.get(version, ""),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_summary(per_frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = per_frame_df.groupby(["segment_version", "base_preset", "phase", "phase_order", "method", "description"])
    for (version, base_preset, phase, phase_order, method, description), group in grouped:
        n = len(group)
        mask_rate = group["mask_ok"].mean() * 100
        class_rate = group["class_top1_ok"].mean() * 100
        iou_zero_class_ok_rate = group["iou_zero_class_ok"].mean() * 100
        pipeline_rate = (group["mask_ok"] & group["class_top1_ok"]).mean() * 100
        masked = group[group["mask_ok"]]
        class_given_mask_rate = float(masked["class_top1_ok"].mean() * 100) if len(masked) else 0.0
        final_prob = mask_rate * class_given_mask_rate / 100

        rows.append(
            {
                "segment_version": version,
                "phase": phase,
                "phase_order": phase_order,
                "base_preset": base_preset,
                "method": method,
                "description": description,
                "n_frames": n,
                "mask_success_rate": round(mask_rate, 2),
                "class_top1_rate": round(class_rate, 2),
                "class_given_mask_rate": round(class_given_mask_rate, 2),
                "iou_zero_class_ok_rate": round(iou_zero_class_ok_rate, 2),
                "pipeline_success_rate": round(pipeline_rate, 2),
                "final_probability": round(final_prob, 2),
                "adjusted_final_probability": round(pipeline_rate, 2),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["phase_order", "segment_version", "base_preset"]).reset_index(drop=True)
    return summary


def aggregate_by_phase(summary_df: pd.DataFrame) -> pd.DataFrame:
    phase_rows = []
    for (phase, phase_order), group in summary_df.groupby(["phase", "phase_order"]):
        phase_rows.append(
            {
                "phase": phase,
                "phase_order": phase_order,
                "n_version_preset_pairs": len(group),
                "mean_mask_success_rate": round(group["mask_success_rate"].mean(), 2),
                "mean_class_top1_rate": round(group["class_top1_rate"].mean(), 2),
                "mean_final_probability": round(group["final_probability"].mean(), 2),
                "mean_adjusted_final_probability": round(group["adjusted_final_probability"].mean(), 2),
                "mean_iou_zero_class_ok_rate": round(group["iou_zero_class_ok_rate"].mean(), 2),
            }
        )
    return pd.DataFrame(phase_rows).sort_values("phase_order").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build end-to-end pipeline metrics.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument(
        "--output-per-frame-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_per_frame.csv",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_summary.csv",
    )
    parser.add_argument(
        "--output-by-phase-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_by_phase.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=RESULTS_DIR / "pipeline_e2e_cache",
        help="Cache rerank IoU CSVs to speed up reruns.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Rebuild metrics from cached mask paths only (skip rerank inference).",
    )
    parser.add_argument(
        "--version",
        action="append",
        choices=list(ALL_SEGMENT_VERSIONS),
        help="Limit to specific segment version(s). Default: all.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=E2E_PRESET_KEYS,
        help="Limit to specific base preset(s). Default: all.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    meta_df = pd.read_csv(args.meta_csv)
    cache_dir = None if args.no_cache else args.cache_dir

    per_frame_df = build_per_frame_records(
        meta_df,
        results_dir=args.results_dir,
        device=device,
        cache_dir=cache_dir,
        versions=args.version,
        presets=args.preset,
        from_cache_only=args.from_cache_only,
    )
    if args.preset and args.output_per_frame_csv.is_file():
        existing = pd.read_csv(args.output_per_frame_csv)
        selected_presets = set(per_frame_df["base_preset"].unique())
        existing = existing[~existing["base_preset"].isin(selected_presets)]
        per_frame_df = pd.concat([existing, per_frame_df], ignore_index=True)

    summary_df = aggregate_summary(per_frame_df)
    phase_df = aggregate_by_phase(summary_df)

    args.output_per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    per_frame_df.to_csv(args.output_per_frame_csv, index=False)
    summary_df.to_csv(args.output_summary_csv, index=False)
    phase_df.to_csv(args.output_by_phase_csv, index=False)

    print(f"Saved per-frame metrics: {args.output_per_frame_csv} ({len(per_frame_df)} rows)")
    print(f"Saved summary: {args.output_summary_csv} ({len(summary_df)} rows)")
    print(f"Saved by-phase summary: {args.output_by_phase_csv}")

    best = summary_df.sort_values(
        ["adjusted_final_probability", "base_preset"],
        ascending=[False, False],
    ).iloc[0]
    print(
        f"\nBest adjusted final: {best['segment_version']} + {best['method']} "
        f"= {best['adjusted_final_probability']:.2f}%"
    )


if __name__ == "__main__":
    main()
