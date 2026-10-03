#!/usr/bin/env python3
"""Run inference with trained classifiers and write top-k JSON outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from cnn_model import efficientnet_transform, load_cnn_checkpoint
from config import CLASSIFICATION_RESULTS_DIR, CLASSIFY_CHECKPOINT_DIR, META2_CSV
from dataset_paths import frame_result_id, resolve_meta_asset_path
from fusion_image import predict_crop_probabilities
from fusion_model import build_fusion_features, load_fusion_checkpoint
from presets import TRAINED_PRESETS
from training_data import (
    ClassificationSample,
    PlasticImageDataset,
    load_classification_samples,
    load_label_maps,
    load_text_category_map,
)
from train import collate_batch
from vision_model import load_vision_checkpoint

try:
    from segment_rerank import RerankClassificationSample, load_candidate_crops
except ImportError:
    RerankClassificationSample = None  # type: ignore[misc, assignment]
    load_candidate_crops = None  # type: ignore[assignment]


def probs_to_topk(probs: np.ndarray, id2label: dict[int, str], top_k: int = 5) -> list[dict]:
    indices = np.argsort(probs)[::-1][:top_k]
    return [{"label": id2label[int(idx)], "score": float(probs[idx])} for idx in indices]


@torch.no_grad()
def infer_vision_checkpoint(
    checkpoint_dir: Path,
    samples: list[ClassificationSample],
    *,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    model, processor, label2id, config = load_vision_checkpoint(checkpoint_dir, device)
    id2label = {idx: name for name, idx in label2id.items()}
    dataset = PlasticImageDataset(
        samples,
        label2id,
        use_mask=config["use_mask"],
        processor=processor,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    outputs: dict[str, list[dict]] = {}
    offset = 0
    for batch in tqdm(loader, desc=checkpoint_dir.name):
        pixel_values = batch["pixel_values"].to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        for idx in range(len(batch["gx_id"])):
            sample = samples[offset + idx]
            frame_id = frame_result_id(sample.gx_id, sample.frame_path)
            outputs[frame_id] = probs_to_topk(probs[idx], id2label, top_k=top_k)
        offset += len(batch["gx_id"])
    return outputs


@torch.no_grad()
def infer_cnn_checkpoint(
    checkpoint_dir: Path,
    samples: list[ClassificationSample],
    *,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    model, label2id, config = load_cnn_checkpoint(checkpoint_dir, device)
    id2label = {idx: name for name, idx in label2id.items()}
    transform = efficientnet_transform()
    dataset = PlasticImageDataset(
        samples,
        label2id,
        use_mask=config["use_mask"],
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    outputs: dict[str, list[dict]] = {}
    offset = 0
    for batch in tqdm(loader, desc=checkpoint_dir.name):
        pixel_values = batch["pixel_values"].to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        for idx in range(len(batch["gx_id"])):
            sample = samples[offset + idx]
            frame_id = frame_result_id(sample.gx_id, sample.frame_path)
            outputs[frame_id] = probs_to_topk(probs[idx], id2label, top_k=top_k)
        offset += len(batch["gx_id"])
    return outputs


def _encode_crops(
    crops: list,
    *,
    processor,
    transform,
) -> torch.Tensor:
    if processor is not None:
        pixel_values = [processor(images=crop, return_tensors="pt")["pixel_values"].squeeze(0) for crop in crops]
        return torch.stack(pixel_values, dim=0)
    if transform is not None:
        return torch.stack([transform(crop) for crop in crops], dim=0)
    raise ValueError("Either processor or transform must be provided for rerank inference.")


def _normalize_scores(values: list[float] | None, length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float32)
    if not values:
        return arr
    for idx in range(min(length, len(values))):
        arr[idx] = float(values[idx])
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def _candidate_margins(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    if sorted_probs.shape[1] < 2:
        return probs.max(axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def pick_best_candidate_index(
    probs: np.ndarray,
    *,
    strategy: str = "confidence",
    candidate_scores: list[float] | None = None,
    candidate_hand_ious: list[float] | None = None,
    id2label: dict[int, str] | None = None,
    text_category: str = "",
) -> int:
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0

    conf = probs.max(axis=1)
    margins = _candidate_margins(probs)
    sam = _normalize_scores(candidate_scores, probs.shape[0])
    hand = _normalize_scores(candidate_hand_ious, probs.shape[0])

    if strategy == "margin":
        return int(np.argmax(margins))
    if strategy == "composite":
        combined = 0.7 * conf + 0.3 * sam
        return int(np.argmax(combined))
    if strategy == "composite_margin":
        combined = 0.7 * margins + 0.3 * sam
        return int(np.argmax(combined))
    if strategy == "text_aware":
        combined = margins.copy()
        if id2label and text_category:
            for cand_idx, row in enumerate(probs):
                pred_label = id2label[int(np.argmax(row))]
                if pred_label == text_category:
                    combined[cand_idx] += 0.2
        return int(np.argmax(combined))
    if strategy == "geo_composite":
        combined = 0.5 * margins + 0.3 * sam + 0.2 * hand
        return int(np.argmax(combined))
    return int(np.argmax(conf))


def _pick_best_candidate_probs(
    probs: np.ndarray,
    *,
    strategy: str = "confidence",
    candidate_scores: list[float] | None = None,
    candidate_hand_ious: list[float] | None = None,
    id2label: dict[int, str] | None = None,
    text_category: str = "",
) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[0] == 0:
        return probs.reshape(-1)
    idx = pick_best_candidate_index(
        probs,
        strategy=strategy,
        candidate_scores=candidate_scores,
        candidate_hand_ious=candidate_hand_ious,
        id2label=id2label,
        text_category=text_category,
    )
    return probs[idx]


@torch.no_grad()
def pick_fusion_rerank_candidate_index(
    checkpoint_dir: Path,
    sample: RerankClassificationSample,
    *,
    device: torch.device,
    rerank_strategy: str = "confidence",
) -> int:
    if load_candidate_crops is None:
        raise RuntimeError("segment_rerank module is unavailable")

    fusion_model, label2id, config = load_fusion_checkpoint(checkpoint_dir)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    image_checkpoint = Path(config["image_checkpoint"])

    crops = load_candidate_crops(sample)
    if not crops:
        return 0

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
    return best_idx


@torch.no_grad()
def infer_vision_rerank(
    checkpoint_dir: Path,
    samples: list[RerankClassificationSample],
    *,
    top_k: int,
    device: torch.device,
    rerank_strategy: str = "confidence",
) -> dict[str, list[dict]]:
    if load_candidate_crops is None:
        raise RuntimeError("segment_rerank module is unavailable")

    model, processor, label2id, _config = load_vision_checkpoint(checkpoint_dir, device)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    outputs: dict[str, list[dict]] = {}

    for sample in tqdm(samples, desc=f"{checkpoint_dir.name} rerank"):
        crops = load_candidate_crops(sample)
        if not crops:
            continue
        pixel_values = _encode_crops(crops, processor=processor, transform=None).to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        text_category = clip_to_category.get(sample.clip_id or "", "")
        best_probs = _pick_best_candidate_probs(
            probs,
            strategy=rerank_strategy,
            candidate_scores=sample.candidate_scores,
            candidate_hand_ious=sample.candidate_hand_ious,
            id2label=id2label,
            text_category=text_category,
        )
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        outputs[frame_id] = probs_to_topk(best_probs, id2label, top_k=top_k)
    return outputs


@torch.no_grad()
def infer_cnn_rerank(
    checkpoint_dir: Path,
    samples: list[RerankClassificationSample],
    *,
    top_k: int,
    device: torch.device,
    rerank_strategy: str = "confidence",
) -> dict[str, list[dict]]:
    if load_candidate_crops is None:
        raise RuntimeError("segment_rerank module is unavailable")

    model, label2id, _config = load_cnn_checkpoint(checkpoint_dir, device)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    transform = efficientnet_transform()
    outputs: dict[str, list[dict]] = {}

    for sample in tqdm(samples, desc=f"{checkpoint_dir.name} rerank"):
        crops = load_candidate_crops(sample)
        if not crops:
            continue
        pixel_values = _encode_crops(crops, processor=None, transform=transform).to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        text_category = clip_to_category.get(sample.clip_id or "", "")
        best_probs = _pick_best_candidate_probs(
            probs,
            strategy=rerank_strategy,
            candidate_scores=sample.candidate_scores,
            candidate_hand_ious=sample.candidate_hand_ious,
            id2label=id2label,
            text_category=text_category,
        )
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        outputs[frame_id] = probs_to_topk(best_probs, id2label, top_k=top_k)
    return outputs


@torch.no_grad()
def infer_fusion_rerank(
    checkpoint_dir: Path,
    samples: list[RerankClassificationSample],
    *,
    batch_size: int,
    top_k: int,
    device: torch.device,
    rerank_strategy: str = "confidence",
) -> dict[str, list[dict]]:
    if load_candidate_crops is None:
        raise RuntimeError("segment_rerank module is unavailable")

    fusion_model, label2id, config = load_fusion_checkpoint(checkpoint_dir)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    image_checkpoint = Path(config["image_checkpoint"])

    outputs: dict[str, list[dict]] = {}
    for sample in tqdm(samples, desc=f"{checkpoint_dir.name} rerank"):
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
        best_fusion_probs = None
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
                best_fusion_probs = fusion_probs

        if best_fusion_probs is None:
            continue
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        outputs[frame_id] = probs_to_topk(best_fusion_probs, id2label, top_k=top_k)
    return outputs


def infer_fusion_checkpoint(
    checkpoint_dir: Path,
    samples: list[ClassificationSample],
    *,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    fusion_model, label2id, config = load_fusion_checkpoint(checkpoint_dir)
    id2label = {idx: name for name, idx in label2id.items()}
    clip_to_category = load_text_category_map()
    image_checkpoint = Path(config["image_checkpoint"])
    image_model_type = config.get("image_model_type", "vision")
    if image_model_type == "efficientnet":
        image_outputs = infer_cnn_checkpoint(
            image_checkpoint,
            samples,
            batch_size=batch_size,
            top_k=len(label2id),
            device=device,
        )
    else:
        image_outputs = infer_vision_checkpoint(
            image_checkpoint,
            samples,
            batch_size=batch_size,
            top_k=len(label2id),
            device=device,
        )

    outputs: dict[str, list[dict]] = {}
    for sample in samples:
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        image_topk = image_outputs.get(frame_id)
        if not image_topk:
            continue
        image_probs = np.zeros(len(label2id), dtype=np.float32)
        for item in image_topk:
            if item["label"] in label2id:
                image_probs[label2id[item["label"]]] = item["score"]
        features = build_fusion_features(
            image_probs,
            clip_to_category.get(sample.clip_id or ""),
            label2id,
        ).reshape(1, -1)
        fusion_probs = fusion_model.predict_proba(features)[0]
        outputs[frame_id] = probs_to_topk(fusion_probs, id2label, top_k=top_k)
    return outputs


def resolve_checkpoint(preset: dict) -> Path:
    if "checkpoint_dir" in preset:
        return Path(preset["checkpoint_dir"])
    return CLASSIFY_CHECKPOINT_DIR / preset["checkpoint_name"]


def run_preset(
    preset_name: str,
    preset: dict,
    *,
    samples: list[ClassificationSample],
    results_dir: Path,
    batch_size: int,
    top_k: int,
    resume: bool,
    device: torch.device,
) -> None:
    checkpoint_dir = resolve_checkpoint(preset)
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint not found for {preset_name}: {checkpoint_dir}")

    output_dir = results_dir / preset["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inferring {preset_name} from {checkpoint_dir} -> {output_dir}")

    rerank = preset.get("rerank", False)
    rerank_strategy = preset.get("rerank_strategy", "confidence")
    model_type = preset.get("model_type")
    if rerank:
        if RerankClassificationSample is None:
            raise RuntimeError("Rerank inference requires segment_rerank module")
        rerank_samples = samples  # type: ignore[assignment]
        if model_type == "fusion":
            predictions = infer_fusion_rerank(
                checkpoint_dir,
                rerank_samples,
                batch_size=batch_size,
                top_k=top_k,
                device=device,
                rerank_strategy=rerank_strategy,
            )
        elif model_type == "efficientnet":
            predictions = infer_cnn_rerank(
                checkpoint_dir,
                rerank_samples,
                top_k=top_k,
                device=device,
                rerank_strategy=rerank_strategy,
            )
        else:
            predictions = infer_vision_rerank(
                checkpoint_dir,
                rerank_samples,
                top_k=top_k,
                device=device,
                rerank_strategy=rerank_strategy,
            )
    elif model_type == "fusion":
        predictions = infer_fusion_checkpoint(
            checkpoint_dir,
            samples,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
        )
    elif model_type == "efficientnet":
        predictions = infer_cnn_checkpoint(
            checkpoint_dir,
            samples,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
        )
    else:
        predictions = infer_vision_checkpoint(
            checkpoint_dir,
            samples,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
        )

    for frame_id, result in tqdm(predictions.items(), desc=f"write {preset_name}"):
        output_path = output_dir / f"{frame_id}.json"
        if resume and output_path.is_file():
            continue
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer trained image classifiers.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument(
        "--preset",
        choices=list(TRAINED_PRESETS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    presets = list(TRAINED_PRESETS.keys()) if args.preset == "all" else [args.preset]

    for preset_name in presets:
        preset = dict(TRAINED_PRESETS[preset_name])
        if args.checkpoint is not None:
            preset["checkpoint_dir"] = str(args.checkpoint)
        use_mask = preset.get("use_mask", True)
        samples = load_classification_samples(use_mask=use_mask)
        run_preset(
            preset_name,
            preset,
            samples=samples,
            results_dir=args.results_dir,
            batch_size=args.batch_size,
            top_k=args.top_k,
            resume=args.resume,
            device=device,
        )


if __name__ == "__main__":
    main()
