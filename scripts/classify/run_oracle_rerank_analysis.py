#!/usr/bin/env python3
"""Oracle rerank analysis: mask-pool coverage vs classification upper bound."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
SEGMENT_DIR = PACKAGE_DIR.parent / "segment"
for path in (SHARED_DIR, PACKAGE_DIR, SEGMENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, RESULTS_DIR
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import build_summary
from geometry import compute_mask_iou
from presets import TRAINED_PRESETS
from segment_rerank import RerankClassificationSample, load_candidate_crops, load_rerank_classification_samples


def load_gt_mask_tensor(row: pd.Series) -> torch.Tensor | None:
    frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
    mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
    if frame_path is None or mask_path is None or not mask_path.is_file():
        return None

    frame = Image.open(frame_path)
    gt = Image.open(mask_path).convert("L")
    if gt.size != frame.size:
        gt = gt.resize(frame.size, Image.NEAREST)
    return torch.from_numpy(np.array(gt) > 0).bool()


def load_pred_mask_tensor(mask_path: Path, frame_size: tuple[int, int]) -> torch.Tensor | None:
    if not mask_path.is_file():
        return None
    mask = Image.open(mask_path).convert("L")
    if mask.size != frame_size:
        mask = mask.resize(frame_size, Image.NEAREST)
    return torch.from_numpy(np.array(mask) > 0).bool()


def analyze_mask_pool(
    samples: list[RerankClassificationSample],
    meta_df: pd.DataFrame,
    *,
    version: str,
) -> pd.DataFrame:
    meta_by_gx_frame: dict[tuple[str, str], pd.Series] = {}
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        meta_by_gx_frame[(str(row["gx_id"]), str(frame_path))] = row

    rows: list[dict] = []
    for sample in tqdm(samples, desc=f"{version} mask pool"):
        row = meta_by_gx_frame.get((sample.gx_id, str(sample.frame_path)))
        if row is None:
            continue
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        gt_mask = load_gt_mask_tensor(row)
        if gt_mask is None:
            continue

        frame_size = Image.open(frame_path).size
        ious: list[float] = []
        for mask_path in sample.candidate_mask_paths:
            pred_mask = load_pred_mask_tensor(mask_path, frame_size)
            if pred_mask is None:
                ious.append(0.0)
            else:
                ious.append(compute_mask_iou(gt_mask, pred_mask))

        if not ious:
            continue

        oracle_idx = int(np.argmax(ious))
        top1_idx = 0
        rows.append(
            {
                "frame_id": frame_result_id(sample.gx_id, sample.frame_path),
                "num_candidates": len(ious),
                "oracle_iou": ious[oracle_idx],
                "top1_score_iou": ious[top1_idx],
                "oracle_rank": oracle_idx + 1,
                "oracle_in_pool": ious[oracle_idx] >= 0.5,
            }
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def classification_oracle_rate(
    samples: list[RerankClassificationSample],
    preset_name: str,
    preset: dict,
    *,
    device: torch.device,
    top_k: int = 5,
) -> float:
    from config import CLASSIFY_CHECKPOINT_DIR

    checkpoint_raw = preset.get("checkpoint_dir")
    if checkpoint_raw:
        checkpoint_dir = Path(checkpoint_raw)
    else:
        checkpoint_dir = CLASSIFY_CHECKPOINT_DIR / preset["checkpoint_name"]
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    model_type = preset.get("model_type")
    correct = 0
    total = 0

    if model_type == "fusion":
        from fusion_model import build_fusion_features, load_fusion_checkpoint
        from training_data import load_text_category_map
        from vision_model import load_vision_checkpoint

        fusion_model, label2id, config = load_fusion_checkpoint(checkpoint_dir)
        id2label = {idx: name for name, idx in label2id.items()}
        clip_to_category = load_text_category_map()
        image_checkpoint = Path(config["image_checkpoint"])
        model, processor, image_label2id, _ = load_vision_checkpoint(image_checkpoint, device)
        image_id2label = {idx: name for name, idx in image_label2id.items()}

        from infer_trained import _encode_crops

        for sample in tqdm(samples, desc=f"oracle {preset_name}"):
            crops = load_candidate_crops(sample)
            if not crops:
                continue
            total += 1
            actual = sample.category_name
            pixel_values = _encode_crops(crops, processor=processor, transform=None).to(device)
            image_probs_batch = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
            oracle_hit = False
            for candidate_probs in image_probs_batch:
                image_probs = np.zeros(len(label2id), dtype=np.float32)
                for class_idx, score in enumerate(candidate_probs):
                    label_name = image_id2label.get(class_idx)
                    if label_name and label_name in label2id:
                        image_probs[label2id[label_name]] = float(score)
                features = build_fusion_features(
                    image_probs,
                    clip_to_category.get(sample.clip_id or ""),
                    label2id,
                ).reshape(1, -1)
                fusion_probs = fusion_model.predict_proba(features)[0]
                pred_idx = int(np.argmax(fusion_probs))
                if id2label[pred_idx] == actual:
                    oracle_hit = True
                    break
            correct += int(oracle_hit)
        return (correct / total * 100) if total else 0.0

    from cnn_model import efficientnet_transform, load_cnn_checkpoint
    from infer_trained import _encode_crops
    from vision_model import load_vision_checkpoint

    if model_type == "efficientnet":
        model, label2id, _ = load_cnn_checkpoint(checkpoint_dir, device)
        transform = efficientnet_transform()
        processor = None
    else:
        model, processor, label2id, _ = load_vision_checkpoint(checkpoint_dir, device)
        transform = None

    id2label = {idx: name for name, idx in label2id.items()}
    for sample in tqdm(samples, desc=f"oracle {preset_name}"):
        crops = load_candidate_crops(sample)
        if not crops:
            continue
        total += 1
        actual = sample.category_name
        pixel_values = _encode_crops(
            crops,
            processor=processor,
            transform=transform,
        ).to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        oracle_hit = any(id2label[int(row.argmax())] == actual for row in probs)
        correct += int(oracle_hit)

    return (correct / total * 100) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle rerank analysis for segment candidates.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--segment-version", default="v7")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "oracle_rerank_analysis.csv",
    )
    parser.add_argument(
        "--output-mask-csv",
        type=Path,
        default=RESULTS_DIR / "oracle_rerank_mask_pool.csv",
    )
    parser.add_argument("--skip-classification-oracle", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_csv(args.meta_csv)
    samples = load_rerank_classification_samples(args.segment_version, meta_csv=args.meta_csv)

    mask_df = analyze_mask_pool(samples, meta_df, version=args.segment_version)
    mask_df.to_csv(args.output_mask_csv, index=False)

    mask_summary = {
        "segment_version": args.segment_version,
        "frames_analyzed": len(mask_df),
        "mean_oracle_iou": round(float(mask_df["oracle_iou"].mean()), 4) if not mask_df.empty else 0.0,
        "mean_top1_iou": round(float(mask_df["top1_score_iou"].mean()), 4) if not mask_df.empty else 0.0,
        "oracle_iou_ge_0.5_pct": round(float(mask_df["oracle_in_pool"].mean()) * 100, 2)
        if not mask_df.empty
        else 0.0,
        "mean_oracle_rank": round(float(mask_df["oracle_rank"].mean()), 2) if not mask_df.empty else 0.0,
    }

    rows = [mask_summary]

    if not args.skip_classification_oracle:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        key_presets = ["efficientnet_b0_crop", "late_fusion_clip_whisper_crop"]
        for base_name in key_presets:
            preset = dict(TRAINED_PRESETS[base_name])
            preset["output_dir"] = f"{preset['output_dir']}_seg_{args.segment_version}"
            oracle_top1 = classification_oracle_rate(
                samples,
                f"{base_name}_oracle",
                preset,
                device=device,
            )
            actual_summary = build_summary(
                [f"{base_name}_seg_{args.segment_version}"],
                meta_csv=args.meta_csv,
                results_dir=CLASSIFICATION_RESULTS_DIR,
                preset_map={f"{base_name}_seg_{args.segment_version}": preset},
            )
            actual_top1 = (
                float(actual_summary["top_1_success_rate"].iloc[0])
                if not actual_summary.empty
                else 0.0
            )
            rows.append(
                {
                    "segment_version": args.segment_version,
                    "model": base_name,
                    "v7_actual_top1": actual_top1,
                    "classification_oracle_top1": round(oracle_top1, 2),
                    "oracle_gap_pct": round(oracle_top1 - actual_top1, 2),
                }
            )

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved mask pool details: {args.output_mask_csv}")
    print(f"Saved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
