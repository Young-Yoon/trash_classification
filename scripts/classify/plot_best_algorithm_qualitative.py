#!/usr/bin/env python3
"""Export qualitative examples for the best end-to-end algorithm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
SEGMENT_DIR_PKG = PACKAGE_DIR.parent / "segment"
for path in (SHARED_DIR, PACKAGE_DIR, SEGMENT_DIR_PKG):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_RESULTS_DIR, CLASSIFY_CHECKPOINT_DIR, META2_CSV, RESULTS_DIR, SEGMENT_DIR
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import predicted_label_name, result_json_path
from image_utils import load_classification_image
from presets import TRAINED_PRESETS
from rerank_visualization import (
    compute_after_candidate_index,
    is_rerank_version,
    render_rerank_before_after,
)
from segment_masks import SEGMENT_BBOX_PADDING, SEGMENT_CROP_MODES
from segment_rerank import load_rerank_classification_samples


METHOD_LABELS = {
    "clip_linear_probe_crop": "CLIP linear probe",
    "clip_lora_crop": "CLIP LoRA",
    "siglip_linear_probe_crop": "SigLIP linear probe",
    "siglip_lora_crop": "SigLIP LoRA",
    "efficientnet_b0_crop": "EfficientNet-B0",
    "late_fusion_clip_whisper_crop": "Late fusion (CLIP)",
    "late_fusion_efficientnet_whisper_crop": "Late fusion (EfficientNet)",
}


GALLERY_CATEGORIES = {
    "mask_success": {
        "label": "Mask success",
        "condition": lambda df: df["mask_ok"],
        "sort_by": "selected_bbox_iou",
        "ascending": False,
    },
    "mask_fail": {
        "label": "Mask fail",
        "condition": lambda df: ~df["mask_ok"],
        "sort_by": "selected_bbox_iou",
        "ascending": True,
    },
    "class_success": {
        "label": "Class success",
        "condition": lambda df: df["class_top1_ok"],
        "sort_by": "selected_bbox_iou",
        "ascending": False,
    },
    "class_fail": {
        "label": "Class fail",
        "condition": lambda df: ~df["class_top1_ok"],
        "sort_by": "selected_bbox_iou",
        "ascending": False,
        "prefer_mask_ok": True,
    },
}


def segment_output_dir(base_output_dir: str, version: str) -> str:
    return f"{base_output_dir}_seg_{version}"


def pick_best_algorithm(summary_df: pd.DataFrame) -> tuple[str, str]:
    ranked = summary_df.sort_values(
        ["adjusted_final_probability", "base_preset"],
        ascending=[False, False],
    )
    top_score = float(ranked.iloc[0]["adjusted_final_probability"])
    tied = ranked[ranked["adjusted_final_probability"] == top_score]
    for preset in (
        "late_fusion_efficientnet_whisper_crop",
        "late_fusion_clip_whisper_crop",
    ):
        match = tied[tied["base_preset"] == preset]
        if not match.empty:
            row = match.iloc[0]
            return str(row["segment_version"]), str(row["base_preset"])
    row = tied.iloc[0]
    return str(row["segment_version"]), str(row["base_preset"])


def load_predicted_label(row: pd.Series, output_dir: Path) -> str:
    json_path = result_json_path(row, output_dir)
    if not json_path.is_file():
        return "missing"
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and payload:
        return predicted_label_name(payload[0]["label"])
    if isinstance(payload, dict) and "label" in payload:
        return predicted_label_name(payload["label"])
    return "missing"


def overlay_mask_on_frame(frame: Image.Image, mask_path: Path, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.array(frame.convert("RGB"), dtype=np.float32)
    if mask_path.is_file():
        mask = Image.open(mask_path).convert("L")
        if mask.size != frame.size:
            mask = mask.resize(frame.size, Image.NEAREST)
        mask_arr = np.array(mask) > 0
        alpha = 0.45
        for channel in range(3):
            rgb[:, :, channel] = np.where(
                mask_arr,
                rgb[:, :, channel] * (1 - alpha) + color[channel] * alpha,
                rgb[:, :, channel],
            )
    return rgb.astype(np.uint8)


def render_example(
    row: pd.Series,
    frame_row: pd.Series,
    *,
    version: str,
    base_preset: str,
    results_dir: Path,
    output_path: Path,
    rerank_sample=None,
    device: torch.device | None = None,
) -> dict | None:
    frame_path = resolve_meta_asset_path(frame_row.get("copied_frame_path"))
    gt_mask_path = resolve_meta_asset_path(frame_row.get("copied_mask_path"))
    if frame_path is None or not frame_path.is_file():
        return None

    gt_label = str(frame_row["category_name"])

    if is_rerank_version(version) and rerank_sample is not None and device is not None:
        before_idx = 0
        after_idx = compute_after_candidate_index(
            rerank_sample,
            base_preset,
            version,  # type: ignore[arg-type]
            device=device,
        )
        cls_output_dir = results_dir / segment_output_dir(TRAINED_PRESETS[base_preset]["output_dir"], version)
        pred_label = load_predicted_label(frame_row, cls_output_dir)
        meta = render_rerank_before_after(
            rerank_sample,
            version=version,  # type: ignore[arg-type]
            base_preset=base_preset,
            gt_mask_path=Path(gt_mask_path) if gt_mask_path else Path(),
            gt_label=gt_label,
            before_idx=before_idx,
            after_idx=after_idx,
            output_path=output_path,
            device=device,
            pred_label=pred_label,
        )
        meta["mask_ok"] = bool(row["mask_ok"])
        meta["class_top1_ok"] = bool(row["class_top1_ok"])
        meta["selected_bbox_iou"] = row.get("selected_bbox_iou")
        return meta

    crop_mode = SEGMENT_CROP_MODES[version]  # type: ignore[index]
    padding = SEGMENT_BBOX_PADDING[version]  # type: ignore[index]

    pred_mask_path = Path(str(row.get("pred_mask_path", "") or ""))
    if not pred_mask_path.is_file():
        pred_mask_path = SEGMENT_DIR / f"predicted_masks_{version}" / f"{row['frame_id']}.png"
    mask_for_crop = pred_mask_path if pred_mask_path.is_file() else gt_mask_path

    crop = load_classification_image(
        frame_path,
        mask_for_crop,
        crop_mode=crop_mode,
        bbox_padding_ratio=padding,
    )

    frame = Image.open(frame_path).convert("RGB")
    gt_overlay = overlay_mask_on_frame(frame, Path(gt_mask_path) if gt_mask_path else Path(), (0, 200, 0))
    pred_overlay = overlay_mask_on_frame(
        frame,
        pred_mask_path if pred_mask_path.is_file() else Path(),
        (255, 80, 80),
    )

    cls_output_dir = results_dir / segment_output_dir(TRAINED_PRESETS[base_preset]["output_dir"], version)
    pred_label = load_predicted_label(frame_row, cls_output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(crop)
    axes[0].set_title("Crop")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT mask")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Pred mask")

    status = []
    if row["mask_ok"]:
        status.append("mask OK")
    else:
        status.append("mask fail")
    if row["class_top1_ok"]:
        status.append("class OK")
    else:
        status.append("class fail")

    title = (
        f"{row['frame_id']}\n"
        f"GT: {gt_label} | Pred: {pred_label}\n"
        f"mask IoU={row.get('selected_mask_iou', row.get('selected_iou', 0)):.3f} | "
        f"bbox IoU={row.get('selected_bbox_iou', 0):.3f} | {', '.join(status)}"
    )
    fig.suptitle(title, fontsize=10)
    for ax in axes:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=(0, 0.78, 0), label="GT mask"),
        Patch(facecolor=(1, 0.31, 0.31), label="Pred mask"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {
        "mask_ok": bool(row["mask_ok"]),
        "class_top1_ok": bool(row["class_top1_ok"]),
        "selected_bbox_iou": row.get("selected_bbox_iou"),
    }


def sample_frames(
    group: pd.DataFrame,
    condition,
    *,
    n: int = 5,
    sort_by: str | None = None,
    ascending: bool = True,
    prefer_mask_ok: bool | None = None,
) -> pd.DataFrame:
    subset = group[condition(group)].copy()
    if subset.empty:
        return subset
    subset = subset.drop_duplicates(subset=["frame_id"])
    if prefer_mask_ok is not None and "mask_ok" in subset.columns:
        preferred = subset[subset["mask_ok"] == prefer_mask_ok]
        if len(preferred) >= n:
            subset = preferred
    if sort_by and sort_by in subset.columns:
        subset = subset.sort_values(sort_by, ascending=ascending)
    pool = subset.head(max(n * 4, n))
    if len(pool) <= n:
        return pool
    return pool.sample(n=n, random_state=42)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot qualitative examples for best algorithm.")
    parser.add_argument(
        "--per-frame-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_per_frame.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_summary.csv",
    )
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--base-preset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument(
        "--output-name",
        type=str,
        default="mask_class_qualitative",
        help="Subdirectory name under results/examples/.",
    )
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)
    per_frame_df = pd.read_csv(args.per_frame_csv)
    meta_df = pd.read_csv(args.meta_csv)
    meta_by_frame = {}
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        meta_by_frame[frame_result_id(str(row["gx_id"]), frame_path)] = row

    if args.version and args.base_preset:
        version, base_preset = args.version, args.base_preset
    else:
        version, base_preset = pick_best_algorithm(summary_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rerank_samples_by_frame: dict[str, object] = {}
    if is_rerank_version(version):
        rerank_samples = load_rerank_classification_samples(
            version,  # type: ignore[arg-type]
            meta_csv=args.meta_csv,
            crop_mode=SEGMENT_CROP_MODES[version],  # type: ignore[index]
            bbox_padding_ratio=SEGMENT_BBOX_PADDING[version],  # type: ignore[index]
        )
        rerank_samples_by_frame = {
            frame_result_id(sample.gx_id, sample.frame_path): sample for sample in rerank_samples
        }

    subset = per_frame_df[
        (per_frame_df["segment_version"] == version) & (per_frame_df["base_preset"] == base_preset)
    ]
    method = METHOD_LABELS.get(base_preset, base_preset)
    output_root = args.output_dir or (
        RESULTS_DIR / "examples" / f"{args.output_name}_{version}_{base_preset}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    n = args.n_examples
    for category, spec in GALLERY_CATEGORIES.items():
        out_dir = output_root / category
        for stale in out_dir.glob("*.png"):
            stale.unlink()
        picked = sample_frames(
            subset,
            spec["condition"],
            n=n,
            sort_by=spec.get("sort_by"),
            ascending=bool(spec.get("ascending", True)),
            prefer_mask_ok=spec.get("prefer_mask_ok"),
        )
        for idx, (_, row) in enumerate(picked.iterrows(), start=1):
            frame_row = meta_by_frame.get(row["frame_id"])
            if frame_row is None:
                continue
            output_path = out_dir / f"{idx:02d}_{row['frame_id']}.png"
            extra = render_example(
                row,
                frame_row,
                version=version,
                base_preset=base_preset,
                results_dir=args.results_dir,
                output_path=output_path,
                rerank_sample=rerank_samples_by_frame.get(row["frame_id"]),
                device=device,
            )
            if extra is None:
                continue
            entry = {
                    "category": category,
                    "category_label": spec["label"],
                    "frame_id": row["frame_id"],
                    "category_name": row.get("category_name"),
                    "selected_mask_iou": row.get("selected_mask_iou", row.get("selected_iou")),
                    "selected_bbox_iou": row.get("selected_bbox_iou"),
                    "mask_ok": bool(row["mask_ok"]),
                    "class_top1_ok": bool(row["class_top1_ok"]),
                    "output_path": str(output_path),
                }
            if extra:
                entry.update(extra)
            manifest_rows.append(entry)

    manifest = {
        "segment_version": version,
        "base_preset": base_preset,
        "method": method,
        "output_dir": str(output_root),
        "examples": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Algorithm: {version} + {method}")
    print(f"Saved qualitative gallery to {output_root}")
    for category, spec in GALLERY_CATEGORIES.items():
        count = sum(1 for item in manifest_rows if item["category"] == category)
        print(f"  {spec['label']}: {count} examples -> {output_root / category}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
