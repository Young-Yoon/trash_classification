#!/usr/bin/env python3
"""Build unified classification comparison table (GT / segment v1 / v2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import (
    CLASSIFICATION_SEGMENT_SUMMARY_CSV,
    CLASSIFICATION_SUMMARY_CSV,
    RESULTS_DIR,
)
from presets import PRESETS, TRAINED_PRESETS

ZERO_SHOT_GT = ["clip_full_labels_top5", "siglip_full_labels_top5"]
TRAINED_GT = list(TRAINED_PRESETS.keys())
ZERO_SHOT_SEGMENT = ["clip_full_labels_top5", "siglip_full_labels_top5"]
TRAINED_SEGMENT = list(TRAINED_PRESETS.keys())

METHOD_LABELS = {
    "clip_full_labels_top5": "CLIP zero-shot",
    "siglip_full_labels_top5": "SigLIP zero-shot",
    "clip_linear_probe_crop": "CLIP linear probe",
    "clip_lora_crop": "CLIP LoRA",
    "siglip_linear_probe_crop": "SigLIP linear probe",
    "siglip_lora_crop": "SigLIP LoRA",
    "efficientnet_b0_crop": "EfficientNet-B0",
    "late_fusion_clip_whisper_crop": "Late fusion (CLIP+Whisper)",
}


def load_gt_rows(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    rows = []
    for _, row in df.iterrows():
        preset = row["preset"]
        if preset in ZERO_SHOT_GT:
            method_type = "zero_shot"
        elif preset in TRAINED_GT:
            method_type = "supervised"
        else:
            continue
        rows.append(
            {
                "method_type": method_type,
                "method": METHOD_LABELS.get(preset, preset),
                "base_preset": preset,
                "mask_source": "gt",
                "segment_version": "",
                "top_1_success_rate": row["top_1_success_rate"],
                "top_3_success_rate": row["top_3_success_rate"],
                "top_5_success_rate": row["top_5_success_rate"],
            }
        )
    return pd.DataFrame(rows)


def format_metric_triplet(top_1: float, top_3: float, top_5: float) -> str:
    return f"{top_1:.2f} / {top_3:.2f} / {top_5:.2f}"


def build_wide_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["top_1_success_rate", "top_3_success_rate", "top_5_success_rate"]
    wide_rows = []
    for (method_type, method, base_preset), group in combined.groupby(
        ["method_type", "method", "base_preset"], sort=False, observed=True
    ):
        by_mask = {row["mask_source"]: row for _, row in group.iterrows()}
        row = {
            "method_type": method_type,
            "method": method,
            "base_preset": base_preset,
        }
        for mask_source, label in (
            ("gt", "gt"),
            ("segment_v1", "segment_v1"),
            ("segment_v2", "segment_v2"),
            ("segment_v5", "segment_v5"),
        ):
            metrics = by_mask.get(mask_source)
            if metrics is None:
                row[f"{label}_top_1"] = None
                row[f"{label}_top_3"] = None
                row[f"{label}_top_5"] = None
                row[f"{label}_top_1_3_5"] = ""
            else:
                top_1 = float(metrics["top_1_success_rate"])
                top_3 = float(metrics["top_3_success_rate"])
                top_5 = float(metrics["top_5_success_rate"])
                row[f"{label}_top_1"] = top_1
                row[f"{label}_top_3"] = top_3
                row[f"{label}_top_5"] = top_5
                row[f"{label}_top_1_3_5"] = format_metric_triplet(top_1, top_3, top_5)
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    method_order = list(METHOD_LABELS.values())
    type_order = ["zero_shot", "supervised"]
    wide_df["method"] = pd.Categorical(wide_df["method"], categories=method_order, ordered=True)
    wide_df["method_type"] = pd.Categorical(
        wide_df["method_type"], categories=type_order, ordered=True
    )
    return wide_df.sort_values(["method_type", "method"]).reset_index(drop=True)


def load_segment_rows(segment_csv: Path) -> pd.DataFrame:
    if not segment_csv.is_file():
        return pd.DataFrame()
    df = pd.read_csv(segment_csv)
    rows = []
    for _, row in df.iterrows():
        base = row.get("base_preset", row["preset"])
        if base in ZERO_SHOT_SEGMENT:
            method_type = "zero_shot"
        elif base in TRAINED_SEGMENT:
            method_type = "supervised"
        else:
            method_type = "supervised"
        rows.append(
            {
                "method_type": method_type,
                "method": METHOD_LABELS.get(base, base),
                "base_preset": base,
                "mask_source": f"segment_{row['segment_version']}",
                "segment_version": row["segment_version"],
                "top_1_success_rate": row["top_1_success_rate"],
                "top_3_success_rate": row["top_3_success_rate"],
                "top_5_success_rate": row["top_5_success_rate"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified classification comparison CSV.")
    parser.add_argument("--gt-summary", type=Path, default=CLASSIFICATION_SUMMARY_CSV)
    parser.add_argument(
        "--segment-summary",
        type=Path,
        default=CLASSIFICATION_SEGMENT_SUMMARY_CSV,
    )
    parser.add_argument(
        "--segment-zeroshot-summary",
        type=Path,
        default=RESULTS_DIR / "classification_segment_zeroshot_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "classification_comparison.csv",
    )
    parser.add_argument(
        "--output-long-csv",
        type=Path,
        default=RESULTS_DIR / "classification_comparison_long.csv",
    )
    args = parser.parse_args()

    gt_df = load_gt_rows(args.gt_summary)
    segment_df = load_segment_rows(args.segment_summary)
    zeroshot_segment_df = load_segment_rows(args.segment_zeroshot_summary)

    combined = pd.concat([gt_df, segment_df, zeroshot_segment_df], ignore_index=True)
    method_order = list(METHOD_LABELS.values())
    mask_order = ["gt", "segment_v1", "segment_v2", "segment_v5"]
    combined["method"] = pd.Categorical(combined["method"], categories=method_order, ordered=True)
    combined["mask_source"] = pd.Categorical(combined["mask_source"], categories=mask_order, ordered=True)
    combined = combined.sort_values(["method_type", "method", "mask_source"]).reset_index(drop=True)
    wide_df = build_wide_comparison(combined)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.output_csv, index=False)
    combined.to_csv(args.output_long_csv, index=False)

    display_cols = [
        "method_type",
        "method",
        "gt_top_1_3_5",
        "segment_v1_top_1_3_5",
        "segment_v2_top_1_3_5",
        "segment_v5_top_1_3_5",
    ]
    print(wide_df[display_cols].to_string(index=False))
    print(f"\nSaved wide comparison: {args.output_csv}")
    print(f"Saved long comparison: {args.output_long_csv}")


if __name__ == "__main__":
    main()
