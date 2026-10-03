#!/usr/bin/env python3
"""Build segment version comparison table (Top-1 classification + pipeline metrics)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_SEGMENT_SUMMARY_CSV, RESULTS_DIR
from segment_masks import ALL_SEGMENT_VERSIONS, SEGMENT_VERSION_DESCRIPTIONS

METHOD_LABELS = {
    "clip_linear_probe_crop": "CLIP linear probe",
    "clip_lora_crop": "CLIP LoRA",
    "siglip_linear_probe_crop": "SigLIP linear probe",
    "siglip_lora_crop": "SigLIP LoRA",
    "efficientnet_b0_crop": "EfficientNet-B0",
    "late_fusion_clip_whisper_crop": "Late fusion (CLIP)",
    "late_fusion_efficientnet_whisper_crop": "Late fusion (EN)",
}

KEY_MODELS = list(METHOD_LABELS.keys())

VERSION_ORDER = list(ALL_SEGMENT_VERSIONS)

VERSION_PHASE = {
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

PHASE_COLORS = {
    "Phase1_Backbone": "#4C72B0",
    "Phase2_MaskSelect": "#55A868",
    "Phase3_RerankPool": "#C44E52",
    "Phase4_RerankStrategy": "#8172B2",
}


def build_comparison_table(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    for _, row in summary_df.iterrows():
        version = row["segment_version"]
        base = row["base_preset"]
        long_rows.append(
            {
                "segment_version": version,
                "method": METHOD_LABELS.get(base, base),
                "base_preset": base,
                "top_1": float(row["top_1_success_rate"]),
                "top_3": float(row["top_3_success_rate"]),
                "top_5": float(row["top_5_success_rate"]),
                "description": SEGMENT_VERSION_DESCRIPTIONS.get(version, ""),
            }
        )
    long_df = pd.DataFrame(long_rows)

    wide_rows = []
    for base in KEY_MODELS:
        method = METHOD_LABELS[base]
        row = {"method": method, "base_preset": base}
        values = []
        for version in VERSION_ORDER:
            match = long_df[(long_df["segment_version"] == version) & (long_df["base_preset"] == base)]
            top_1 = float(match["top_1"].iloc[0]) if not match.empty else None
            row[f"{version}_top_1"] = top_1
            if top_1 is not None:
                values.append(top_1)
        row["best_version"] = VERSION_ORDER[values.index(max(values))] if values else ""
        row["best_top_1"] = max(values) if values else None
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    return wide_df, long_df


def build_pipeline_wide_table(pipeline_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for version in VERSION_ORDER:
        phase, phase_order = VERSION_PHASE[version]
        version_rows = pipeline_summary[pipeline_summary["segment_version"] == version]
        if version_rows.empty:
            continue
        row: dict = {
            "segment_version": version,
            "phase": phase,
            "phase_order": phase_order,
            "description": SEGMENT_VERSION_DESCRIPTIONS.get(version, ""),
        }
        non_rerank = version not in {"v7", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}
        if non_rerank and not version_rows.empty:
            row["mask_success_rate"] = round(version_rows["mask_success_rate"].iloc[0], 2)
        for base in KEY_MODELS:
            match = version_rows[version_rows["base_preset"] == base]
            if match.empty:
                continue
            m = match.iloc[0]
            prefix = base.replace("_crop", "").replace("_clip_whisper", "")
            if not non_rerank:
                row[f"{prefix}_mask_rate"] = m["mask_success_rate"]
            row[f"{prefix}_class_top1"] = m["class_top1_rate"]
            row[f"{prefix}_class_given_mask"] = m["class_given_mask_rate"]
            row[f"{prefix}_iou0_cls_ok"] = m["iou_zero_class_ok_rate"]
            row[f"{prefix}_final"] = m["final_probability"]
            row[f"{prefix}_adjusted"] = m["adjusted_final_probability"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase_order", "segment_version"]).reset_index(drop=True)


def plot_pipeline_chart(pipeline_summary: pd.DataFrame, output_path: Path) -> None:
    highlight_presets = [
        ("late_fusion_clip_whisper_crop", "Late fusion"),
        ("efficientnet_b0_crop", "EfficientNet-B0"),
    ]
    fig, axes = plt.subplots(len(highlight_presets), 1, figsize=(14, 5 * len(highlight_presets)), sharex=True)
    if len(highlight_presets) == 1:
        axes = [axes]

    for ax, (preset, label) in zip(axes, highlight_presets):
        subset = pipeline_summary[pipeline_summary["base_preset"] == preset].copy()
        subset["phase"] = subset["segment_version"].map(lambda v: VERSION_PHASE[v][0])
        subset["phase_order"] = subset["segment_version"].map(lambda v: VERSION_PHASE[v][1])
        subset = subset.sort_values(["phase_order", "segment_version"])
        colors = [PHASE_COLORS.get(phase, "#999999") for phase in subset["phase"]]
        x = range(len(subset))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        bar_specs = [
            (offsets[0], subset["class_top1_rate"], "#DD8452", "Class only"),
            (offsets[1], subset["mask_success_rate"], "#4C72B0", "Mask"),
            (offsets[2], subset["class_given_mask_rate"], "#8C8C8C", "Class | mask"),
            (offsets[3], subset["final_probability"], colors, "Final"),
        ]
        for idx, (offset, values, color, bar_label) in enumerate(bar_specs):
            ax.bar(
                [i + offset * width for i in x],
                values,
                width=width,
                color=color if idx < 3 else colors,
                alpha=0.95 if idx == 3 else 0.85,
                label=bar_label,
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(subset["segment_version"], rotation=45, ha="right")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"End-to-end pipeline metrics — {label}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left")

    phase_patches = [
        plt.Line2D([0], [0], color=color, lw=6, label=phase)
        for phase, color in PHASE_COLORS.items()
    ]
    fig.legend(handles=phase_patches, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment version comparison table.")
    parser.add_argument("--input-csv", type=Path, default=CLASSIFICATION_SEGMENT_SUMMARY_CSV)
    parser.add_argument(
        "--pipeline-summary-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_summary.csv",
    )
    parser.add_argument(
        "--output-wide-csv",
        type=Path,
        default=RESULTS_DIR / "classification_segment_versions_comparison.csv",
    )
    parser.add_argument(
        "--output-long-csv",
        type=Path,
        default=RESULTS_DIR / "classification_segment_versions_comparison_long.csv",
    )
    parser.add_argument(
        "--output-pipeline-wide-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_comparison_wide.csv",
    )
    parser.add_argument(
        "--output-pipeline-chart",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_comparison.png",
    )
    parser.add_argument(
        "--include-pipeline-metrics",
        action="store_true",
        help="Build pipeline wide table and chart from pipeline summary CSV.",
    )
    args = parser.parse_args()

    summary_df = pd.read_csv(args.input_csv)
    wide_df, long_df = build_comparison_table(summary_df)

    args.output_wide_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.output_wide_csv, index=False)
    long_df.to_csv(args.output_long_csv, index=False)

    print(wide_df[["method", "best_version", "best_top_1"]].to_string(index=False))
    print(f"\nSaved wide table: {args.output_wide_csv}")
    print(f"Saved long table: {args.output_long_csv}")

    if args.include_pipeline_metrics and args.pipeline_summary_csv.is_file():
        pipeline_summary = pd.read_csv(args.pipeline_summary_csv)
        pipeline_wide = build_pipeline_wide_table(pipeline_summary)
        pipeline_wide.to_csv(args.output_pipeline_wide_csv, index=False)
        plot_pipeline_chart(pipeline_summary, args.output_pipeline_chart)
        print(f"Saved pipeline wide table: {args.output_pipeline_wide_csv}")
        print(f"Saved pipeline chart: {args.output_pipeline_chart}")
    elif args.include_pipeline_metrics:
        print(f"Pipeline summary not found: {args.pipeline_summary_csv}")


if __name__ == "__main__":
    main()
