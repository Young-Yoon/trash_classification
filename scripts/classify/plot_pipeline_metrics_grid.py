#!/usr/bin/env python3
"""Plot pipeline metrics grid: 4 metrics × 3 eval modes, 7 classifiers per segment version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import PROJECT_ROOT, RESULTS_DIR

DEMO_DIR = PROJECT_ROOT / "demo"
from segment_masks import ALL_SEGMENT_VERSIONS

PRESET_ORDER = [
    "clip_linear_probe_crop",
    "clip_lora_crop",
    "siglip_linear_probe_crop",
    "siglip_lora_crop",
    "efficientnet_b0_crop",
    "late_fusion_clip_whisper_crop",
    "late_fusion_efficientnet_whisper_crop",
]

METHOD_LABELS = {
    "clip_linear_probe_crop": "CLIP linear",
    "clip_lora_crop": "CLIP LoRA",
    "siglip_linear_probe_crop": "SigLIP linear",
    "siglip_lora_crop": "SigLIP LoRA",
    "efficientnet_b0_crop": "EfficientNet-B0",
    "late_fusion_clip_whisper_crop": "Late fusion (CLIP)",
    "late_fusion_efficientnet_whisper_crop": "Late fusion (EN)",
}

PRESET_COLORS = {
    "clip_linear_probe_crop": "#4C72B0",
    "clip_lora_crop": "#55A868",
    "siglip_linear_probe_crop": "#C44E52",
    "siglip_lora_crop": "#8172B2",
    "efficientnet_b0_crop": "#CCB974",
    "late_fusion_clip_whisper_crop": "#64B5CD",
    "late_fusion_efficientnet_whisper_crop": "#937860",
}

VERSION_ORDER = list(ALL_SEGMENT_VERSIONS)

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

PHASE_COLORS = {
    "Phase1_Backbone": "#4C72B0",
    "Phase2_MaskSelect": "#55A868",
    "Phase3_RerankPool": "#C44E52",
    "Phase4_RerankStrategy": "#8172B2",
}

PHASE_LABELS = {
    "Phase1_Backbone": "Phase 1: Backbone",
    "Phase2_MaskSelect": "Phase 2: Mask select",
    "Phase3_RerankPool": "Phase 3: Rerank pool",
    "Phase4_RerankStrategy": "Phase 4: Rerank strategy",
}

PHASE_ORDER = list(PHASE_LABELS.keys())

PHASE_BAND_ALPHA = 0.10

METRIC_ROWS = [
    ("Class only", "class_top1_rate", "class_success_rate"),
    ("Mask", "mask_success_rate", "mask_success_rate"),
    ("Class | mask", "class_given_mask_rate", "class_given_mask_rate"),
    ("Final", "final_probability", "final_probability"),
]

EVAL_COLUMNS = [
    ("End-to-end\n(frame)", None),
    ("Majority\n(group)", "majority_vote_any_bbox"),
    ("Any frame\n(group)", "any_frame_any_bbox"),
]


def metric_values(
    df: pd.DataFrame,
    *,
    version: str,
    preset: str,
    metric_col: str,
) -> float:
    match = df[(df["segment_version"] == version) & (df["base_preset"] == preset)]
    if match.empty:
        return 0.0
    return float(match.iloc[0][metric_col])


def build_phase_spans(version_order: list[str]) -> list[tuple[str, int, int]]:
    if not version_order:
        return []
    spans: list[tuple[str, int, int]] = []
    start = 0
    current_phase = VERSION_PHASE[version_order[0]][0]
    for idx, version in enumerate(version_order[1:], start=1):
        phase = VERSION_PHASE[version][0]
        if phase != current_phase:
            spans.append((current_phase, start, idx - 1))
            start = idx
            current_phase = phase
    spans.append((current_phase, start, len(version_order) - 1))
    return spans


def add_phase_background(ax: plt.Axes, phase_spans: list[tuple[str, int, int]]) -> None:
    for phase, start_idx, end_idx in phase_spans:
        ax.axvspan(
            start_idx - 0.5,
            end_idx + 0.5,
            color=PHASE_COLORS[phase],
            alpha=PHASE_BAND_ALPHA,
            zorder=0,
        )


def plot_pipeline_metrics_grid(
    frame_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    n_versions = len(VERSION_ORDER)
    n_presets = len(PRESET_ORDER)
    x = np.arange(n_versions)
    width = 0.10
    phase_spans = build_phase_spans(VERSION_ORDER)

    fig, axes = plt.subplots(
        len(METRIC_ROWS),
        len(EVAL_COLUMNS),
        figsize=(20, 14),
        sharex=True,
    )

    for row_idx, (row_label, frame_metric, group_metric) in enumerate(METRIC_ROWS):
        for col_idx, (col_label, group_policy) in enumerate(EVAL_COLUMNS):
            ax = axes[row_idx, col_idx]
            add_phase_background(ax, phase_spans)
            source_df = frame_summary if group_policy is None else group_summary[
                group_summary["aggregation_policy"] == group_policy
            ]
            metric_col = frame_metric if group_policy is None else group_metric

            for preset_idx, preset in enumerate(PRESET_ORDER):
                values = [
                    metric_values(
                        source_df,
                        version=version,
                        preset=preset,
                        metric_col=metric_col,
                    )
                    for version in VERSION_ORDER
                ]
                offset = (preset_idx - (n_presets - 1) / 2) * width
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    color=PRESET_COLORS[preset],
                    label=METHOD_LABELS[preset],
                    alpha=0.92,
                    zorder=2,
                )

            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25, zorder=1)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=11, pad=8)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11)
            if row_idx == len(METRIC_ROWS) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(VERSION_ORDER, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

    classifier_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PRESET_COLORS[preset], alpha=0.92)
        for preset in PRESET_ORDER
    ]
    classifier_labels = [METHOD_LABELS[preset] for preset in PRESET_ORDER]
    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[phase], alpha=PHASE_BAND_ALPHA + 0.35)
        for phase in PHASE_ORDER
    ]
    phase_labels = [PHASE_LABELS[phase] for phase in PHASE_ORDER]
    separator = Line2D([], [], linestyle="none", marker="", label="|", alpha=0.0)

    fig.legend(
        classifier_handles + [separator] + phase_handles,
        classifier_labels + ["|"] + phase_labels,
        loc="lower center",
        ncol=len(PRESET_ORDER) + 1 + len(PHASE_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9,
        columnspacing=1.0,
        handlelength=1.2,
        handleheight=1.0,
    )
    fig.suptitle(
        "Pipeline metrics by segmentation version (7 classifiers per version)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 4×3 pipeline metrics grid chart.")
    parser.add_argument(
        "--frame-summary-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_summary.csv",
    )
    parser.add_argument(
        "--group-summary-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_group_summary.csv",
    )
    parser.add_argument(
        "--output-chart",
        type=Path,
        default=DEMO_DIR / "pipeline_end_to_end_metrics_grid.png",
    )
    args = parser.parse_args()

    frame_summary = pd.read_csv(args.frame_summary_csv)
    group_summary = pd.read_csv(args.group_summary_csv)
    plot_pipeline_metrics_grid(frame_summary, group_summary, args.output_chart)
    print(f"Saved metrics grid chart: {args.output_chart}")


if __name__ == "__main__":
    main()
