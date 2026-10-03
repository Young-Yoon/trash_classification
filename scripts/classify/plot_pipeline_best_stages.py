#!/usr/bin/env python3
"""Plot and export best-per-stage pipeline performance tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, PROJECT_ROOT, RESULTS_DIR
from evaluate import calculate_success_rates

DEMO_DIR = PROJECT_ROOT / "demo"

PRESET_SETS: dict[str, list[str]] = {
    "clip_table": [
        "clip_full_labels_top5",
        "siglip_full_labels_top5",
        "clip_linear_probe_crop",
        "clip_lora_crop",
    ],
    "clip_ablation": [
        "clip_full_labels_top5",
        "siglip_full_labels_top5",
        "clip_linear_probe_crop",
        "clip_lora_crop",
        "efficientnet_b0_crop",
        "late_fusion_clip_whisper_crop",
        "late_fusion_efficientnet_whisper_crop",
    ],
    "major": [
        "clip_linear_probe_crop",
        "siglip_linear_probe_crop",
        "efficientnet_b0_crop",
        "late_fusion_clip_whisper_crop",
    ],
}

METHOD_LABELS = {
    "clip_full_labels_top5": "CLIP ZS",
    "siglip_full_labels_top5": "SigLIP ZS",
    "clip_linear_probe_crop": "CLIP LP",
    "clip_lora_crop": "CLIP LoRA",
    "siglip_linear_probe_crop": "SigLIP",
    "efficientnet_b0_crop": "EfficientNet-B0",
    "late_fusion_clip_whisper_crop": "Fusion (CLIP)",
    "late_fusion_efficientnet_whisper_crop": "Fusion (EN)",
}

METHOD_COLORS = {
    "clip_full_labels_top5": "#A8C5DA",
    "siglip_full_labels_top5": "#E8A0A4",
    "clip_linear_probe_crop": "#4C72B0",
    "clip_lora_crop": "#2F4B7C",
    "siglip_linear_probe_crop": "#C44E52",
    "efficientnet_b0_crop": "#CCB974",
    "late_fusion_clip_whisper_crop": "#64B5CD",
    "late_fusion_efficientnet_whisper_crop": "#4DB6AC",
}

PHASE_BEST_VERSION = {
    "Phase1_Backbone": "v2",
    "Phase2_MaskSelect": "v5",
    "Phase3_RerankPool": "v10",
    "Phase4_RerankStrategy": "v12",
}

METHOD_DESCRIPTIONS = {
    "clip_full_labels_top5": "OpenAI CLIP ViT-B/32 zero-shot (full label set, Top-5)",
    "siglip_full_labels_top5": "SigLIP zero-shot (full label set, Top-5)",
    "clip_linear_probe_crop": "OpenCLIP ViT linear probe on bbox crop",
    "clip_lora_crop": "OpenCLIP ViT LoRA fine-tune on bbox crop",
    "siglip_linear_probe_crop": "SigLIP linear probe on bbox crop",
    "efficientnet_b0_crop": "Supervised EfficientNet-B0 on bbox crop",
    "late_fusion_clip_whisper_crop": "CLIP image probs + Whisper LoRA phrase → fused class",
    "late_fusion_efficientnet_whisper_crop": "EfficientNet probs + Whisper LoRA phrase → fused class",
    "_detection": "SAM3 plastic segmentation vs GT mask (v1 hand ROI / v2 full-image)",
}

STAGE_NOTES = [
    ("Detection", "v1 / v2", "Mask IoU (segmentation quality)"),
    ("Ph.1 Backbone", "v2", "Full-image SAM Top-1 mask"),
    ("Ph.2 Mask select", "v5", "In-hand score Top-1, v2 fallback"),
    ("Ph.3 Rerank pool", "v10", "Top-5 candidate pool rerank"),
    ("Ph.4 Rerank", "v12", "Margin rerank on v10 candidate pool"),
    ("Group eval", "preset-specific", "Any-frame group adjusted final probability"),
]

GROUP_BEST_VERSION = {
    "clip_full_labels_top5": "v15",
    "siglip_full_labels_top5": "v15",
    "clip_linear_probe_crop": "v12",
    "clip_lora_crop": "v12",
    "siglip_linear_probe_crop": "v12",
    "efficientnet_b0_crop": "v15",
    "late_fusion_clip_whisper_crop": "v13",
    "late_fusion_efficientnet_whisper_crop": "v12",
}

METRIC_COLUMN = "adjusted_final_probability"
ZERO_SHOT_PRESETS = frozenset({"clip_full_labels_top5", "siglip_full_labels_top5"})


def load_segment_class_top1_lookup(
    presets: list[str],
    versions: set[str],
    *,
    comparison_long_csv: Path = RESULTS_DIR / "classification_comparison_long.csv",
    zeroshot_summary_csv: Path = RESULTS_DIR / "classification_segment_zeroshot_summary.csv",
) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}

    if comparison_long_csv.is_file():
        df = pd.read_csv(comparison_long_csv)
        for _, row in df.iterrows():
            preset = str(row.get("base_preset", ""))
            if preset not in presets:
                continue
            version = row.get("segment_version")
            if pd.isna(version) or not str(version).strip():
                continue
            version = str(version).strip()
            if version not in versions:
                continue
            lookup[(version, preset)] = float(row["top_1_success_rate"])

    if zeroshot_summary_csv.is_file():
        df = pd.read_csv(zeroshot_summary_csv)
        for _, row in df.iterrows():
            preset = str(row.get("base_preset", ""))
            if preset not in presets:
                continue
            version = str(row["segment_version"])
            if version not in versions:
                continue
            lookup[(version, preset)] = float(row["top_1_success_rate"])

    meta_df = pd.read_csv(META2_CSV)
    for version in sorted(versions):
        for preset in presets:
            if preset not in ZERO_SHOT_PRESETS:
                continue
            if (version, preset) in lookup:
                continue
            output_dir = CLASSIFICATION_RESULTS_DIR / f"{preset}_seg_{version}"
            if not output_dir.is_dir():
                continue
            metrics = calculate_success_rates(output_dir, meta_df)
            lookup[(version, preset)] = float(metrics["top_1"])

    return lookup


def load_zeroshot_lookup(confusion_long_csv: Path) -> dict[tuple[str, str, str], float]:
    if not confusion_long_csv.is_file():
        return {}
    df = pd.read_csv(confusion_long_csv)
    lookup: dict[tuple[str, str, str], float] = {}
    for _, row in df.iterrows():
        preset = str(row.get("base_preset", ""))
        if preset not in ZERO_SHOT_PRESETS:
            continue
        key = (str(row["segment_version"]), str(row["eval_mode"]), preset)
        value = row.get("exact_match_accuracy_pct")
        if pd.notna(value):
            lookup[key] = float(value)
    return lookup


def metric_at(
    frame_df: pd.DataFrame,
    version: str,
    preset: str,
    *,
    column: str = METRIC_COLUMN,
    segment_top1_lookup: dict[tuple[str, str], float] | None = None,
    zeroshot_lookup: dict[tuple[str, str, str], float] | None = None,
) -> tuple[float | None, str]:
    match = frame_df[(frame_df["segment_version"] == version) & (frame_df["base_preset"] == preset)]
    if not match.empty:
        return float(match.iloc[0][column]), "e2e_adjusted_final"

    if preset in ZERO_SHOT_PRESETS and segment_top1_lookup:
        segment_value = segment_top1_lookup.get((version, preset))
        if segment_value is not None:
            return segment_value, "segment_class_top1"

    if preset in ZERO_SHOT_PRESETS and zeroshot_lookup:
        fallback = zeroshot_lookup.get((version, "frame", preset))
        if fallback is not None:
            return fallback, "class_top1_only"
    return None, "missing"


def group_metric_at(
    group_df: pd.DataFrame,
    version: str,
    preset: str,
    *,
    policy: str = "any_frame_any_bbox",
    column: str = METRIC_COLUMN,
    zeroshot_lookup: dict[tuple[str, str, str], float] | None = None,
) -> tuple[float | None, str]:
    match = group_df[
        (group_df["segment_version"] == version)
        & (group_df["base_preset"] == preset)
        & (group_df["aggregation_policy"] == policy)
    ]
    if not match.empty:
        return float(match.iloc[0][column]), "e2e_group_adjusted_final"

    if preset in ZERO_SHOT_PRESETS and zeroshot_lookup:
        fallback = zeroshot_lookup.get((version, "group_majority_vote", preset))
        if fallback is not None:
            return fallback, "group_majority_class_only"
    return None, "missing"


def build_stage_rows(
    frame_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    presets: list[str],
    *,
    segment_top1_lookup: dict[tuple[str, str], float] | None = None,
    zeroshot_lookup: dict[tuple[str, str, str], float] | None = None,
) -> list[dict]:
    rows: list[dict] = []

    for version in ("v1", "v2"):
        seg = segment_summary[segment_summary["version"] == version]
        if seg.empty:
            continue
        row = {
            "stage": f"Detection {version}",
            "stage_kind": "detection",
            "version": version,
            "values": {},
            "metric_types": {},
        }
        row["values"]["_mask_iou"] = float(seg.iloc[0]["top_1_mean_iou"])
        row["metric_types"]["_mask_iou"] = "mask_iou"
        rows.append(row)

    phase_defs = [
        ("Ph.1 Backbone", "Phase1_Backbone"),
        ("Ph.2 Mask select", "Phase2_MaskSelect"),
        ("Ph.3 Rerank pool", "Phase3_RerankPool"),
        ("Ph.4 Rerank", "Phase4_RerankStrategy"),
    ]
    for stage_label, phase in phase_defs:
        version = PHASE_BEST_VERSION[phase]
        row = {
            "stage": stage_label,
            "stage_kind": "classification",
            "version": version,
            "values": {},
            "metric_types": {},
        }
        for preset in presets:
            value, metric_type = metric_at(
                frame_summary,
                version,
                preset,
                segment_top1_lookup=segment_top1_lookup,
                zeroshot_lookup=zeroshot_lookup,
            )
            row["values"][preset] = value
            row["metric_types"][preset] = metric_type
        rows.append(row)

    group_row = {
        "stage": "Group (any frame)",
        "stage_kind": "classification",
        "version": "",
        "values": {},
        "metric_types": {},
    }
    for preset in presets:
        version = GROUP_BEST_VERSION.get(preset, PHASE_BEST_VERSION["Phase4_RerankStrategy"])
        group_row["version"] = group_row["version"] or version
        value, metric_type = group_metric_at(
            group_summary,
            version,
            preset,
            zeroshot_lookup=zeroshot_lookup,
        )
        group_row["values"][preset] = value
        group_row["metric_types"][preset] = metric_type
    rows.append(group_row)
    return rows


def build_stage_table(
    frame_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    presets: list[str],
    *,
    segment_top1_lookup: dict[tuple[str, str], float] | None = None,
    zeroshot_lookup: dict[tuple[str, str, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = build_stage_rows(
        frame_summary,
        group_summary,
        segment_summary,
        presets,
        segment_top1_lookup=segment_top1_lookup,
        zeroshot_lookup=zeroshot_lookup,
    )

    wide_rows = []
    long_rows = []
    for row in rows:
        wide_entry = {
            "stage": row["stage"],
            "stage_kind": row["stage_kind"],
            "version": row["version"],
        }
        if row["stage_kind"] == "detection":
            wide_entry["mask_iou"] = round(row["values"]["_mask_iou"], 1)
            for preset in presets:
                wide_entry[METHOD_LABELS[preset]] = ""
        else:
            wide_entry["mask_iou"] = ""
            for preset in presets:
                label = METHOD_LABELS[preset]
                value = row["values"].get(preset)
                wide_entry[label] = "" if value is None else round(value, 1)
                long_rows.append(
                    {
                        "stage": row["stage"],
                        "version": row["version"],
                        "preset": preset,
                        "method": label,
                        "value_pct": None if value is None else round(value, 2),
                        "metric_type": row["metric_types"].get(preset, "missing"),
                    }
                )
        wide_rows.append(wide_entry)

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def plot_pipeline_best_stages(
    frame_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    output_path: Path,
    presets: list[str],
    *,
    zeroshot_lookup: dict[tuple[str, str, str], float] | None = None,
) -> None:
    stage_rows = build_stage_rows(
        frame_summary,
        group_summary,
        segment_summary,
        presets,
        zeroshot_lookup=zeroshot_lookup,
    )

    chart_rows = []
    for row in stage_rows:
        if row["stage_kind"] == "detection":
            chart_rows.append(
                {
                    "label": f"Det {row['version']}\n(mask IoU)",
                    "kind": "detection",
                    "values": {"_iou": row["values"]["_mask_iou"]},
                }
            )
        else:
            label = row["stage"].replace(" ", "\n", 1)
            if row["version"]:
                label = f"{row['stage']}\n({row['version']})"
            chart_rows.append(
                {
                    "label": label,
                    "kind": "classification",
                    "values": {
                        preset: (row["values"].get(preset) or 0.0)
                        for preset in presets
                    },
                }
            )

    n_rows = len(chart_rows)
    n_presets = len(presets)
    fig_w = max(12.5, 2.0 + n_presets * 1.35)
    fig = plt.figure(figsize=(fig_w, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.05], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_table_algo = fig.add_subplot(gs[1, 0])
    ax_table_algo.axis("off")

    x = np.arange(n_rows)
    width = min(0.16, 0.78 / max(n_presets, 1))
    offsets = np.linspace(-(n_presets - 1) / 2, (n_presets - 1) / 2, n_presets)

    for col_idx, preset in enumerate(presets):
        heights = []
        for row in chart_rows:
            if row["kind"] == "detection":
                heights.append(np.nan)
            else:
                heights.append(row["values"].get(preset, 0.0))
        ax.bar(
            x + offsets[col_idx] * width,
            heights,
            width=width,
            color=METHOD_COLORS[preset],
            label=METHOD_LABELS[preset],
            alpha=0.92,
            zorder=2,
        )

    det_x = [idx for idx, row in enumerate(chart_rows) if row["kind"] == "detection"]
    det_vals = [chart_rows[idx]["values"]["_iou"] for idx in det_x]
    ax.bar(
        det_x,
        det_vals,
        width=width * 1.6,
        color="#8172B2",
        alpha=0.85,
        zorder=2,
    )

    ax.set_ylabel("Accuracy / IoU (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in chart_rows], fontsize=8.5)
    ax.set_ylim(0, 82)
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.set_title(
        "Pipeline best-per-stage performance",
        fontsize=13,
        pad=12,
    )
    legend_labels = [METHOD_LABELS[preset] for preset in presets] + ["Detection mask IoU"]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS[preset], alpha=0.92)
        for preset in presets
    ]
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc="#8172B2", alpha=0.85))
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    ax.text(
        0.99,
        0.02,
        "* CLIP/SigLIP ZS use class Top-1 only when e2e row is missing",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#666666",
    )

    algo_table_rows = [["Algorithm", "Description", "Best config (group eval)"]]
    for preset in presets:
        version = GROUP_BEST_VERSION.get(preset, PHASE_BEST_VERSION["Phase4_RerankStrategy"])
        group_val, metric_type = group_metric_at(
            group_summary,
            version,
            preset,
            zeroshot_lookup=zeroshot_lookup,
        )
        suffix = ""
        if metric_type != "e2e_group_adjusted_final":
            suffix = " *"
        algo_table_rows.append(
            [
                METHOD_LABELS[preset],
                METHOD_DESCRIPTIONS[preset],
                f"{version} ({group_val:.1f}%){suffix}" if group_val is not None else f"{version} (n/a)",
            ]
        )
    det_v1 = float(segment_summary.loc[segment_summary["version"] == "v1", "top_1_mean_iou"].iloc[0])
    det_v2 = float(segment_summary.loc[segment_summary["version"] == "v2", "top_1_mean_iou"].iloc[0])
    algo_table_rows.append(
        [
            "Detection IoU",
            METHOD_DESCRIPTIONS["_detection"],
            f"v1 {det_v1:.1f}% / v2 {det_v2:.1f}%",
        ]
    )

    stage_table_rows = [["Stage", "Version", "What it measures"]]
    stage_table_rows.extend(list(row) for row in STAGE_NOTES)

    algo_table = ax_table_algo.table(
        cellText=algo_table_rows,
        colWidths=[0.14, 0.56, 0.22],
        cellLoc="left",
        loc="upper center",
        bbox=[0.0, 0.52, 1.0, 0.46],
    )
    stage_table = ax_table_algo.table(
        cellText=stage_table_rows,
        colWidths=[0.18, 0.18, 0.54],
        cellLoc="left",
        loc="lower center",
        bbox=[0.0, 0.0, 1.0, 0.46],
    )

    for table in (algo_table, stage_table):
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        for (row_idx, _col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            cell.set_linewidth(0.6)
            if row_idx == 0:
                cell.set_facecolor("#F0F0F0")
                cell.set_text_props(fontweight="bold", color="#222222")
            else:
                cell.set_facecolor("#FFFFFF")
            cell.PAD = 0.05

    ax_table_algo.text(
        0.0,
        1.02,
        "Algorithms",
        transform=ax_table_algo.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#333333",
    )
    ax_table_algo.text(
        0.0,
        0.48,
        "Pipeline stages",
        transform=ax_table_algo.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#333333",
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and export best-per-stage pipeline performance.")
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
        "--segment-summary-csv",
        type=Path,
        default=RESULTS_DIR / "segment_evaluation_summary.csv",
    )
    parser.add_argument(
        "--confusion-long-csv",
        type=Path,
        default=DEMO_DIR / "confusion_matrix_preset_version_grid_long.csv",
    )
    parser.add_argument(
        "--preset-set",
        choices=sorted(PRESET_SETS),
        default="clip_table",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Export CSV tables only (skip chart PNG).",
    )
    parser.add_argument(
        "--output-chart",
        type=Path,
        default=DEMO_DIR / "pipeline_best_stages_chart.png",
    )
    parser.add_argument(
        "--output-table-csv",
        type=Path,
        default=DEMO_DIR / "pipeline_best_stages_table.csv",
    )
    parser.add_argument(
        "--output-table-long-csv",
        type=Path,
        default=DEMO_DIR / "pipeline_best_stages_table_long.csv",
    )
    args = parser.parse_args()

    presets = PRESET_SETS[args.preset_set]
    frame_summary = pd.read_csv(args.frame_summary_csv)
    group_summary = pd.read_csv(args.group_summary_csv)
    segment_summary = pd.read_csv(args.segment_summary_csv)
    zeroshot_lookup = load_zeroshot_lookup(args.confusion_long_csv)

    stage_versions = set(PHASE_BEST_VERSION.values()) | set(GROUP_BEST_VERSION.values())
    segment_top1_lookup = load_segment_class_top1_lookup(presets, stage_versions)

    wide_df, long_df = build_stage_table(
        frame_summary,
        group_summary,
        segment_summary,
        presets,
        segment_top1_lookup=segment_top1_lookup,
        zeroshot_lookup=zeroshot_lookup,
    )
    args.output_table_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.output_table_csv, index=False)
    long_df.to_csv(args.output_table_long_csv, index=False)

    if not args.table_only:
        plot_pipeline_best_stages(
            frame_summary,
            group_summary,
            segment_summary,
            args.output_chart,
            presets,
            zeroshot_lookup=zeroshot_lookup,
        )
        print(f"Saved best-stages chart: {args.output_chart}")

    print(f"Saved stage table: {args.output_table_csv}")
    print(f"Saved long stage table: {args.output_table_long_csv}")
    print(wide_df.to_string(index=False))


if __name__ == "__main__":
    main()
