#!/usr/bin/env python3
"""Build preset × segment-version exact-match grid CSV for confusion matrix metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_pipeline_end_to_end import segment_output_dir
from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, PROJECT_ROOT, TIMESTAMP_GROUPS_CSV
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import result_json_path
from plot_confusion_matrices import (
    collect_frame_pairs,
    collect_group_majority_pairs,
    normalize_labels,
)
from aggregate_pipeline_by_timestamp_group import load_frame_group_map, load_prediction_label
from presets import ALL_PRESETS

DEMO_DIR = PROJECT_ROOT / "demo"

PRESET_SPECS: list[tuple[str, str]] = [
    ("clip_full_labels_top5", "CLIP zero-shot"),
    ("clip_linear_probe_crop", "CLIP linear probe"),
    ("clip_lora_crop", "CLIP LoRA"),
    ("siglip_full_labels_top5", "SigLIP zero-shot"),
    ("siglip_linear_probe_crop", "SigLIP linear probe"),
    ("siglip_lora_crop", "SigLIP LoRA"),
    ("efficientnet_b0_crop", "EfficientNet-B0"),
    ("late_fusion_clip_whisper_crop", "Late fusion (CLIP)"),
    ("late_fusion_efficientnet_whisper_crop", "Late fusion (EN)"),
]
PRESET_ORDER = [preset for preset, _ in PRESET_SPECS]
PRESET_LABELS = dict(PRESET_SPECS)

ROW_SPECS: list[tuple[str, str, str]] = [
    ("v1", "frame", "v1 | frame Top-1"),
    ("v2", "frame", "v2 | frame Top-1"),
    ("v5", "frame", "v5 | frame Top-1"),
    ("v15", "frame", "v15 | frame Top-1"),
    ("v15", "group_majority_vote", "v15 | group (majority vote)"),
]


def load_prediction_map(
    meta_df: pd.DataFrame,
    *,
    base_preset: str,
    segment_version: str,
    results_dir: Path,
) -> dict[str, str | None]:
    output_dir = results_dir / segment_output_dir(
        ALL_PRESETS[base_preset]["output_dir"],
        segment_version,
    )
    pred_map: dict[str, str | None] = {}
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        pred_map[frame_id] = load_prediction_label(result_json_path(row, output_dir))
    return pred_map


def compute_metrics(
    meta_df: pd.DataFrame,
    frame_group_df: pd.DataFrame,
    *,
    base_preset: str,
    segment_version: str,
    eval_mode: str,
    results_dir: Path,
) -> dict[str, float | int | None]:
    prediction_map = load_prediction_map(
        meta_df,
        base_preset=base_preset,
        segment_version=segment_version,
        results_dir=results_dir,
    )
    include_no_prediction = eval_mode == "group_majority_vote"

    if eval_mode == "frame":
        y_true, y_pred = collect_frame_pairs(meta_df, prediction_map)
        pipeline_ok = None
    else:
        y_true, y_pred, pipeline_ok = collect_group_majority_pairs(frame_group_df, prediction_map)

    y_true = normalize_labels(y_true, include_no_prediction=include_no_prediction)
    y_pred = normalize_labels(y_pred, include_no_prediction=include_no_prediction)
    if not y_true:
        return {
            "exact_match_accuracy_pct": None,
            "pipeline_class_rate_pct": None,
            "n_samples": 0,
        }

    exact_accuracy = accuracy_score(y_true, y_pred)
    pipeline_accuracy = (
        sum(pipeline_ok) / len(pipeline_ok) if pipeline_ok is not None else exact_accuracy
    )
    return {
        "exact_match_accuracy_pct": round(exact_accuracy * 100, 2),
        "pipeline_class_rate_pct": round(pipeline_accuracy * 100, 2),
        "n_samples": len(y_true),
    }


def build_grid_tables(
    meta_df: pd.DataFrame,
    frame_group_df: pd.DataFrame,
    *,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict] = []

    for segment_version, eval_mode, eval_label in ROW_SPECS:
        for base_preset in PRESET_ORDER:
            metrics = compute_metrics(
                meta_df,
                frame_group_df,
                base_preset=base_preset,
                segment_version=segment_version,
                eval_mode=eval_mode,
                results_dir=results_dir,
            )
            long_rows.append(
                {
                    "eval_label": eval_label,
                    "segment_version": segment_version,
                    "eval_mode": eval_mode,
                    "base_preset": base_preset,
                    "preset": PRESET_LABELS[base_preset],
                    **metrics,
                }
            )

    long_df = pd.DataFrame(long_rows)
    eval_order = [label for _, _, label in ROW_SPECS]
    preset_order = [PRESET_LABELS[p] for p in PRESET_ORDER]

    wide_exact = (
        long_df.pivot(index="preset", columns="eval_label", values="exact_match_accuracy_pct")
        .reindex(preset_order)
        .reindex(columns=eval_order)
    )
    wide_exact.index.name = "preset"
    wide_exact.columns.name = None

    wide_pipeline = (
        long_df.pivot(index="preset", columns="eval_label", values="pipeline_class_rate_pct")
        .reindex(preset_order)
        .reindex(columns=eval_order)
    )
    wide_pipeline.index.name = "preset"
    wide_pipeline.columns.name = None

    return long_df, wide_exact, wide_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build preset × version exact-match grid CSV (confusion matrix metrics)."
    )
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--groups-csv", type=Path, default=TIMESTAMP_GROUPS_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR)
    args = parser.parse_args()

    meta_df = pd.read_csv(args.meta_csv)
    frame_group_df = load_frame_group_map(args.meta_csv, args.groups_csv)

    long_df, wide_exact, wide_pipeline = build_grid_tables(
        meta_df,
        frame_group_df,
        results_dir=args.results_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / "confusion_matrix_preset_version_grid_long.csv"
    exact_path = args.output_dir / "confusion_matrix_preset_version_grid.csv"
    pipeline_path = args.output_dir / "confusion_matrix_preset_version_grid_pipeline.csv"

    long_df.to_csv(long_path, index=False)
    wide_exact.to_csv(exact_path)
    wide_pipeline.to_csv(pipeline_path)

    print(wide_exact.to_string())
    print(f"\nSaved wide exact-match grid: {exact_path}")
    print(f"Saved wide pipeline grid: {pipeline_path}")
    print(f"Saved long format: {long_path}")


if __name__ == "__main__":
    main()
