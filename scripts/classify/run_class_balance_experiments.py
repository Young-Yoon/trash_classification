#!/usr/bin/env python3
"""Train A1-A3 class-balancing variants, fusion heads, and evaluate on v15 group metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from aggregate_pipeline_by_timestamp_group import (
    build_per_group_records,
    load_frame_group_map,
    load_prediction_label,
    parse_gt_categories,
)
from build_pipeline_end_to_end import e2e_segment_output_dir, load_class_top1_map
from config import CLASSIFY_CHECKPOINT_DIR, META2_CSV, PROJECT_ROOT, RESULTS_DIR, TIMESTAMP_GROUPS_CSV
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import result_json_path
from presets import TRAINED_PRESETS
from segment_masks import SEGMENT_CROP_MODES
from segment_rerank import load_rerank_classification_samples

EXPERIMENTS = {
    "A1": {
        "balance_mode": "reweight",
        "efficientnet_dir": "efficientnet_b0_crop_balance_a1",
        "fusion_dir": "late_fusion_efficientnet_whisper_crop_balance_a1",
    },
    "A2": {
        "balance_mode": "oversample",
        "efficientnet_dir": "efficientnet_b0_crop_balance_a2",
        "fusion_dir": "late_fusion_efficientnet_whisper_crop_balance_a2",
    },
    "A3": {
        "balance_mode": "combined",
        "efficientnet_dir": "efficientnet_b0_crop_balance_a3",
        "fusion_dir": "late_fusion_efficientnet_whisper_crop_balance_a3",
    },
}

TAIL_CLASSES = [
    "Garbage Bags",
    "Other Plastic Film",
    "Plastic",
    "Other Rigid/Bulky Plastics",
    "Other Unnumbered Containers & Fragments",
    "Black Plastic Containers",
]

OTHER1_SOURCES = [
    "Other Unnumbered Containers & Fragments",
    "Containers #2 Colored",
    "Other Plastics",
]


def run_cmd(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def train_variants(
    *,
    epochs: int,
    batch_size: int,
    experiments: list[str],
    skip_train: bool,
    skip_fusion: bool,
) -> None:
    print(f"Training on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    for experiment in experiments:
        spec = EXPERIMENTS[experiment]
        eff_dir = CLASSIFY_CHECKPOINT_DIR / spec["efficientnet_dir"]
        fusion_dir = CLASSIFY_CHECKPOINT_DIR / spec["fusion_dir"]

        if not skip_train:
            run_cmd(
                [
                    sys.executable,
                    str(PACKAGE_DIR / "train.py"),
                    "--method",
                    "efficientnet",
                    "--epochs",
                    str(epochs),
                    "--batch-size",
                    str(batch_size),
                    "--balance-mode",
                    spec["balance_mode"],
                    "--output-dir",
                    str(eff_dir),
                ]
            )

        if not skip_fusion:
            run_cmd(
                [
                    sys.executable,
                    str(PACKAGE_DIR / "train.py"),
                    "--method",
                    "fusion",
                    "--fusion-image",
                    "efficientnet",
                    "--image-checkpoint",
                    str(eff_dir),
                    "--output-dir",
                    str(fusion_dir),
                ]
            )


def infer_v15_with_checkpoint(base_preset: str, checkpoint_dir: Path, *, batch_size: int = 32) -> None:
    from infer_trained import run_preset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_rerank_classification_samples(
        "v15",
        meta_csv=META2_CSV,
        crop_mode=SEGMENT_CROP_MODES["v15"],
    )
    preset_payload = {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "use_mask": True,
        "checkpoint_name": checkpoint_dir.name,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": f"{base_preset}_seg_v15",
        "segment_version": "v15",
        "mask_source": "segment_v15",
        "crop_mode": SEGMENT_CROP_MODES["v15"],
        "rerank": True,
        "rerank_strategy": "text_aware",
    }
    run_preset(
        f"{base_preset}_seg_v15",
        preset_payload,
        samples=samples,
        results_dir=PROJECT_ROOT / "classification_results",
        batch_size=batch_size,
        top_k=5,
        resume=False,
        device=device,
    )


def load_prediction_label_map(base_preset: str) -> dict[str, str | None]:
    meta_df = pd.read_csv(META2_CSV)
    output_dir = PROJECT_ROOT / "classification_results" / e2e_segment_output_dir(base_preset, "v15")
    prediction_map: dict[str, str | None] = {}
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        prediction_map[frame_id] = load_prediction_label(result_json_path(row, output_dir))
    return prediction_map


def load_pipeline_group_eval(
    base_preset: str,
    *,
    policy: str = "any_frame_any_bbox",
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    per_frame_csv = RESULTS_DIR / "pipeline_end_to_end_per_frame.csv"
    if not per_frame_csv.is_file():
        raise FileNotFoundError(f"Missing per-frame pipeline CSV: {per_frame_csv}")

    meta_df = pd.read_csv(META2_CSV)
    frame_group_df = load_frame_group_map(META2_CSV, TIMESTAMP_GROUPS_CSV)
    valid_frame_ids = set(frame_group_df["frame_id"])
    mask_template = (
        pd.read_csv(per_frame_csv)
        .query("segment_version == 'v15' and base_preset == 'late_fusion_efficientnet_whisper_crop'")
        .loc[lambda df: df["frame_id"].isin(valid_frame_ids)]
        .drop_duplicates(subset=["frame_id"], keep="first")
        .copy()
    )
    class_map = load_class_top1_map(base_preset, "v15", meta_df, results_dir=PROJECT_ROOT / "classification_results")
    per_frame_eval = mask_template.copy()
    per_frame_eval["base_preset"] = base_preset
    per_frame_eval["class_top1_ok"] = per_frame_eval["frame_id"].map(lambda frame_id: class_map.get(frame_id, False))

    label_map = load_prediction_label_map(base_preset)
    prediction_map = {
        frame_id: (label or None)
        for frame_id, label in label_map.items()
    }
    rerank_pred_map = {
        frame_id: label
        for frame_id, label in prediction_map.items()
        if label
    }
    per_group_df = build_per_group_records(
        per_frame_eval,
        frame_group_df,
        {("v15", base_preset): rerank_pred_map},
        policies=[policy],
    )
    per_group_df["gt_categories"] = per_group_df["category_names"].map(parse_gt_categories)
    per_group_df["pred_label"] = per_group_df["group_key"].map(
        lambda group_key: next(
            (
                prediction_map.get(frame_id) or ""
                for frame_id in frame_group_df.loc[frame_group_df["group_key"] == group_key, "frame_id"]
                if prediction_map.get(frame_id)
            ),
            "",
        )
    )
    return per_group_df, prediction_map


def load_group_predictions(base_preset: str, *, policy: str = "any_frame_any_bbox") -> tuple[pd.DataFrame, dict[str, str]]:
    per_group_df, prediction_map = load_pipeline_group_eval(base_preset, policy=policy)
    group_df = per_group_df.rename(columns={"pipeline_ok": "pipeline_ok"}).copy()
    string_prediction_map = {
        frame_id: (label or "")
        for frame_id, label in prediction_map.items()
    }
    return group_df, string_prediction_map


def compute_group_final_probability(group_df: pd.DataFrame) -> float:
    """Match Table~I: mask_success_rate * class_given_mask_rate / 100 at group level."""
    mask_rate = float(group_df["mask_ok"].mean() * 100)
    masked = group_df[group_df["mask_ok"]]
    class_given_mask_rate = float(masked["class_ok"].mean() * 100) if len(masked) else 0.0
    return mask_rate * class_given_mask_rate / 100


def compute_metrics(group_df: pd.DataFrame, prediction_map: dict[str, str]) -> dict:
    group_top1 = compute_group_final_probability(group_df)
    y_true, y_pred = [], []
    for _, row in group_df.iterrows():
        gt = sorted(row["gt_categories"])
        if not gt:
            continue
        y_true.append(gt[0])
        y_pred.append(row["pred_label"] or "No prediction")
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100)

    tail_recall: dict[str, float] = {}
    for cls in TAIL_CLASSES:
        cls_rows = group_df[group_df["gt_categories"].map(lambda cats: cls in cats)]
        tail_recall[cls] = float(cls_rows["class_ok"].mean() * 100) if len(cls_rows) else float("nan")

    meta_df = pd.read_csv(META2_CSV)
    fp_into_other1 = 0
    source_total = 0
    for _, row in meta_df.iterrows():
        gt = str(row["category_name"])
        if gt not in OTHER1_SOURCES:
            continue
        source_total += 1
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        if prediction_map.get(frame_id, "") == "Other #1 Bottles":
            fp_into_other1 += 1
    other1_fp_rate = (fp_into_other1 / source_total * 100) if source_total else 0.0

    return {
        "group_top1": round(group_top1, 2),
        "macro_f1": round(macro_f1, 2),
        "other1_fp_rate": round(other1_fp_rate, 2),
        "film_recall": round(tail_recall.get("Other Plastic Film", float("nan")), 2),
        "bags_recall": round(tail_recall.get("Garbage Bags", float("nan")), 2),
        "tail_recall": {key: round(value, 2) for key, value in tail_recall.items()},
    }


def evaluate_experiments(experiments: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    try:
        baseline_df, baseline_preds = load_group_predictions("late_fusion_efficientnet_whisper_crop")
        a0_metrics = compute_metrics(baseline_df, baseline_preds)
        rows.append({"variant": "A0", **a0_metrics})
        print(f"A0: {json.dumps(a0_metrics, indent=2)}")
    except FileNotFoundError as exc:
        print(f"Baseline missing: {exc}")

    for experiment in experiments:
        preset = EXPERIMENTS[experiment]["fusion_dir"]
        group_df, prediction_map = load_group_predictions(preset)
        metrics = compute_metrics(group_df, prediction_map)
        rows.append({"variant": experiment, **metrics})
        print(f"{experiment}: {json.dumps(metrics, indent=2)}")

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "class_balance_experiments_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics: {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run class-balancing experiments A1-A3.")
    parser.add_argument("--experiments", nargs="+", choices=[*EXPERIMENTS, "all"], default=["all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    experiments = list(EXPERIMENTS) if "all" in args.experiments else args.experiments

    train_variants(
        epochs=args.epochs,
        batch_size=args.batch_size,
        experiments=experiments,
        skip_train=args.skip_train,
        skip_fusion=args.skip_fusion,
    )

    if not args.skip_infer:
        for experiment in experiments:
            fusion_dir = CLASSIFY_CHECKPOINT_DIR / EXPERIMENTS[experiment]["fusion_dir"]
            infer_v15_with_checkpoint(EXPERIMENTS[experiment]["fusion_dir"], fusion_dir, batch_size=args.batch_size)

    if not args.skip_eval:
        evaluate_experiments(experiments)


if __name__ == "__main__":
    main()
