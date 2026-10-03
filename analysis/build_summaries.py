#!/usr/bin/env python3
"""Rebuild compact analysis tables from meta CSVs + shipped summary CSVs.

Does not require raw masks, candidate crops, or per-frame JSON.
Outputs land under summaries/ (overwrite curated tables when sources exist).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
CONFIGS = REPO_ROOT / "configs"
sys.path.insert(0, str(SCRIPTS / "shared"))
sys.path.insert(0, str(CONFIGS))

from config import (  # noqa: E402
    COMMON_PHRASES_CSV,
    META2_CSV,
    SUMMARIES_DIR,
    TAXONOMY_JSON,
    TIMESTAMP_GROUPS_CSV,
)
from segment_versions import (  # noqa: E402
    BBOX_PADDING,
    CROP_MODES,
    PAPER_GROUP_ANY_FRAME,
    PHASES,
    RERANK_STRATEGIES,
    REPORTED_FINAL_VERSION,
    SEGMENT_VERSION_DESCRIPTIONS,
)


def dataset_stats(out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    if META2_CSV.is_file():
        meta = pd.read_csv(META2_CSV)
        n_frames = len(meta)
        n_cat = int(meta["category_name"].nunique()) if "category_name" in meta else None
        n_gx = int(meta["gx_id"].nunique()) if "gx_id" in meta else None
        rows.append(
            {
                "metric": "n_frames",
                "value": n_frames,
                "source": str(META2_CSV.name),
            }
        )
        rows.append({"metric": "n_categories", "value": n_cat, "source": str(META2_CSV.name)})
        rows.append({"metric": "n_gx_sequences", "value": n_gx, "source": str(META2_CSV.name)})

        counts = (
            meta["category_name"]
            .value_counts()
            .rename_axis("category_name")
            .reset_index(name="n_frames")
        )
        counts.to_csv(out_dir / "dataset_category_counts.csv", index=False)

        phrase_counts = (
            meta["phrase"]
            .fillna("")
            .astype(str)
            .str.strip()
            .value_counts()
            .head(30)
            .rename_axis("phrase")
            .reset_index(name="n_frames")
        )
        phrase_counts.to_csv(out_dir / "dataset_top_phrases.csv", index=False)
    else:
        rows.append(
            {
                "metric": "n_frames",
                "value": 3538,
                "source": "hardcoded_fallback",
            }
        )
        rows.append({"metric": "n_categories", "value": 15, "source": "hardcoded_fallback"})

    if TIMESTAMP_GROUPS_CSV.is_file():
        groups = pd.read_csv(TIMESTAMP_GROUPS_CSV)
        rows.append(
            {
                "metric": "n_clips",
                "value": len(groups),
                "source": str(TIMESTAMP_GROUPS_CSV.name),
            }
        )
    else:
        rows.append({"metric": "n_clips", "value": 463, "source": "hardcoded_fallback"})

    if COMMON_PHRASES_CSV.is_file():
        phrases = pd.read_csv(COMMON_PHRASES_CSV)
        rows.append(
            {
                "metric": "n_common_phrases",
                "value": len(phrases),
                "source": str(COMMON_PHRASES_CSV.name),
            }
        )
    else:
        rows.append(
            {
                "metric": "n_common_phrases",
                "value": 104,
                "source": "hardcoded_fallback",
            }
        )

    if TAXONOMY_JSON.is_file():
        taxonomy = json.loads(TAXONOMY_JSON.read_text())
        rows.append(
            {
                "metric": "taxonomy_categories",
                "value": len(taxonomy.get("categories", [])),
                "source": str(TAXONOMY_JSON.name),
            }
        )

    for key, value in PAPER_GROUP_ANY_FRAME.items():
        rows.append({"metric": key, "value": value, "source": "paper_table"})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dataset_stats.csv", index=False)
    return df


def version_catalog(out_dir: Path) -> pd.DataFrame:
    rows = []
    phase_of = {
        version: phase for phase, versions in PHASES.items() for version in versions
    }
    for version, description in SEGMENT_VERSION_DESCRIPTIONS.items():
        rows.append(
            {
                "segment_version": version,
                "phase": phase_of.get(version, ""),
                "crop_mode": CROP_MODES.get(version, ""),
                "bbox_padding": BBOX_PADDING.get(version, 0.0),
                "rerank_strategy": RERANK_STRATEGIES.get(version, ""),
                "description": description,
                "is_reported_final": version == REPORTED_FINAL_VERSION,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segment_version_catalog.csv", index=False)
    return df


def merge_version_metrics(out_dir: Path) -> pd.DataFrame:
    """Join shipped Fusion-EN frame Top-1 with version catalog."""
    catalog = version_catalog(out_dir)
    class_path = out_dir / "classification_segment_summary.csv"
    if not class_path.is_file():
        return catalog

    cls = pd.read_csv(class_path)
    # Prefer Fusion-EN rows when present.
    fusion = cls[cls["base_preset"].astype(str).str.contains("late_fusion_efficientnet", na=False)].copy()
    if fusion.empty:
        fusion = cls.copy()
    fusion = fusion.rename(
        columns={
            "top_1_success_rate": "frame_top1_fusion_en",
            "top_3_success_rate": "frame_top3_fusion_en",
            "top_5_success_rate": "frame_top5_fusion_en",
        }
    )
    keep = [
        "segment_version",
        "frame_top1_fusion_en",
        "frame_top3_fusion_en",
        "frame_top5_fusion_en",
    ]
    merged = catalog.merge(fusion[keep], on="segment_version", how="left")

    group_path = out_dir / "pipeline_end_to_end_group_summary.csv"
    if group_path.is_file():
        group = pd.read_csv(group_path)
        # Prefer speech-aligned any-frame rows when duplicated policies exist:
        # keep first policy block for Fusion-EN (any_frame_any_bbox appears first in file).
        fusion_g = group[
            group["base_preset"].astype(str).eq("late_fusion_efficientnet_whisper_crop")
            & group["aggregation_policy"].astype(str).eq("any_frame_any_bbox")
        ].copy()
        if fusion_g.empty:
            fusion_g = group[
                group["base_preset"].astype(str).eq("late_fusion_efficientnet_whisper_crop")
            ].copy()
        fusion_g = fusion_g.drop_duplicates("segment_version", keep="first")
        fusion_g = fusion_g.rename(
            columns={"final_probability": "group_any_frame_final_prob_fusion_en"}
        )
        merged = merged.merge(
            fusion_g[["segment_version", "group_any_frame_final_prob_fusion_en"]],
            on="segment_version",
            how="left",
        )

        en = group[
            group["base_preset"].astype(str).eq("efficientnet_b0_crop")
            & group["aggregation_policy"].astype(str).eq("any_frame_any_bbox")
        ].drop_duplicates("segment_version", keep="first")
        if not en.empty:
            en = en.rename(columns={"final_probability": "group_any_frame_final_prob_efficientnet"})
            merged = merged.merge(
                en[["segment_version", "group_any_frame_final_prob_efficientnet"]],
                on="segment_version",
                how="left",
            )

    iou_path = out_dir / "segment_iou_summary.csv"
    if iou_path.is_file():
        iou = pd.read_csv(iou_path)
        iou = iou.rename(
            columns={
                "version": "segment_version",
                "mean_max_iou_top1": "csv_mean_max_iou_top1",
                "mean_iou_all_detections": "csv_mean_iou_all_detections",
            }
        )
        cols = [
            c
            for c in (
                "segment_version",
                "csv_mean_max_iou_top1",
                "csv_mean_iou_all_detections",
                "pct_top1_iou_ge_0.5",
            )
            if c in iou.columns
        ]
        merged = merged.merge(iou[cols], on="segment_version", how="left")

    merged.to_csv(out_dir / "version_metrics_joined.csv", index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SUMMARIES_DIR,
        help="Directory for summary CSVs (default: summaries/)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = dataset_stats(out_dir)
    joined = merge_version_metrics(out_dir)
    print(f"Wrote dataset_stats.csv ({len(stats)} rows) -> {out_dir}")
    print(f"Wrote segment_version_catalog.csv / version_metrics_joined.csv ({len(joined)} versions)")
    if "frame_top1_fusion_en" in joined.columns:
        print(joined[["segment_version", "frame_top1_fusion_en", "phase"]].to_string(index=False))


if __name__ == "__main__":
    main()
