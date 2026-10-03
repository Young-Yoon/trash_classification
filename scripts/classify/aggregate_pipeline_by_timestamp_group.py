#!/usr/bin/env python3
"""Aggregate per-frame pipeline metrics to timestamp_group level."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_pipeline_end_to_end import IOU_ZERO_EPS, METHOD_LABELS, VERSION_PHASE, e2e_segment_output_dir
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import predicted_label_name
from segment_masks import ALL_SEGMENT_VERSIONS, SEGMENT_VERSION_DESCRIPTIONS, SegmentVersion

AGGREGATION_POLICIES = {
    "majority_vote_any_bbox": {
        "label": "Majority vote (class) + any frame bbox (mask)",
        "class_mode": "majority_vote",
        "mask_mode": "any_frame",
    },
    "any_frame_any_bbox": {
        "label": "Any frame (class) + any frame bbox (mask)",
        "class_mode": "any_frame",
        "mask_mode": "any_frame",
    },
}

PHASE_COLORS = {
    "Phase1_Backbone": "#4C72B0",
    "Phase2_MaskSelect": "#55A868",
    "Phase3_RerankPool": "#C44E52",
    "Phase4_RerankStrategy": "#8172B2",
}


def build_group_key(gx_id: str, begin_time: float, end_time: float) -> str:
    return f"{gx_id}|{begin_time}|{end_time}"


def parse_gt_categories(category_names: str | float) -> set[str]:
    if pd.isna(category_names) or not str(category_names).strip():
        return set()
    return {part.strip() for part in str(category_names).split("|") if part.strip()}


def load_frame_group_map(meta_csv: Path, groups_csv: Path) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_csv)
    groups_df = pd.read_csv(groups_csv)

    valid_meta = meta_df.dropna(subset=["begin_time", "end_time"]).copy()
    valid_meta["group_key"] = valid_meta.apply(
        lambda row: build_group_key(str(row["gx_id"]), float(row["begin_time"]), float(row["end_time"])),
        axis=1,
    )

    groups_df = groups_df.copy()
    groups_df["group_key"] = groups_df.apply(
        lambda row: build_group_key(str(row["gx_name"]), float(row["begin_time"]), float(row["end_time"])),
        axis=1,
    )
    group_cols = [
        "group_key",
        "gx_name",
        "begin_time",
        "end_time",
        "meta2_phrase",
        "category_names",
        "clip_path",
        "number_of_meta2_rows",
        "number_of_unique_frames",
    ]
    groups_lookup = groups_df[group_cols].drop_duplicates("group_key")

    merged = valid_meta.merge(groups_lookup, on="group_key", how="inner", suffixes=("", "_group"))
    merged["gt_categories"] = merged["category_names"].map(parse_gt_categories)

    frame_rows: list[dict] = []
    for _, row in merged.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "group_key": row["group_key"],
                "gx_name": row["gx_name"],
                "begin_time": row["begin_time"],
                "end_time": row["end_time"],
                "meta2_phrase": row["meta2_phrase"],
                "category_names": row["category_names"],
                "clip_path": row.get("clip_path", ""),
                "frame_category_name": str(row["category_name"]),
            }
        )
    frame_df = pd.DataFrame(frame_rows)
    return frame_df.drop_duplicates(subset=["frame_id"], keep="first")


def load_prediction_label(json_path: Path) -> str | None:
    if not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, list) and payload:
        return predicted_label_name(payload[0]["label"])
    if isinstance(payload, dict) and "label" in payload:
        return predicted_label_name(payload["label"])
    return None


def load_prediction_maps(
    meta_df: pd.DataFrame,
    *,
    results_dir: Path,
    versions: list[SegmentVersion],
    presets: list[str],
) -> dict[tuple[str, str], dict[str, str | None]]:
    prediction_maps: dict[tuple[str, str], dict[str, str | None]] = {}
    for version in versions:
        for base_preset in presets:
            output_dir = results_dir / e2e_segment_output_dir(base_preset, version)
            pred_map: dict[str, str | None] = {}
            for _, row in meta_df.iterrows():
                frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
                if frame_path is None:
                    continue
                frame_id = frame_result_id(str(row["gx_id"]), frame_path)
                pred_map[frame_id] = load_prediction_label(result_json_path(row, output_dir))
            prediction_maps[(version, base_preset)] = pred_map
    return prediction_maps


def gt_matches_prediction(gt_categories: set[str], frame_categories: set[str], prediction: str | None) -> bool:
    if prediction is None:
        return False
    if gt_categories:
        return prediction in gt_categories
    return prediction in frame_categories


def majority_vote_label(predictions: list[str | None]) -> str | None:
    valid = [pred for pred in predictions if pred]
    if not valid:
        return None
    counts = Counter(valid)
    top_count = counts.most_common(1)[0][1]
    tied = [label for label, count in counts.items() if count == top_count]
    return sorted(tied)[0]


def aggregate_group_booleans(
    group_frames: pd.DataFrame,
    *,
    class_mode: str,
    prediction_map: dict[str, str | None],
    gt_categories: set[str],
    frame_categories: set[str],
) -> dict[str, bool]:
    mask_ok = bool(group_frames["mask_ok"].any())

    if class_mode == "any_frame":
        class_ok = bool(group_frames["class_top1_ok"].any())
    elif class_mode == "majority_vote":
        predictions = [prediction_map.get(frame_id) for frame_id in group_frames["frame_id"]]
        majority_label = majority_vote_label(predictions)
        class_ok = gt_matches_prediction(gt_categories, frame_categories, majority_label)
    else:
        raise ValueError(f"Unknown class_mode: {class_mode}")

    same_frame_ok = bool((group_frames["mask_ok"] & group_frames["class_top1_ok"]).any())
    iou_zero_class_ok = bool(
        ((group_frames["selected_bbox_iou"] <= IOU_ZERO_EPS) & group_frames["class_top1_ok"]).any()
    )
    # Group class success that never co-occurs with mask on the same frame.
    maskless_class_ok = bool(class_ok and not same_frame_ok)
    if class_mode == "majority_vote":
        pipeline_ok = bool(class_ok and same_frame_ok)
    else:
        pipeline_ok = same_frame_ok

    return {
        "mask_ok": mask_ok,
        "class_ok": class_ok,
        "same_frame_ok": same_frame_ok,
        "iou_zero_class_ok": iou_zero_class_ok,
        "maskless_class_ok": maskless_class_ok,
        "pipeline_ok": pipeline_ok,
    }


def build_per_group_records(
    per_frame_df: pd.DataFrame,
    frame_group_df: pd.DataFrame,
    prediction_maps: dict[tuple[str, str], dict[str, str | None]],
    *,
    policies: list[str],
) -> pd.DataFrame:
    enriched = per_frame_df.merge(
        frame_group_df[
            [
                "frame_id",
                "group_key",
                "gx_name",
                "begin_time",
                "end_time",
                "meta2_phrase",
                "category_names",
                "clip_path",
                "frame_category_name",
            ]
        ],
        on="frame_id",
        how="inner",
    )

    rows: list[dict] = []
    group_meta = (
        frame_group_df.groupby("group_key", as_index=False)
        .agg(
            gx_name=("gx_name", "first"),
            begin_time=("begin_time", "first"),
            end_time=("end_time", "first"),
            meta2_phrase=("meta2_phrase", "first"),
            category_names=("category_names", "first"),
            clip_path=("clip_path", "first"),
            n_frames=("frame_id", "count"),
            frame_categories=("frame_category_name", lambda s: set(s.astype(str))),
        )
        .set_index("group_key")
    )

    combo_keys = enriched.groupby(["segment_version", "base_preset"], sort=False).groups.keys()
    for version, base_preset in combo_keys:
        subset = enriched[
            (enriched["segment_version"] == version) & (enriched["base_preset"] == base_preset)
        ]
        prediction_map = prediction_maps[(version, base_preset)]
        phase, phase_order = VERSION_PHASE[version]
        description = SEGMENT_VERSION_DESCRIPTIONS.get(version, "")

        for group_key, group_frames in subset.groupby("group_key", sort=False):
            group_frames = group_frames.drop_duplicates(subset=["frame_id"], keep="first")
            meta = group_meta.loc[group_key]
            gt_categories = parse_gt_categories(meta["category_names"])
            frame_categories = meta["frame_categories"]

            for policy_name in policies:
                policy = AGGREGATION_POLICIES[policy_name]
                flags = aggregate_group_booleans(
                    group_frames,
                    class_mode=policy["class_mode"],
                    prediction_map=prediction_map,
                    gt_categories=gt_categories,
                    frame_categories=frame_categories,
                )
                rows.append(
                    {
                        "group_key": group_key,
                        "gx_name": meta["gx_name"],
                        "begin_time": meta["begin_time"],
                        "end_time": meta["end_time"],
                        "meta2_phrase": meta["meta2_phrase"],
                        "category_names": meta["category_names"],
                        "clip_path": meta["clip_path"],
                        "n_frames_meta2": int(meta["n_frames"]),
                        "n_frames_evaluated": len(group_frames),
                        "segment_version": version,
                        "phase": phase,
                        "phase_order": phase_order,
                        "base_preset": base_preset,
                        "method": METHOD_LABELS.get(base_preset, base_preset),
                        "description": description,
                        "aggregation_policy": policy_name,
                        "aggregation_label": policy["label"],
                        **flags,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_group_summary(per_group_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = [
        "segment_version",
        "base_preset",
        "phase",
        "phase_order",
        "method",
        "description",
        "aggregation_policy",
        "aggregation_label",
    ]
    for key, group in per_group_df.groupby(group_cols, sort=False):
        (
            version,
            base_preset,
            phase,
            phase_order,
            method,
            description,
            aggregation_policy,
            aggregation_label,
        ) = key
        n_groups = len(group)
        mask_rate = group["mask_ok"].mean() * 100
        class_rate = group["class_ok"].mean() * 100
        pipeline_rate = group["pipeline_ok"].mean() * 100
        iou_zero_class_ok_rate = group["iou_zero_class_ok"].mean() * 100
        maskless_class_rate = group["maskless_class_ok"].mean() * 100
        masked = group[group["mask_ok"]]
        class_given_mask_rate = float(masked["class_ok"].mean() * 100) if len(masked) else 0.0
        final_prob = mask_rate * class_given_mask_rate / 100

        rows.append(
            {
                "segment_version": version,
                "phase": phase,
                "phase_order": phase_order,
                "base_preset": base_preset,
                "method": method,
                "description": description,
                "aggregation_policy": aggregation_policy,
                "aggregation_label": aggregation_label,
                "n_groups": n_groups,
                "n_frames_meta2": int(group["n_frames_meta2"].sum()),
                "n_frames_evaluated": int(group["n_frames_evaluated"].sum()),
                "mask_success_rate": round(mask_rate, 2),
                "class_success_rate": round(class_rate, 2),
                "class_given_mask_rate": round(class_given_mask_rate, 2),
                "pipeline_success_rate": round(pipeline_rate, 2),
                "iou_zero_class_ok_rate": round(iou_zero_class_ok_rate, 2),
                "maskless_class_ok_rate": round(maskless_class_rate, 2),
                "final_probability": round(final_prob, 2),
                "adjusted_final_probability": round(pipeline_rate, 2),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["aggregation_policy", "phase_order", "segment_version", "base_preset"]
    ).reset_index(drop=True)


def build_group_comparison_wide(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (policy, policy_label), policy_df in summary_df.groupby(["aggregation_policy", "aggregation_label"], sort=False):
        for version in ALL_SEGMENT_VERSIONS:
            version_rows = policy_df[policy_df["segment_version"] == version]
            if version_rows.empty:
                continue
            phase, phase_order = VERSION_PHASE[version]
            row: dict = {
                "aggregation_policy": policy,
                "aggregation_label": policy_label,
                "segment_version": version,
                "phase": phase,
                "phase_order": phase_order,
                "description": SEGMENT_VERSION_DESCRIPTIONS.get(version, ""),
                "n_groups": int(version_rows["n_groups"].iloc[0]),
            }
            non_rerank = version not in {"v7", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}
            if non_rerank:
                row["mask_success_rate"] = round(version_rows["mask_success_rate"].iloc[0], 2)
            for base_preset, method in METHOD_LABELS.items():
                match = version_rows[version_rows["base_preset"] == base_preset]
                if match.empty:
                    continue
                m = match.iloc[0]
                prefix = base_preset.replace("_crop", "").replace("_clip_whisper", "")
                if not non_rerank:
                    row[f"{prefix}_mask_rate"] = m["mask_success_rate"]
                row[f"{prefix}_class_rate"] = m["class_success_rate"]
                row[f"{prefix}_class_given_mask"] = m["class_given_mask_rate"]
                row[f"{prefix}_pipeline_rate"] = m["pipeline_success_rate"]
                row[f"{prefix}_iou0_cls_ok"] = m["iou_zero_class_ok_rate"]
                row[f"{prefix}_final"] = m["final_probability"]
                row[f"{prefix}_adjusted"] = m["adjusted_final_probability"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["aggregation_policy", "phase_order", "segment_version"]
    ).reset_index(drop=True)


def plot_group_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    highlight_presets = [
        ("late_fusion_clip_whisper_crop", "Late fusion"),
        ("efficientnet_b0_crop", "EfficientNet-B0"),
    ]
    policies = list(AGGREGATION_POLICIES.keys())
    fig, axes = plt.subplots(len(highlight_presets), len(policies), figsize=(16, 5 * len(highlight_presets)), sharex=True)
    if len(highlight_presets) == 1:
        axes = [axes]
    if len(policies) == 1:
        axes = [[ax] for ax in axes]

    for row_axes, (preset, preset_label) in zip(axes, highlight_presets):
        for ax, policy in zip(row_axes, policies):
            subset = summary_df[
                (summary_df["base_preset"] == preset) & (summary_df["aggregation_policy"] == policy)
            ].copy()
            subset["phase"] = subset["segment_version"].map(lambda v: VERSION_PHASE[v][0])
            subset["phase_order"] = subset["segment_version"].map(lambda v: VERSION_PHASE[v][1])
            subset = subset.sort_values(["phase_order", "segment_version"])
            colors = [PHASE_COLORS.get(phase, "#999999") for phase in subset["phase"]]
            x = range(len(subset))
            width = 0.18
            offsets = [-1.5, -0.5, 0.5, 1.5]
            bar_specs = [
                (offsets[0], subset["class_success_rate"], "#DD8452", "Class only"),
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
            ax.set_title(
                f"{preset_label} — {AGGREGATION_POLICIES[policy]['label']}\n"
                f"463 timestamp groups (NaN-time rows excluded)"
            )
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

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
    parser = argparse.ArgumentParser(description="Aggregate pipeline metrics by timestamp group.")
    parser.add_argument(
        "--per-frame-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_per_frame.csv",
    )
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--groups-csv", type=Path, default=TIMESTAMP_GROUPS_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument(
        "--output-per-group-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_per_group.csv",
    )
    parser.add_argument(
        "--output-group-summary-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_group_summary.csv",
    )
    parser.add_argument(
        "--output-group-wide-csv",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_group_comparison_wide.csv",
    )
    parser.add_argument(
        "--output-group-chart",
        type=Path,
        default=RESULTS_DIR / "pipeline_end_to_end_group_comparison.png",
    )
    parser.add_argument(
        "--policy",
        action="append",
        choices=list(AGGREGATION_POLICIES.keys()),
        help="Limit aggregation policy. Default: all.",
    )
    args = parser.parse_args()

    if not args.per_frame_csv.is_file():
        raise SystemExit(f"Missing per-frame CSV: {args.per_frame_csv}")

    policies = args.policy or list(AGGREGATION_POLICIES.keys())
    per_frame_df = pd.read_csv(args.per_frame_csv)
    frame_group_df = load_frame_group_map(args.meta_csv, args.groups_csv)
    valid_frame_ids = set(frame_group_df["frame_id"])
    per_frame_eval = per_frame_df[per_frame_df["frame_id"].isin(valid_frame_ids)].copy()
    per_frame_eval = per_frame_eval.drop_duplicates(
        subset=["frame_id", "segment_version", "base_preset"],
        keep="first",
    )

    meta_df = pd.read_csv(args.meta_csv)
    meta_df = meta_df.dropna(subset=["begin_time", "end_time"])
    versions = sorted(per_frame_eval["segment_version"].unique().tolist())
    presets = sorted(per_frame_eval["base_preset"].unique().tolist())
    prediction_maps = load_prediction_maps(
        meta_df,
        results_dir=args.results_dir,
        versions=versions,
        presets=presets,
    )

    per_group_df = build_per_group_records(
        per_frame_eval,
        frame_group_df,
        prediction_maps,
        policies=policies,
    )
    summary_df = aggregate_group_summary(per_group_df)
    wide_df = build_group_comparison_wide(summary_df)

    args.output_per_group_csv.parent.mkdir(parents=True, exist_ok=True)
    per_group_df.to_csv(args.output_per_group_csv, index=False)
    summary_df.to_csv(args.output_group_summary_csv, index=False)
    wide_df.to_csv(args.output_group_wide_csv, index=False)
    plot_group_chart(summary_df, args.output_group_chart)

    evaluated_frames = per_frame_eval["frame_id"].nunique()
    print(f"Evaluated frames with valid timestamp groups: {evaluated_frames} / {len(valid_frame_ids)} mapped")
    print(f"Timestamp groups: {frame_group_df['group_key'].nunique()}")
    print(f"Saved per-group metrics: {args.output_per_group_csv} ({len(per_group_df)} rows)")
    print(f"Saved group summary: {args.output_group_summary_csv} ({len(summary_df)} rows)")
    print(f"Saved group wide table: {args.output_group_wide_csv}")
    print(f"Saved group chart: {args.output_group_chart}")

    for policy in policies:
        policy_summary = summary_df[summary_df["aggregation_policy"] == policy]
        best = policy_summary.sort_values(
            ["adjusted_final_probability", "class_success_rate"],
            ascending=[False, False],
        ).iloc[0]
        print(
            f"\n[{policy}] Best final: {best['segment_version']} + {best['method']} "
            f"class={best['class_success_rate']:.2f}% class|mask={best['class_given_mask_rate']:.2f}% "
            f"final={best['final_probability']:.2f}%"
        )


if __name__ == "__main__":
    main()
