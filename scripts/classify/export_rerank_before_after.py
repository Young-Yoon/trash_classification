#!/usr/bin/env python3
"""Export top rerank before/after qualitative examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
SEGMENT_DIR_PKG = PACKAGE_DIR.parent / "segment"
for path in (SHARED_DIR, PACKAGE_DIR, SEGMENT_DIR_PKG):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import META2_CSV, RESULTS_DIR, SEGMENT_DIR
from dataset_paths import frame_result_id, resolve_meta_asset_path
from paths import clip_id_from_path
from plot_best_algorithm_qualitative import pick_best_algorithm
from rerank_visualization import render_rerank_before_after
from segment_masks import SEGMENT_BBOX_PADDING, SEGMENT_CROP_MODES, segment_result_path
from segment_rerank import RerankClassificationSample, select_rerank_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rerank before/after qualitative examples.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("demo/mask_class_qualitative/manifest.json"),
        help="Use precomputed before/after metadata when available.",
    )
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--summary-csv", type=Path, default=RESULTS_DIR / "pipeline_end_to_end_summary.csv")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--base-preset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("demo/rerank_before_after"))
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)
    if args.version and args.base_preset:
        version, base_preset = args.version, args.base_preset
    else:
        version, base_preset = pick_best_algorithm(summary_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_df = pd.read_csv(args.meta_csv)
    meta_by_frame = {}
    meta_by_index = {}
    for row_index, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        meta_by_frame[frame_id] = row
        meta_by_index[int(row_index)] = row

    cache_key = "v7" if version in {"v13", "v14", "v15", "v16"} else version
    cache_root = SEGMENT_DIR / f"rerank_candidates_{cache_key}"

    def load_sample_for_frame(frame_id: str) -> RerankClassificationSample | None:
        meta_row = meta_by_frame.get(frame_id)
        if meta_row is None:
            return None
        frame_path = resolve_meta_asset_path(meta_row.get("copied_frame_path"))
        if frame_path is None:
            return None
        candidate_paths = sorted(cache_root.glob(f"{frame_id}_cand*.png"))
        if not candidate_paths:
            return None
        row_index = meta_row.name
        v1_path = segment_result_path(int(row_index), version)  # type: ignore[arg-type]
        scores = [0.0] * len(candidate_paths)
        hand_ious = [0.0] * len(candidate_paths)
        if v1_path.is_file():
            import pickle

            from segment_masks import segment_v2_result_path

            with v1_path.open("rb") as handle:
                v1_result = pickle.load(handle)
            v2_result = None
            if version in {"v9", "v10", "v12"}:
                v2_path = segment_v2_result_path(int(row_index))
                if v2_path.is_file():
                    with v2_path.open("rb") as handle:
                        v2_result = pickle.load(handle)
            _, scores, hand_ious = select_rerank_masks(
                version,  # type: ignore[arg-type]
                v1_result,
                v2_result,
                k=len(candidate_paths),
            )
        return RerankClassificationSample(
            frame_path=frame_path,
            mask_path=candidate_paths[0],
            category_name=str(meta_row["category_name"]),
            gx_id=str(meta_row["gx_id"]),
            clip_id=clip_id_from_path(meta_row.get("timestamp_clip_path")),
            crop_mode=SEGMENT_CROP_MODES[version],  # type: ignore[index]
            bbox_padding_ratio=SEGMENT_BBOX_PADDING[version],  # type: ignore[index]
            candidate_mask_paths=candidate_paths,
            candidate_scores=scores[: len(candidate_paths)],
            candidate_hand_ious=hand_ious[: len(candidate_paths)],
        )

    candidates: list[dict] = []
    if args.source_manifest.is_file():
        with args.source_manifest.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        for item in source.get("examples", []):
            if not item.get("rerank_changed"):
                continue
            delta = float(item.get("after_bbox_iou", 0)) - float(item.get("before_bbox_iou", 0))
            candidates.append({**item, "delta": delta})

    candidates.sort(key=lambda item: item["delta"], reverse=True)
    seen: set[str] = set()
    picked: list[dict] = []
    for item in candidates:
        frame_id = item["frame_id"]
        if frame_id in seen:
            continue
        seen.add(frame_id)
        picked.append(item)
        if len(picked) >= args.n_examples:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.png"):
        stale.unlink()

    manifest_rows: list[dict] = []
    for idx, item in enumerate(picked, start=1):
        frame_id = item["frame_id"]
        sample = load_sample_for_frame(frame_id)
        meta_row = meta_by_frame.get(frame_id)
        if sample is None or meta_row is None:
            continue
        gt_mask_path = resolve_meta_asset_path(meta_row.get("copied_mask_path"))
        before_idx = int(item.get("before_idx", 0))
        after_idx = int(item.get("after_idx", 0))
        output_path = args.output_dir / f"{idx:02d}_{frame_id}.png"
        render_rerank_before_after(
            sample,
            version=version,  # type: ignore[arg-type]
            base_preset=base_preset,
            gt_mask_path=Path(gt_mask_path) if gt_mask_path else Path(),
            gt_label=str(meta_row["category_name"]),
            before_idx=before_idx,
            after_idx=after_idx,
            output_path=output_path,
            device=device,
        )
        manifest_rows.append(
            {
                "frame_id": frame_id,
                "before_idx": before_idx,
                "after_idx": after_idx,
                "before_bbox_iou": item.get("before_bbox_iou"),
                "after_bbox_iou": item.get("after_bbox_iou"),
                "delta_bbox_iou": item.get("delta"),
                "output_path": str(output_path),
            }
        )

    manifest = {
        "segment_version": version,
        "base_preset": base_preset,
        "output_dir": str(args.output_dir),
        "examples": manifest_rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved {len(manifest_rows)} rerank before/after examples to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
