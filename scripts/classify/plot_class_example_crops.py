#!/usr/bin/env python3
"""Export one GT bbox crop example per plastic category."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import META2_CSV, RESULTS_DIR, TAXONOMY_JSON
from dataset_paths import frame_result_id, resolve_meta_asset_path
from image_utils import load_classification_image


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one crop example per class.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--taxonomy-json", type=Path, default=TAXONOMY_JSON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "examples" / "class_crops",
    )
    args = parser.parse_args()

    with args.taxonomy_json.open("r", encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    categories = [item["name"] for item in taxonomy.get("categories", [])]

    meta_df = pd.read_csv(args.meta_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    for category in tqdm(categories, desc="class crops"):
        class_rows = meta_df[meta_df["category_name"] == category]
        saved = False
        for _, row in class_rows.iterrows():
            frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
            mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
            if frame_path is None or mask_path is None:
                continue
            if not frame_path.is_file() or not mask_path.is_file():
                continue

            crop = load_classification_image(frame_path, mask_path, crop_mode="bbox")
            filename = f"{safe_filename(category)}.jpg"
            output_path = args.output_dir / filename
            crop.save(output_path, quality=92)
            frame_id = frame_result_id(str(row["gx_id"]), frame_path)
            index_rows.append(
                {
                    "category_name": category,
                    "frame_id": frame_id,
                    "gx_id": str(row["gx_id"]),
                    "frame_path": str(frame_path),
                    "mask_path": str(mask_path),
                    "output_path": str(output_path),
                }
            )
            saved = True
            break
        if not saved:
            print(f"Warning: no valid sample found for category '{category}'")

    index_csv = args.output_dir.parent / "class_crops_index.csv"
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)
    print(f"Saved {len(index_rows)} class crops to {args.output_dir}")
    print(f"Saved index: {index_csv}")


if __name__ == "__main__":
    main()
