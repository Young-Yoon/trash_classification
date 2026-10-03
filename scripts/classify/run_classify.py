#!/usr/bin/env python3
"""Batch plastic image classification (CLIP / SigLIP / Gemini)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from classifiers import plastic_classify_clip, plastic_classify_gemini, plastic_classify_siglip
from config import CLASSIFICATION_RESULTS_DIR, META2_CSV
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import build_summary
from labels import load_label_sets
from presets import PRESETS


def load_models(requested_models: set[str]) -> dict:
    models: dict = {}
    if "clip" in requested_models:
        models["clip"] = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
        )
    if "siglip" in requested_models:
        models["siglip"] = pipeline(
            "zero-shot-image-classification",
            model="google/siglip-base-patch16-224",
        )
    if "gemini" in requested_models:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY is required for Gemini classification.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        models["gemini"] = genai.GenerativeModel(
            os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        )
    return models


def classify_frame(
    preset_name: str,
    preset: dict,
    row: pd.Series,
    models: dict,
    label_sets: dict[str, list[str]],
    *,
    dataset_root: Path,
    top_k: int,
) -> tuple[str, list | dict]:
    frame_path = resolve_meta_asset_path(row.get("copied_frame_path"), dataset_root)
    if frame_path is None or not frame_path.is_file():
        return "", {"error": "Frame file not found"}

    mask_path = None
    if preset["use_mask"]:
        mask_path = resolve_meta_asset_path(row.get("copied_mask_path"), dataset_root)

    labels = label_sets[preset["label_set"]]
    model_name = preset["model"]

    if model_name == "clip":
        result = plastic_classify_clip(
            frame_path,
            mask_path,
            models["clip"],
            labels,
            top_k=top_k,
        )
    elif model_name == "siglip":
        result = plastic_classify_siglip(
            frame_path,
            mask_path,
            models["siglip"],
            labels,
            top_k=top_k,
        )
    else:
        result = plastic_classify_gemini(frame_path, mask_path, models["gemini"], labels)

    return frame_result_id(row["gx_id"], frame_path), result


def run_preset(
    preset_name: str,
    preset: dict,
    meta_df: pd.DataFrame,
    models: dict,
    label_sets: dict[str, list[str]],
    *,
    dataset_root: Path,
    results_dir: Path,
    test_limit: int | None,
    resume: bool,
    top_k: int,
) -> None:
    output_dir = results_dir / preset["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running preset {preset_name} -> {output_dir}")

    processed = 0
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=preset_name):
        if test_limit is not None and processed >= test_limit:
            break

        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"), dataset_root)
        if frame_path is None:
            continue

        frame_id = frame_result_id(row["gx_id"], frame_path)
        output_filepath = output_dir / f"{frame_id}.json"
        if resume and output_filepath.is_file():
            processed += 1
            continue

        _, result = classify_frame(
            preset_name,
            preset,
            row,
            models,
            label_sets,
            dataset_root=dataset_root,
            top_k=top_k,
        )
        with output_filepath.open("w", encoding="utf-8") as handle:
            json.dump(result, handle)
        processed += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plastic image classification presets.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()) + ["all"],
        default="all",
        help="Classification preset to run.",
    )
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evaluate", action="store_true", help="Compute summary CSV after classification.")
    args = parser.parse_args()

    meta_df = pd.read_csv(args.meta_csv)
    full_labels, concise_labels = load_label_sets()
    label_sets = {"full": full_labels, "concise": concise_labels}

    selected = list(PRESETS.keys()) if args.preset == "all" else [args.preset]
    requested_models = {PRESETS[name]["model"] for name in selected}
    models = load_models(requested_models)

    for preset_name in selected:
        preset = PRESETS[preset_name]
        preset_top_k = preset.get("top_k", args.top_k)
        run_preset(
            preset_name,
            preset,
            meta_df,
            models,
            label_sets,
            dataset_root=args.meta_csv.parent,
            results_dir=args.results_dir,
            test_limit=args.test_limit,
            resume=args.resume,
            top_k=preset_top_k,
        )

    if args.evaluate or args.preset == "all":
        summary = build_summary(selected, meta_csv=args.meta_csv, results_dir=args.results_dir)
        from config import CLASSIFICATION_SUMMARY_CSV

        CLASSIFICATION_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(CLASSIFICATION_SUMMARY_CSV, index=False)
        print(summary.to_string(index=False))
        print(f"\nSaved summary: {CLASSIFICATION_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
