#!/usr/bin/env python3
"""Run zero-shot CLIP/SigLIP classification on SAM3 segment predicted masks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from classifiers import plastic_classify_clip, plastic_classify_siglip
from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, RESULTS_DIR, SEGMENT_DIR
from dataset_paths import frame_result_id
from evaluate import build_summary, predicted_label_name
from infer_trained import pick_best_candidate_index
from labels import load_label_sets
from presets import PRESETS
from segment_masks import (
    ALL_SEGMENT_VERSIONS,
    RERANK_SEGMENT_VERSIONS,
    RERANK_STRATEGIES,
    SEGMENT_CROP_MODES,
    SegmentVersion,
    export_segment_masks,
    load_segment_classification_samples,
)
from segment_rerank import RerankClassificationSample, load_rerank_classification_samples
from training_data import load_text_category_map

ZERO_SHOT_SEGMENT_PRESETS = {
    "clip_full_labels_top5": PRESETS["clip_full_labels_top5"],
    "siglip_full_labels_top5": PRESETS["siglip_full_labels_top5"],
}


def segment_preset_name(base_name: str, version: SegmentVersion) -> str:
    return f"{base_name}_seg_{version}"


def segment_output_dir(base_output_dir: str, version: SegmentVersion) -> str:
    return f"{base_output_dir}_seg_{version}"


def build_segment_zeroshot_presets(version: SegmentVersion) -> dict[str, dict]:
    presets: dict[str, dict] = {}
    for base_name, preset in ZERO_SHOT_SEGMENT_PRESETS.items():
        name = segment_preset_name(base_name, version)
        preset_copy = deepcopy(preset)
        preset_copy["output_dir"] = segment_output_dir(preset["output_dir"], version)
        preset_copy["crop_mode"] = SEGMENT_CROP_MODES[version]
        presets[name] = preset_copy
    return presets


def run_zeroshot_preset(
    preset_name: str,
    preset: dict,
    samples,
    *,
    models: dict,
    label_sets: dict[str, list[str]],
    results_dir: Path,
    top_k: int,
    resume: bool,
) -> None:
    output_dir = results_dir / preset["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = label_sets[preset["label_set"]]
    model_name = preset["model"]
    crop_mode = preset.get("crop_mode", "bbox")
    print(f"Zero-shot {preset_name} -> {output_dir} (crop_mode={crop_mode})")

    for sample in tqdm(samples, desc=preset_name):
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        output_path = output_dir / f"{frame_id}.json"
        if resume and output_path.is_file():
            continue

        mask_path = sample.mask_path if preset.get("use_mask", True) else None
        if model_name == "clip":
            result = plastic_classify_clip(
                sample.frame_path,
                mask_path,
                models["clip"],
                labels,
                top_k=top_k,
                crop_mode=crop_mode,
            )
        else:
            result = plastic_classify_siglip(
                sample.frame_path,
                mask_path,
                models["siglip"],
                labels,
                top_k=top_k,
                crop_mode=crop_mode,
            )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle)


def zeroshot_results_to_probs(
    results: list[dict] | dict,
    concise_labels: list[str],
) -> np.ndarray:
    probs = np.zeros(len(concise_labels), dtype=np.float32)
    if isinstance(results, dict):
        return probs
    for item in results:
        short_label = predicted_label_name(item["label"])
        if short_label in concise_labels:
            idx = concise_labels.index(short_label)
            probs[idx] = max(probs[idx], float(item["score"]))
    total = float(probs.sum())
    if total > 0:
        probs /= total
    return probs


def classify_sample_candidates(
    sample: RerankClassificationSample,
    *,
    preset: dict,
    models: dict,
    full_labels: list[str],
    concise_labels: list[str],
    top_k: int,
) -> np.ndarray:
    model_name = preset["model"]
    crop_mode = preset.get("crop_mode", "bbox")
    candidate_probs: list[np.ndarray] = []

    for mask_path in sample.candidate_mask_paths:
        if model_name == "clip":
            result = plastic_classify_clip(
                sample.frame_path,
                mask_path,
                models["clip"],
                full_labels,
                top_k=top_k,
                crop_mode=crop_mode,
            )
        else:
            result = plastic_classify_siglip(
                sample.frame_path,
                mask_path,
                models["siglip"],
                full_labels,
                top_k=top_k,
                crop_mode=crop_mode,
            )
        candidate_probs.append(zeroshot_results_to_probs(result, concise_labels))

    if not candidate_probs:
        return np.zeros(len(concise_labels), dtype=np.float32)
    return np.stack(candidate_probs, axis=0)


def run_zeroshot_rerank_preset(
    preset_name: str,
    preset: dict,
    samples: list[RerankClassificationSample],
    *,
    version: SegmentVersion,
    models: dict,
    full_labels: list[str],
    concise_labels: list[str],
    results_dir: Path,
    top_k: int,
    resume: bool,
) -> None:
    output_dir = results_dir / preset["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rerank_strategy = RERANK_STRATEGIES.get(version, "confidence")
    id2label = {idx: label for idx, label in enumerate(concise_labels)}
    clip_to_category = load_text_category_map()
    print(
        f"Zero-shot rerank {preset_name} -> {output_dir} "
        f"(strategy={rerank_strategy}, candidates)"
    )

    for sample in tqdm(samples, desc=preset_name):
        frame_id = frame_result_id(sample.gx_id, sample.frame_path)
        output_path = output_dir / f"{frame_id}.json"
        if resume and output_path.is_file():
            continue

        if not sample.candidate_mask_paths:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump({"error": "no candidates"}, handle)
            continue

        probs = classify_sample_candidates(
            sample,
            preset=preset,
            models=models,
            full_labels=full_labels,
            concise_labels=concise_labels,
            top_k=top_k,
        )
        text_category = clip_to_category.get(sample.clip_id or "", "")
        best_idx = pick_best_candidate_index(
            probs,
            strategy=rerank_strategy,
            candidate_scores=sample.candidate_scores,
            candidate_hand_ious=sample.candidate_hand_ious,
            id2label=id2label,
            text_category=text_category,
        )
        best_probs = probs[best_idx]
        ranked = sorted(
            (
                {"label": concise_labels[idx], "score": float(score)}
                for idx, score in enumerate(best_probs)
                if score > 0
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(ranked, handle)


def rerank_cache_dir_for(version: SegmentVersion) -> Path:
    cache_key = "v7" if version in {"v13", "v14", "v15", "v16"} else version
    return SEGMENT_DIR / f"rerank_candidates_{cache_key}"


def parse_versions(raw: str) -> list[SegmentVersion]:
    if raw == "all":
        return list(ALL_SEGMENT_VERSIONS)
    if raw == "both":
        return ["v1", "v2"]
    return [raw]  # type: ignore[list-item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot classify with segment v1-v6 masks.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument(
        "--segment-version",
        choices=[*ALL_SEGMENT_VERSIONS, "both", "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "classification_segment_zeroshot_summary.csv",
    )
    args = parser.parse_args()

    versions = parse_versions(args.segment_version)
    full_labels, concise_labels = load_label_sets()
    label_sets = {"full": full_labels, "concise": concise_labels}
    models = {
        "clip": pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32"),
        "siglip": pipeline("zero-shot-image-classification", model="google/siglip-base-patch16-224"),
    }

    all_preset_names: list[str] = []
    all_preset_map: dict[str, dict] = {}

    for version in versions:
        segment_presets = build_segment_zeroshot_presets(version)
        if version in RERANK_SEGMENT_VERSIONS:
            samples = load_rerank_classification_samples(
                version,
                meta_csv=args.meta_csv,
                crop_mode=SEGMENT_CROP_MODES[version],
                cache_dir=rerank_cache_dir_for(version),
            )
            for preset_name, preset in segment_presets.items():
                run_zeroshot_rerank_preset(
                    preset_name,
                    preset,
                    samples,
                    version=version,
                    models=models,
                    full_labels=full_labels,
                    concise_labels=concise_labels,
                    results_dir=args.results_dir,
                    top_k=args.top_k,
                    resume=args.resume,
                )
                all_preset_names.append(preset_name)
                all_preset_map[preset_name] = preset
            continue

        export_segment_masks(version, meta_csv=args.meta_csv)
        samples = load_segment_classification_samples(
            version,
            meta_csv=args.meta_csv,
            export_if_missing=False,
        )
        for preset_name, preset in segment_presets.items():
            run_zeroshot_preset(
                preset_name,
                preset,
                samples,
                models=models,
                label_sets=label_sets,
                results_dir=args.results_dir,
                top_k=args.top_k,
                resume=args.resume,
            )
            all_preset_names.append(preset_name)
            all_preset_map[preset_name] = preset

    summary = build_summary(
        all_preset_names,
        meta_csv=args.meta_csv,
        results_dir=args.results_dir,
        preset_map=all_preset_map,
    )
    version_pattern = re.compile(r"_seg_(v\d+)$")
    summary.insert(0, "segment_version", summary["preset"].str.extract(version_pattern)[0])
    summary.insert(
        1,
        "base_preset",
        summary["preset"].str.replace(r"_seg_v\d+$", "", regex=True),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
