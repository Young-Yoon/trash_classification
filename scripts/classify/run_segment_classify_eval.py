#!/usr/bin/env python3
"""Run trained classifiers using SAM3 segment v1-v9 predicted masks."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, RESULTS_DIR, SEGMENT_DIR, SEGMENT_V1_DIR
from evaluate import build_summary
from infer_trained import run_preset
from presets import TRAINED_PRESETS
from segment_masks import (
    ALL_SEGMENT_VERSIONS,
    PREDICTED_MASK_DIRS,
    RERANK_SEGMENT_VERSIONS,
    RERANK_STRATEGIES,
    SEGMENT_CROP_MODES,
    SegmentVersion,
    export_segment_masks,
    load_segment_classification_samples,
    set_v1_results_dir,
)
from segment_rerank import load_rerank_classification_samples


def segment_preset_name(base_name: str, version: SegmentVersion) -> str:
    return f"{base_name}_seg_{version}"


def segment_output_dir(base_output_dir: str, version: SegmentVersion, experiment_suffix: str = "") -> str:
    base = f"{base_output_dir}_seg_{version}"
    return f"{base}{experiment_suffix}" if experiment_suffix else base


def mask_dir_for(version: SegmentVersion, experiment_suffix: str = "") -> Path:
    base = PREDICTED_MASK_DIRS[version]
    if not experiment_suffix:
        return base
    return base.parent / f"{base.name}{experiment_suffix}"


def rerank_cache_dir_for(version: SegmentVersion, experiment_suffix: str = "") -> Path:
    cache_key = "v7" if version in {"v13", "v14", "v15", "v16"} else version
    suffix = experiment_suffix or ""
    return SEGMENT_DIR / f"rerank_candidates_{cache_key}{suffix}"


def build_segment_presets(version: SegmentVersion, experiment_suffix: str = "") -> dict[str, dict]:
    presets: dict[str, dict] = {}
    for base_name, preset in TRAINED_PRESETS.items():
        name = segment_preset_name(base_name, version)
        if experiment_suffix:
            name = f"{name}{experiment_suffix}"
        preset_copy = deepcopy(preset)
        preset_copy["output_dir"] = segment_output_dir(
            preset["output_dir"], version, experiment_suffix
        )
        preset_copy["segment_version"] = version
        preset_copy["mask_source"] = f"segment_{version}"
        preset_copy["crop_mode"] = SEGMENT_CROP_MODES[version]
        preset_copy["rerank"] = version in RERANK_SEGMENT_VERSIONS
        preset_copy["rerank_strategy"] = RERANK_STRATEGIES.get(version, "confidence")
        presets[name] = preset_copy
    return presets


def parse_versions(raw: str) -> list[SegmentVersion]:
    if raw == "all":
        return list(ALL_SEGMENT_VERSIONS)
    if raw == "both":
        return ["v1", "v2"]
    return [raw]  # type: ignore[list-item]


def load_samples_for_version(
    version: SegmentVersion,
    meta_csv: Path,
    *,
    v1_results_dir: Path | None = None,
    experiment_suffix: str = "",
):
    mask_dir = mask_dir_for(version, experiment_suffix)
    if version in RERANK_SEGMENT_VERSIONS:
        return load_rerank_classification_samples(
            version,
            meta_csv=meta_csv,
            crop_mode=SEGMENT_CROP_MODES[version],
            v1_results_dir=v1_results_dir,
            cache_dir=rerank_cache_dir_for(version, experiment_suffix),
        )
    return load_segment_classification_samples(
        version,
        meta_csv=meta_csv,
        export_if_missing=False,
        v1_results_dir=v1_results_dir,
        mask_dir=mask_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained classifiers on SAM3 segment v1-v9 predicted masks."
    )
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default="",
        help="Suffix for mask/cache/classification output dirs (e.g. _hand_glove).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Classification results root (default: classification_results[experiment_suffix]).",
    )
    parser.add_argument(
        "--segment-version",
        choices=[*ALL_SEGMENT_VERSIONS, "both", "all"],
        default="all",
    )
    parser.add_argument(
        "--preset",
        choices=list(TRAINED_PRESETS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--v1-results-dir",
        type=Path,
        default=None,
        help="Override v1 SAM3 pickle directory (default: config SEGMENT_V1_DIR).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "classification_segment_summary.csv",
    )
    args = parser.parse_args()

    v1_results_dir = args.v1_results_dir
    experiment_suffix = args.experiment_suffix
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = (
            CLASSIFICATION_RESULTS_DIR.parent
            / f"{CLASSIFICATION_RESULTS_DIR.name}{experiment_suffix}"
        )
    if v1_results_dir is not None:
        set_v1_results_dir(v1_results_dir)
        print(f"Using v1 results dir: {v1_results_dir}")
    if experiment_suffix:
        print(f"Using experiment suffix: {experiment_suffix}")
    print(f"Using results dir: {results_dir}")

    versions = parse_versions(args.segment_version)
    base_presets = list(TRAINED_PRESETS.keys()) if args.preset == "all" else [args.preset]
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_preset_names: list[str] = []
    all_preset_map: dict[str, dict] = {}

    for version in versions:
        if not args.skip_export and version not in RERANK_SEGMENT_VERSIONS:
            export_segment_masks(
                version,
                meta_csv=args.meta_csv,
                output_dir=mask_dir_for(version, experiment_suffix),
                v1_results_dir=v1_results_dir,
            )
        samples = load_samples_for_version(
            version,
            args.meta_csv,
            v1_results_dir=v1_results_dir,
            experiment_suffix=experiment_suffix,
        )
        segment_presets = build_segment_presets(version, experiment_suffix)

        for base_name in base_presets:
            preset_name = segment_preset_name(base_name, version)
            if experiment_suffix:
                preset_name = f"{preset_name}{experiment_suffix}"
            preset = segment_presets[preset_name]
            run_preset(
                preset_name,
                preset,
                samples=samples,
                results_dir=results_dir,
                batch_size=args.batch_size,
                top_k=args.top_k,
                resume=args.resume,
                device=device,
            )
            all_preset_names.append(preset_name)
            all_preset_map[preset_name] = preset

    summary = build_summary(
        all_preset_names,
        meta_csv=args.meta_csv,
        results_dir=results_dir,
        preset_map=all_preset_map,
    )
    summary.insert(0, "segment_version", summary["preset"].str.extract(r"_seg_(v\d+)")[0])
    summary.insert(
        1,
        "base_preset",
        summary["preset"].str.replace(r"_seg_v\d+(?:_hand_glove|.*)?$", "", regex=True),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
