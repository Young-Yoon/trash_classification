#!/usr/bin/env python3
"""Evaluate saved image classification JSON outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_RESULTS_DIR, CLASSIFICATION_SUMMARY_CSV, META2_CSV
from evaluate import build_summary
from presets import ALL_PRESET_NAMES, ALL_PRESETS, PRESETS, TRAINED_PRESETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved classification JSON outputs.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument("--output-csv", type=Path, default=CLASSIFICATION_SUMMARY_CSV)
    parser.add_argument(
        "--preset",
        choices=ALL_PRESET_NAMES + ["all", "zero_shot", "trained"],
        default="all",
    )
    args = parser.parse_args()

    if args.preset == "all":
        presets = ALL_PRESET_NAMES
    elif args.preset == "zero_shot":
        presets = list(PRESETS.keys())
    elif args.preset == "trained":
        presets = list(TRAINED_PRESETS.keys())
    else:
        presets = [args.preset]
    summary = build_summary(presets, meta_csv=args.meta_csv, results_dir=args.results_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
