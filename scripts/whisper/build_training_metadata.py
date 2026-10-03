#!/usr/bin/env python3
"""Build training metadata CSV from local audio + timestamp group labels."""

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

from config import TRAINING_METADATA_CSV
from data_utils import build_training_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LoRA training metadata CSV.")
    parser.add_argument("--output-csv", type=Path, default=TRAINING_METADATA_CSV)
    args = parser.parse_args()
    df = build_training_metadata(args.output_csv)
    print(df.head())


if __name__ == "__main__":
    main()
