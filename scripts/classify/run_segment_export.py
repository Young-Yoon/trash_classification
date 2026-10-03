#!/usr/bin/env python3
"""Export predicted segment masks for v1-v6."""

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

from config import META2_CSV
from segment_masks import ALL_SEGMENT_VERSIONS, SEGMENT_VERSION_DESCRIPTIONS, export_segment_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM3 predicted masks for segment v1-v6.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument(
        "--segment-version",
        choices=[*ALL_SEGMENT_VERSIONS, "all"],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    versions = list(ALL_SEGMENT_VERSIONS) if args.segment_version == "all" else [args.segment_version]
    for version in versions:
        print(f"\n{version}: {SEGMENT_VERSION_DESCRIPTIONS[version]}")
        export_segment_masks(version, meta_csv=args.meta_csv, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
