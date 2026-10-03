#!/usr/bin/env python3
"""Plot segment version comparison (v1-v9) for Top-1 success rate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import CLASSIFICATION_SEGMENT_SUMMARY_CSV, RESULTS_DIR

SEGMENT_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]
COLORS = [
    "#2563eb", "#16a34a", "#9333ea", "#db2777", "#ea580c", "#0891b2",
    "#7c3aed", "#ca8a04", "#dc2626",
]


def plot_segment_versions(df: pd.DataFrame, output_path: Path) -> None:
    methods = sorted(df["base_preset"].unique())
    method_labels = {
        "clip_linear_probe_crop": "CLIP linear probe",
        "clip_lora_crop": "CLIP LoRA",
        "siglip_linear_probe_crop": "SigLIP linear probe",
        "siglip_lora_crop": "SigLIP LoRA",
        "efficientnet_b0_crop": "EfficientNet-B0",
        "late_fusion_clip_whisper_crop": "Late fusion",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    x = range(len(methods))
    width = 0.08

    for idx, version in enumerate(SEGMENT_VERSIONS):
        values = []
        for method in methods:
            row = df[(df["base_preset"] == method) & (df["segment_version"] == version)]
            values.append(float(row["top_1_success_rate"].iloc[0]) if not row.empty else 0.0)
        offset = (idx - (len(SEGMENT_VERSIONS) - 1) / 2) * width
        ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=f"Segment {version}",
            color=COLORS[idx],
        )

    ax.set_title("Top-1 classification success rate by segment mask version")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([method_labels.get(method, method) for method in methods], rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot segment v1-v9 classification comparison.")
    parser.add_argument("--input-csv", type=Path, default=CLASSIFICATION_SEGMENT_SUMMARY_CSV)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=RESULTS_DIR / "classification_segment_versions.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    plot_segment_versions(df, args.output_png)
    print(f"Saved chart: {args.output_png}")


if __name__ == "__main__":
    main()
