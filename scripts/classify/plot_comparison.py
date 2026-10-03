#!/usr/bin/env python3
"""Plot classification comparison chart from classification_comparison.csv."""

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

from config import CLASSIFICATION_COMPARISON_CSV, RESULTS_DIR

MASKS = [
    ("gt", "GT mask"),
    ("segment_v1", "Segment v1"),
    ("segment_v2", "Segment v2"),
    ("segment_v5", "Segment v5"),
]
METRICS = [
    ("top_1", "Top-1"),
    ("top_3", "Top-3"),
    ("top_5", "Top-5"),
]


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    methods = df["method"].tolist()
    x = range(len(methods))
    width = 0.18
    colors = ["#2563eb", "#16a34a", "#ea580c", "#9333ea"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    fig.suptitle(
        "Plastic classification success rate by mask source\n"
        "Zero-shot + supervised models (camera1_plastics_meta2, n=3538)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (metric_suffix, metric_label) in zip(axes, METRICS):
        for idx, (mask_key, mask_label) in enumerate(MASKS):
            values = df[f"{mask_key}_{metric_suffix}"].tolist()
            offset = (idx - (len(MASKS) - 1) / 2) * width
            bars = ax.bar(
                [pos + offset for pos in x],
                values,
                width=width,
                label=mask_label,
                color=colors[idx],
            )
            for bar, value in zip(bars, values):
                if value >= 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_title(metric_label)
        ax.set_ylabel("Success rate (%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classification comparison chart.")
    parser.add_argument("--input-csv", type=Path, default=CLASSIFICATION_COMPARISON_CSV)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=RESULTS_DIR / "classification_comparison.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    plot_comparison(df, args.output_png)
    print(f"Saved chart: {args.output_png}")


if __name__ == "__main__":
    main()
