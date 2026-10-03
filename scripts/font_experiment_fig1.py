#!/usr/bin/env python3
"""Render Fig. 1 at several label font sizes to compare legibility in the paper."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from visualize_dataset_samples import (
    build_category_items,
    pick_diverse_rep_samples,
    render_pipeline_figure,
)
from config import META2_CSV, TAXONOMY_JSON

# Figure canvas is 12.4 in wide; the paper prints it at 0.92\textwidth = 6.5 in.
PAPER_SCALE = 6.5 / 12.4

# (label pt, subtitle pt, canvas width in, canvas height in, left panel width cap)
VARIANTS = [
    (8.0, 10.0, 12.4, 4.15, 1.00),
    (14.0, 15.0, 12.4, 4.15, 0.32),
    (14.0, 15.0, 12.4, 5.00, 0.30),
    (14.0, 16.0, 12.4, 6.00, 0.28),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "demo" / "fig1_fontexp"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(META2_CSV)
    with TAXONOMY_JSON.open() as handle:
        taxonomy = json.load(handle)
    categories = [item["name"] for item in taxonomy["categories"]]

    rep_row = pick_diverse_rep_samples(meta_df, n=1)[0]
    category_items = build_category_items(meta_df, categories)

    for label_fs, subtitle_fs, w_in, h_in, panel_cap in VARIANTS:
        path = out_dir / f"fig1_label{label_fs:g}_{w_in:g}x{h_in:g}_cap{panel_cap:g}.png"
        _, fitted = render_pipeline_figure(
            rep_row,
            category_items,
            path,
            label_fontsize=label_fs,
            subtitle_fontsize=subtitle_fs,
            fig_w_in=w_in,
            fig_h_in=h_in,
            left_panel_max_w=panel_cap,
        )
        scale = 6.5 / w_in
        print(
            f"{path.name}: requested {label_fs:g}pt, fitted {fitted:g}pt "
            f"-> {fitted * scale:.1f}pt on the page, "
            f"figure height {h_in * scale:.2f} in"
        )


if __name__ == "__main__":
    main()
