#!/usr/bin/env python3
"""Plot confusion matrices for best EfficientNet-B0 and Late fusion models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from aggregate_pipeline_by_timestamp_group import (
    build_group_key,
    load_frame_group_map,
    load_prediction_label,
    majority_vote_label,
    parse_gt_categories,
)
from build_pipeline_end_to_end import segment_output_dir
from config import CLASSIFICATION_RESULTS_DIR, META2_CSV, PROJECT_ROOT, RESULTS_DIR, TIMESTAMP_GROUPS_CSV
from dataset_paths import frame_result_id, resolve_meta_asset_path
from evaluate import result_json_path
from presets import TRAINED_PRESETS

BEST_MODELS = [
    {
        "base_preset": "efficientnet_b0_crop",
        "segment_version": "v15",
        "label": "EfficientNet-B0 (v15 best)",
        "slug": "efficientnet_b0_v15",
    },
    {
        "base_preset": "late_fusion_efficientnet_whisper_crop",
        "segment_version": "v15",
        "label": "Late fusion EN (v15 best)",
        "slug": "late_fusion_en_v15",
    },
]

NO_PREDICTION_LABEL = "No prediction"

ALL_LABELS = [
    "Black Plastic Containers",
    "Containers #1/Clamshells",
    "Containers #2 Colored",
    "Containers #2 Natural",
    "Containers #3-7 (Not #5)",
    "Containers #5/PP",
    "Deposit #1 Bottles",
    "Garbage Bags",
    "Non-Deposit",
    "Other #1 Bottles",
    "Other Plastic Film",
    "Other Plastics",
    "Other Rigid/Bulky Plastics",
    "Other Unnumbered Containers & Fragments",
    "Plastic",
    NO_PREDICTION_LABEL,
]

SHORT_LABELS = {
    "Black Plastic Containers": "Black Plastic",
    "Containers #1/Clamshells": "Clamshells",
    "Containers #2 Colored": "C2 Colored",
    "Containers #2 Natural": "C2 Natural",
    "Containers #3-7 (Not #5)": "C3-7",
    "Containers #5/PP": "C5/PP",
    "Deposit #1 Bottles": "Deposit #1",
    "Garbage Bags": "Garbage Bags",
    "Non-Deposit": "Non-Deposit",
    "Other #1 Bottles": "Other #1",
    "Other Plastic Film": "Plastic Film",
    "Other Plastics": "Other Plastics",
    "Other Rigid/Bulky Plastics": "Rigid/Bulky",
    "Other Unnumbered Containers & Fragments": "Unnumbered",
    "Plastic": "Plastic",
    NO_PREDICTION_LABEL: "No prediction",
}

ABBREV_LABELS = {
    "Black Plastic Containers": "BlkPlst",
    "Containers #1/Clamshells": "Clam",
    "Containers #2 Colored": "C2Col",
    "Containers #2 Natural": "C2Nat",
    "Containers #3-7 (Not #5)": "C3-7",
    "Containers #5/PP": "C5/PP",
    "Deposit #1 Bottles": "Dep1",
    "Garbage Bags": "GbBag",
    "Non-Deposit": "NonDep",
    "Other #1 Bottles": "Oth1",
    "Other Plastic Film": "PlFilm",
    "Other Plastics": "OthPlst",
    "Other Rigid/Bulky Plastics": "RigBlk",
    "Other Unnumbered Containers & Fragments": "Unnum",
    "Plastic": "Plastic",
    NO_PREDICTION_LABEL: "NoPred",
}


def load_prediction_map(
    meta_df: pd.DataFrame,
    *,
    base_preset: str,
    segment_version: str,
    results_dir: Path,
) -> dict[str, str | None]:
    output_dir = results_dir / segment_output_dir(
        TRAINED_PRESETS[base_preset]["output_dir"],
        segment_version,
    )
    pred_map: dict[str, str | None] = {}
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        pred_map[frame_id] = load_prediction_label(result_json_path(row, output_dir))
    return pred_map


def representative_group_label(group_frames: pd.DataFrame) -> str | None:
    gt_categories = parse_gt_categories(group_frames["category_names"].iloc[0])
    frame_categories = group_frames["frame_category_name"].astype(str)
    if gt_categories:
        in_group = [cat for cat in frame_categories if cat in gt_categories]
        if in_group:
            return pd.Series(in_group).mode().iloc[0]
        return sorted(gt_categories)[0]
    if frame_categories.empty:
        return None
    return frame_categories.mode().iloc[0]


def collect_frame_pairs(
    meta_df: pd.DataFrame,
    prediction_map: dict[str, str | None],
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None:
            continue
        frame_id = frame_result_id(str(row["gx_id"]), frame_path)
        pred = prediction_map.get(frame_id)
        if pred is None:
            continue
        y_true.append(str(row["category_name"]))
        y_pred.append(pred)
    return y_true, y_pred


def collect_group_majority_pairs(
    frame_group_df: pd.DataFrame,
    prediction_map: dict[str, str | None],
) -> tuple[list[str], list[str], list[bool]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    pipeline_ok: list[bool] = []

    for _, group_frames in frame_group_df.groupby("group_key", sort=False):
        gt_label = representative_group_label(group_frames)
        if gt_label is None:
            continue
        gt_categories = parse_gt_categories(group_frames["category_names"].iloc[0])
        frame_categories = set(group_frames["frame_category_name"].astype(str))
        predictions = [prediction_map.get(frame_id) for frame_id in group_frames["frame_id"]]
        majority_pred = majority_vote_label(predictions) or "No prediction"
        y_true.append(gt_label)
        y_pred.append(majority_pred)
        pipeline_ok.append(
            majority_pred != "No prediction"
            and (
                majority_pred in gt_categories
                if gt_categories
                else majority_pred in frame_categories
            )
        )
    return y_true, y_pred, pipeline_ok


def collect_group_any_frame_pairs(
    frame_group_df: pd.DataFrame,
    prediction_map: dict[str, str | None],
) -> tuple[list[str], list[str], list[bool]]:
    """Clip pairs under the any-frame rule used in the pipeline table.

    A clip counts as correct when any of its frames predicts a ground-truth
    category; the residual clips are charted with their majority-vote label so
    every error still lands in a single off-diagonal cell.
    """
    y_true: list[str] = []
    y_pred: list[str] = []
    class_ok: list[bool] = []

    for _, group_frames in frame_group_df.groupby("group_key", sort=False):
        gt_label = representative_group_label(group_frames)
        if gt_label is None:
            continue
        gt_categories = parse_gt_categories(group_frames["category_names"].iloc[0])
        frame_categories = set(group_frames["frame_category_name"].astype(str))
        targets = gt_categories or frame_categories
        predictions = [prediction_map.get(frame_id) for frame_id in group_frames["frame_id"]]
        hits = [pred for pred in predictions if pred and pred in targets]
        if hits:
            any_frame_pred = gt_label
        else:
            any_frame_pred = majority_vote_label(predictions) or NO_PREDICTION_LABEL
        y_true.append(gt_label)
        y_pred.append(any_frame_pred)
        class_ok.append(bool(hits))
    return y_true, y_pred, class_ok


def normalize_labels(labels: list[str], *, include_no_prediction: bool) -> list[str]:
    allowed = set(ALL_LABELS if include_no_prediction else ALL_LABELS[:-1])
    return [label if label in allowed else label for label in labels]


DEMO_DIR = PROJECT_ROOT / "demo"

MISCLASS_RATE_DEFAULT = 0.07
MISCLASS_RATE_LOW_COUNT = 0.15
GROUP_LOW_COUNT_BOUND = 4
FRAME_LOW_COUNT_BOUND = 30

# Red (low) -> white (mid) -> blue (high).
# Upper half keeps G > R so 70–80% reads as sky blue, not lavender.
CAT_ACC_CMAP = LinearSegmentedColormap.from_list(
    "cat_acc_red_white_blue",
    [
        (0.82, 0.18, 0.18),
        (1.00, 1.00, 1.00),
        (0.10, 0.50, 0.92),
    ],
    N=256,
)


def compute_category_accuracy(cm: np.ndarray) -> np.ndarray:
    """Per ground-truth-class recall: diagonal / row sum."""
    row_sums = cm.sum(axis=1)
    cat_acc = np.full(cm.shape[0], np.nan)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cat_acc[i] = cm[i, i] / row_sums[i]
    return cat_acc


def compute_row_misclass_rates(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1)
    rates = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if row_sums[i] <= 0:
            continue
        for j in range(cm.shape[1]):
            if i != j:
                rates[i, j] = cm[i, j] / row_sums[i]
    return rates


def should_highlight_misclass(count: int, rate: float, *, eval_mode: str) -> bool:
    if count <= 0:
        return False
    low_count_bound = GROUP_LOW_COUNT_BOUND if eval_mode.startswith("group") else FRAME_LOW_COUNT_BOUND
    threshold = MISCLASS_RATE_LOW_COUNT if count <= low_count_bound else MISCLASS_RATE_DEFAULT
    return rate >= threshold


def misclass_highlight_note(eval_mode: str) -> str:
    if eval_mode.startswith("group"):
        return (
            f"Misclass highlight: ≥{MISCLASS_RATE_DEFAULT:.0%} "
            f"(count>{GROUP_LOW_COUNT_BOUND}), "
            f"≥{MISCLASS_RATE_LOW_COUNT:.0%} (count≤{GROUP_LOW_COUNT_BOUND})"
        )
    return (
        f"Misclass highlight: ≥{MISCLASS_RATE_DEFAULT:.0%} "
        f"(count>{FRAME_LOW_COUNT_BOUND}), "
        f"≥{MISCLASS_RATE_LOW_COUNT:.0%} (count≤{FRAME_LOW_COUNT_BOUND})"
    )


FONT = {
    "title": 20,
    "axis_label": 36,
    "tick": 36,
    "cell": 32,
    "cell_highlight": 28,
    "cell_highlight_pct": 21,
    "cat_acc": 28,
    "cbar_label": 32,
    "cbar_tick": 32,
    "grid_title": 26,
    "grid_col_title": 52,
    "grid_panel_title": 44,
    "grid_suptitle": 52,
    "grid_xtick": 24,
}


def _tighten_matrix_axes(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.margins(0)
    ax.set_aspect("auto")
    ax.autoscale(False)


def _draw_confusion_matrix_on_axes(
    ax_acc: plt.Axes,
    ax_cm: plt.Axes,
    *,
    cm: np.ndarray,
    cat_acc: np.ndarray,
    row_tick_labels: list[str],
    col_tick_labels: list[str],
    title: str,
    eval_mode: str,
    accuracy: float,
    n_samples: int,
    show_ylabel_ticks: bool = True,
    show_ylabel_label: bool = True,
    show_xticklabels: bool = True,
    show_xlabel: bool = True,
    show_cat_acc_label: bool = True,
    show_cbar_label: bool = True,
    predicted_note: str | None = None,
    xtick_fontsize: float | None = None,
    panel_title: str | None = None,
    panel_title_pad: float | None = None,
    include_misclass_note: bool = True,
    fonts: dict[str, float] | None = None,
    show_highlight_pct: bool = True,
    cell_linewidth: float = 2.2,
    cat_acc_fontweight: str = "bold",
    cell_fontfamily: str | None = None,
    scale_cell_by_digits: bool = False,
    cbar_size: str = "1.8%",
) -> None:
    fonts = fonts if fonts is not None else FONT
    xtick_size = xtick_fontsize if xtick_fontsize is not None else fonts["tick"]
    misclass_rates = compute_row_misclass_rates(cm)

    acc_values = cat_acc.reshape(-1, 1)
    acc_display = np.ma.masked_where(np.isnan(acc_values), acc_values)
    im_acc = ax_acc.imshow(
        acc_display,
        interpolation="none",
        cmap=CAT_ACC_CMAP,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    im_cm = ax_cm.imshow(cm, interpolation="none", cmap="Blues", aspect="auto")
    n_rows, n_cols = cm.shape
    _tighten_matrix_axes(ax_acc, n_rows, 1)
    _tighten_matrix_axes(ax_cm, n_rows, n_cols)

    ax_acc.set_xticks([0])
    if show_cat_acc_label:
        ax_acc.set_xticklabels(["Cat\nacc"], fontsize=fonts["tick"])
    else:
        ax_acc.set_xticklabels([])
    ax_acc.tick_params(axis="x", labelbottom=show_cat_acc_label)
    ax_acc.set_xlabel("")
    if show_ylabel_label:
        ax_acc.set_ylabel("Ground truth", fontsize=fonts["axis_label"])
    else:
        ax_acc.set_ylabel("")
    ax_acc.set_yticks(np.arange(len(row_tick_labels)))
    if show_ylabel_ticks:
        ax_acc.set_yticklabels(row_tick_labels, fontsize=fonts["tick"])
    else:
        ax_acc.set_yticklabels([])
    ax_acc.tick_params(
        axis="both",
        labelsize=fonts["tick"],
        pad=1,
        left=show_ylabel_ticks,
        labelleft=show_ylabel_ticks,
    )

    ax_cm.set_xticks(np.arange(len(col_tick_labels)))
    if show_xticklabels:
        ax_cm.set_xticklabels(col_tick_labels, fontsize=xtick_size)
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax_cm.set_xticklabels([])
    ax_cm.set_yticks(np.arange(len(row_tick_labels)))
    ax_cm.tick_params(axis="y", left=False, labelleft=False)
    if show_xlabel:
        if predicted_note:
            ax_cm.set_xlabel("")
            note_y = -0.13
            predicted_font = fonts["axis_label"]
            misclass_font = predicted_font * 0.8
            ax_cm.text(
                0.5,
                note_y,
                "Predicted",
                transform=ax_cm.transAxes,
                ha="right",
                va="top",
                fontsize=predicted_font,
            )
            ax_cm.text(
                0.5,
                note_y,
                f" ({predicted_note})",
                transform=ax_cm.transAxes,
                ha="left",
                va="top",
                fontsize=misclass_font,
            )
        else:
            ax_cm.set_xlabel("Predicted", fontsize=fonts["axis_label"])
    else:
        ax_cm.set_xlabel("")
    ax_cm.set_ylabel("")
    if panel_title is not None:
        title_kwargs: dict = {"fontsize": fonts["grid_panel_title"]}
        if panel_title_pad is not None:
            title_kwargs["pad"] = panel_title_pad
        ax_cm.set_title(panel_title, **title_kwargs)
    else:
        title_lines = [f"{title}", f"Exact match: {accuracy * 100:.2f}% (n={n_samples})"]
        if include_misclass_note:
            title_lines.append(misclass_highlight_note(eval_mode))
        ax_cm.set_title("\n".join(title_lines), fontsize=fonts["title"])
    ax_cm.tick_params(
        axis="x",
        labelsize=xtick_size,
        pad=0.5,
        labelbottom=show_xticklabels,
    )

    count_thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0

    def _count_fontsize(count: int, *, highlight: bool) -> float:
        base = fonts["cell_highlight"] if highlight else fonts["cell"]
        if not scale_cell_by_digits:
            return base
        digits = len(str(count))
        if digits >= 3:
            return base * 0.88
        return base * 1.18

    cell_text_kwargs: dict = {"ha": "center", "va": "center", "zorder": 4, "clip_on": True}
    if cell_fontfamily:
        cell_text_kwargs["fontfamily"] = cell_fontfamily

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            if count == 0:
                continue
            rate = misclass_rates[i, j]
            highlight = i != j and should_highlight_misclass(count, rate, eval_mode=eval_mode)
            if highlight:
                ax_cm.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#C0392B",
                        linewidth=cell_linewidth,
                        zorder=3,
                    )
                )
                text_color = "#8B0000"
                fontweight = "bold" if show_highlight_pct else "normal"
                ax_cm.text(
                    j,
                    i - 0.14 if show_highlight_pct else i,
                    str(count),
                    color=text_color,
                    fontsize=_count_fontsize(count, highlight=True),
                    fontweight=fontweight,
                    **cell_text_kwargs,
                )
                if show_highlight_pct:
                    ax_cm.text(
                        j,
                        i + 0.18,
                        f"({rate * 100:.0f}%)",
                        color=text_color,
                        fontsize=fonts["cell_highlight_pct"],
                        fontweight=fontweight,
                        **cell_text_kwargs,
                    )
                continue
            text_color = "white" if count > count_thresh else "black"
            ax_cm.text(
                j,
                i,
                str(count),
                color=text_color,
                fontsize=_count_fontsize(count, highlight=False),
                fontweight="normal",
                **cell_text_kwargs,
            )

    for i, value in enumerate(cat_acc):
        if np.isnan(value):
            label = "—"
        else:
            label = f"{value * 100:.0f}%"
        acc_kwargs: dict = {
            "ha": "center",
            "va": "center",
            "color": "black",
            "fontsize": fonts["cat_acc"],
            "fontweight": cat_acc_fontweight,
            "clip_on": True,
        }
        if cell_fontfamily:
            acc_kwargs["fontfamily"] = cell_fontfamily
        ax_acc.text(0, i, label, **acc_kwargs)

    ax_acc.axvline(-0.5, color="#333333", linewidth=1.0)
    ax_acc.axvline(0.5, color="#333333", linewidth=1.0)

    cm_cbar_ax = make_axes_locatable(ax_cm).append_axes("right", size=cbar_size, pad=0.02)
    count_cbar = ax_cm.figure.colorbar(im_cm, cax=cm_cbar_ax)
    if show_cbar_label:
        count_cbar.set_label("Count", rotation=270, labelpad=10, fontsize=fonts["cbar_label"])
    count_cbar.ax.tick_params(labelsize=fonts["cbar_tick"])
    if cell_fontfamily:
        # Fix the ticks first, otherwise the formatter redraws them in the default font.
        ticks = [t for t in count_cbar.get_ticks() if cm.min() <= t <= cm.max()]
        count_cbar.set_ticks(ticks)
        count_cbar.ax.set_yticklabels(
            [f"{t:g}" for t in ticks],
            fontfamily=cell_fontfamily,
            fontsize=fonts["cbar_tick"],
        )


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    *,
    title: str,
    output_path: Path,
    row_labels: list[str],
    col_labels: list[str],
    exact_accuracy: float,
    eval_mode: str,
    show_ylabel_ticks: bool = True,
    show_ylabel_label: bool = True,
    show_xticklabels: bool = True,
    show_xlabel: bool = True,
) -> dict[str, float]:
    cm_full = confusion_matrix(y_true, y_pred, labels=col_labels)
    row_indices = [col_labels.index(label) for label in row_labels]
    cm = cm_full[row_indices, :]
    accuracy = exact_accuracy
    cat_acc = compute_category_accuracy(cm)
    row_tick_labels = [SHORT_LABELS[label] for label in row_labels]
    col_tick_labels = [SHORT_LABELS[label] for label in col_labels]

    fig, (ax_acc, ax_cm) = plt.subplots(
        1,
        2,
        figsize=(22 + len(col_labels) * 0.25, 10 + len(row_labels) * 0.55),
        gridspec_kw={"width_ratios": [1.0, len(col_labels)], "wspace": 0.0},
        sharey=True,
    )

    _draw_confusion_matrix_on_axes(
        ax_acc,
        ax_cm,
        cm=cm,
        cat_acc=cat_acc,
        row_tick_labels=row_tick_labels,
        col_tick_labels=col_tick_labels,
        title=title,
        eval_mode=eval_mode,
        accuracy=accuracy,
        n_samples=len(y_true),
        show_ylabel_ticks=show_ylabel_ticks,
        show_ylabel_label=show_ylabel_label,
        show_xticklabels=show_xticklabels,
        show_xlabel=show_xlabel,
    )

    left_margin = 0.14 if show_ylabel_ticks else 0.08
    bottom_margin = 0.14 if show_xticklabels else 0.08
    fig.subplots_adjust(
        left=left_margin,
        right=0.97,
        top=0.82,
        bottom=bottom_margin,
        wspace=0.0,
        hspace=0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {"accuracy": accuracy, "n_samples": len(y_true), "category_accuracy": cat_acc}


def save_report(
    y_true: list[str],
    y_pred: list[str],
    *,
    output_path: Path,
    labels: list[str],
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[SHORT_LABELS[label] for label in labels],
        zero_division=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _grid_panel_title(panel: dict) -> str:
    parts = [f"Exact match: {panel['accuracy'] * 100:.2f}%"]
    pipeline_accuracy = panel.get("pipeline_accuracy")
    if pipeline_accuracy is not None:
        parts.append(f"Pipeline: {pipeline_accuracy * 100:.2f}%")
    parts.append(f"n={panel['n_samples']}")
    return " | ".join(parts)


def plot_combined_grid(
    panels: list[dict],
    output_path: Path,
) -> None:
    n_model_rows = len(BEST_MODELS)
    n_cols = 2
    sample_col_labels = panels[0]["col_labels"]
    panel_figsize = (
        22 + len(sample_col_labels) * 0.25,
        10 + len(panels[0]["row_labels"]) * 0.55,
    )
    fig = plt.figure(figsize=(panel_figsize[0] * n_cols, panel_figsize[1] * n_model_rows))
    outer_gs = GridSpec(
        n_model_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[0.12, 1, 1],
        left=0.01,
        right=0.98,
        top=0.90,
        bottom=0.09,
        wspace=0.12,
        hspace=0.1,
    )

    eval_titles = {
        "frame": "Final prob (frame-level Top-1)",
        "group_majority_vote": "Group (majority vote)",
    }

    top_row_axes: dict[int, plt.Axes] = {}
    row_model_labels: dict[int, str] = {}

    for panel in panels:
        row_idx = panel["row_idx"]
        col_idx = panel["col_idx"]
        col_labels = panel["col_labels"]
        width_ratios = [1.0, len(col_labels)]
        inner = outer_gs[row_idx, col_idx + 1].subgridspec(
            1,
            2,
            width_ratios=width_ratios,
            wspace=0.0,
        )
        ax_acc = fig.add_subplot(inner[0, 0])
        ax_cm = fig.add_subplot(inner[0, 1], sharey=ax_acc)

        show_ylabel = col_idx == 0
        show_xlabels = row_idx == n_model_rows - 1
        col_tick_labels = panel["col_tick_labels"]
        xtick_fontsize = None
        if show_xlabels:
            col_tick_labels = [ABBREV_LABELS[label] for label in panel["col_labels"]]
            xtick_fontsize = FONT["grid_xtick"]
        _draw_confusion_matrix_on_axes(
            ax_acc,
            ax_cm,
            cm=panel["cm"],
            cat_acc=panel["cat_acc"],
            row_tick_labels=panel["row_tick_labels"],
            col_tick_labels=col_tick_labels,
            title=panel["title"],
            eval_mode=panel["eval_mode"],
            accuracy=panel["accuracy"],
            n_samples=panel["n_samples"],
            show_ylabel_ticks=show_ylabel,
            show_ylabel_label=show_ylabel,
            show_xticklabels=show_xlabels,
            show_xlabel=show_xlabels,
            show_cat_acc_label=row_idx != 0,
            show_cbar_label=False,
            predicted_note=misclass_highlight_note(panel["eval_mode"]) if show_xlabels else None,
            xtick_fontsize=xtick_fontsize,
            panel_title=_grid_panel_title(panel),
            panel_title_pad=2,
            include_misclass_note=False,
        )

        if row_idx == 0:
            top_row_axes[col_idx] = ax_cm
        if col_idx == 0:
            row_model_labels[row_idx] = panel["model_label"]

    for row_idx, model_label in row_model_labels.items():
        ax_label = fig.add_subplot(outer_gs[row_idx, 0])
        ax_label.axis("off")
        bbox = ax_label.get_position()
        fig.text(
            0.001,
            bbox.y0 + bbox.height / 2,
            model_label,
            rotation=90,
            ha="left",
            va="center",
            fontsize=FONT["grid_col_title"],
        )

    for col_idx, eval_mode in enumerate(("frame", "group_majority_vote")):
        ax = top_row_axes[col_idx]
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.03,
            eval_titles[eval_mode],
            ha="center",
            va="bottom",
            fontsize=FONT["grid_col_title"],
        )

    fig.suptitle(
        "Confusion matrices — EfficientNet-B0 vs Late fusion (preset 7: EN+Whisper)",
        fontsize=FONT["grid_suptitle"],
        y=0.985,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Three panels side by side, drawn at the width of a two-column IEEE figure.
ROW_PANEL_SPECS = [
    ("efficientnet_b0_v15", "frame", "EfficientNet-B0 — frame", "Exact match"),
    ("late_fusion_en_v15", "frame", "Fusion-EN — frame", "Exact match"),
    ("late_fusion_en_v15", "group_any_frame", "Fusion-EN — clip (any-frame)", "Class match"),
]

ROW_FIGSIZE = (7.16, 2.75)

ROW_FONT_SCALE = 1.1

ROW_FONT = {
    key: round(size * ROW_FONT_SCALE, 2)
    for key, size in {
        "title": 5.0,
        "axis_label": 5.0,
        "tick": 4.0,
        "cell": 4.5,
        "cell_highlight": 4.5,
        "cell_highlight_pct": 3.0,
        "cat_acc": 4.5,
        "cbar_label": 4.0,
        "cbar_tick": 3.4,
        "grid_panel_title": 5.0,
    }.items()
}
ROW_FONT["cat_acc"] = round(ROW_FONT["cell"] * 1.18, 2)
ROW_FONT["cbar_tick"] = ROW_FONT["cat_acc"]

# Recall column width, in units of one confusion-matrix column.
ROW_RECALL_WIDTH = 1.6


def _row_panel_title(panel: dict) -> str:
    return (
        f"{panel['panel_label']}\n"
        f"{panel['metric_label']}: {panel['accuracy'] * 100:.1f}% (n={panel['n_samples']})"
    )


def plot_panel_row(panels: list[dict], output_path: Path) -> None:
    fig = plt.figure(figsize=ROW_FIGSIZE)
    outer_gs = GridSpec(
        1,
        len(panels),
        figure=fig,
        left=0.105,
        right=0.962,
        top=0.855,
        bottom=0.175,
        wspace=0.16,
    )

    for col_idx, panel in enumerate(panels):
        col_labels = panel["col_labels"]
        inner = outer_gs[0, col_idx].subgridspec(
            1,
            2,
            width_ratios=[ROW_RECALL_WIDTH, len(col_labels)],
            wspace=0.0,
        )
        ax_acc = fig.add_subplot(inner[0, 0])
        ax_cm = fig.add_subplot(inner[0, 1], sharey=ax_acc)

        show_ylabel = col_idx == 0
        _draw_confusion_matrix_on_axes(
            ax_acc,
            ax_cm,
            cm=panel["cm"],
            cat_acc=panel["cat_acc"],
            row_tick_labels=panel["row_tick_labels"],
            col_tick_labels=[ABBREV_LABELS[label] for label in col_labels],
            title=panel["title"],
            eval_mode=panel["eval_mode"],
            accuracy=panel["accuracy"],
            n_samples=panel["n_samples"],
            show_ylabel_ticks=show_ylabel,
            show_ylabel_label=show_ylabel,
            show_xticklabels=True,
            show_xlabel=False,
            show_cat_acc_label=True,
            show_cbar_label=False,
            xtick_fontsize=ROW_FONT["tick"],
            panel_title=_row_panel_title(panel),
            panel_title_pad=2,
            fonts=ROW_FONT,
            show_highlight_pct=False,
            cell_linewidth=0.4,
            cat_acc_fontweight="normal",
            cell_fontfamily="Liberation Sans Narrow",
            scale_cell_by_digits=True,
            cbar_size="3.6%",
        )
        # Match the rotated class ticks so the recall column header stays clear of them.
        ax_acc.set_xticklabels(
            ["Recall"],
            fontsize=ROW_FONT["tick"],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        # "Predicted" uses the previous left/right placement (right-aligned at
        # axes x=-0.6, next to Recall) but sits one line below the tick band.
        if col_idx == 0:
            ax_acc.annotate(
                "Predicted",
                xy=(-0.6, -0.02),
                xycoords=ax_acc.transAxes,
                xytext=(0, -ROW_FONT["axis_label"] * 1.2),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=ROW_FONT["axis_label"],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrices for best classifiers.")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--groups-csv", type=Path, default=TIMESTAMP_GROUPS_CSV)
    parser.add_argument("--results-dir", type=Path, default=CLASSIFICATION_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR)
    args = parser.parse_args()

    meta_df = pd.read_csv(args.meta_csv)
    frame_group_df = load_frame_group_map(args.meta_csv, args.groups_csv)

    grid_panels: list[dict] = []
    summary_rows: list[dict] = []

    for model_idx, model in enumerate(BEST_MODELS):
        prediction_map = load_prediction_map(
            meta_df,
            base_preset=model["base_preset"],
            segment_version=model["segment_version"],
            results_dir=args.results_dir,
        )

        frame_pairs = collect_frame_pairs(meta_df, prediction_map)
        group_pairs = collect_group_majority_pairs(frame_group_df, prediction_map)
        any_frame_pairs = collect_group_any_frame_pairs(frame_group_df, prediction_map)
        eval_specs = [
            ("frame", "Final prob (frame-level Top-1)", frame_pairs, False, None),
            (
                "group_majority_vote",
                "Group (majority vote)",
                (group_pairs[0], group_pairs[1]),
                True,
                group_pairs[2],
            ),
            (
                "group_any_frame",
                "Group (any frame)",
                (any_frame_pairs[0], any_frame_pairs[1]),
                True,
                any_frame_pairs[2],
            ),
        ]

        for eval_mode, eval_title, pair, include_no_prediction, pipeline_ok in eval_specs:
            y_true, y_pred = pair
            y_true = normalize_labels(y_true, include_no_prediction=include_no_prediction)
            y_pred = normalize_labels(y_pred, include_no_prediction=include_no_prediction)

            png_path = args.output_dir / f"confusion_matrix_{model['slug']}_{eval_mode}.png"
            report_path = args.output_dir / f"confusion_matrix_{model['slug']}_{eval_mode}.txt"
            cm_csv_path = args.output_dir / f"confusion_matrix_{model['slug']}_{eval_mode}.csv"

            labels = ALL_LABELS if include_no_prediction else ALL_LABELS[:-1]
            row_labels = ALL_LABELS[:-1]
            col_labels = labels
            exact_accuracy = accuracy_score(y_true, y_pred)
            pipeline_accuracy = (
                sum(pipeline_ok) / len(pipeline_ok) if pipeline_ok is not None else None
            )
            title = f"{model['label']} — {eval_title}"
            if pipeline_accuracy is not None:
                title += f"\nPipeline class rate: {pipeline_accuracy * 100:.2f}%"
            stats = plot_confusion_matrix(
                y_true,
                y_pred,
                title=title,
                output_path=png_path,
                row_labels=row_labels,
                col_labels=col_labels,
                exact_accuracy=exact_accuracy,
                eval_mode=eval_mode,
            )
            save_report(y_true, y_pred, output_path=report_path, labels=row_labels)

            cm_full = confusion_matrix(y_true, y_pred, labels=col_labels)
            row_indices = [col_labels.index(label) for label in row_labels]
            cm = cm_full[row_indices, :]
            cat_acc = compute_category_accuracy(cm)
            cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)
            cm_df["category_accuracy"] = cat_acc
            cm_df.to_csv(cm_csv_path)

            row_tick_labels = [SHORT_LABELS[label] for label in row_labels]
            col_tick_labels = [SHORT_LABELS[label] for label in col_labels]
            grid_panels.append(
                {
                    "row_idx": model_idx,
                    "col_idx": 0 if eval_mode == "frame" else 1,
                    "eval_mode": eval_mode,
                    "slug": model["slug"],
                    "model_label": model["label"],
                    "title": title,
                    "cm": cm,
                    "cat_acc": cat_acc,
                    "row_labels": row_labels,
                    "col_labels": col_labels,
                    "row_tick_labels": row_tick_labels,
                    "col_tick_labels": col_tick_labels,
                    "accuracy": exact_accuracy,
                    "pipeline_accuracy": pipeline_accuracy,
                    "n_samples": stats["n_samples"],
                }
            )
            summary_rows.append(
                {
                    "model": model["label"],
                    "base_preset": model["base_preset"],
                    "segment_version": model["segment_version"],
                    "eval_mode": eval_mode,
                    "eval_label": eval_title,
                    "exact_match_accuracy_pct": round(exact_accuracy * 100, 2),
                    "pipeline_class_rate_pct": round(pipeline_accuracy * 100, 2)
                    if pipeline_accuracy is not None
                    else round(exact_accuracy * 100, 2),
                    "n_samples": stats["n_samples"],
                    "png_path": str(png_path),
                    "report_path": str(report_path),
                    "csv_path": str(cm_csv_path),
                }
            )

    combined_path = args.output_dir / "confusion_matrix_best_models_grid.png"
    plot_combined_grid(
        [panel for panel in grid_panels if panel["eval_mode"] != "group_any_frame"],
        combined_path,
    )

    panel_lookup = {(panel["slug"], panel["eval_mode"]): panel for panel in grid_panels}
    row_panels = []
    for slug, eval_mode, panel_label, metric_label in ROW_PANEL_SPECS:
        panel = dict(panel_lookup[(slug, eval_mode)])
        panel["panel_label"] = panel_label
        panel["metric_label"] = metric_label
        row_panels.append(panel)
    row_path = args.output_dir / "confusion_matrix_best_models_row.png"
    plot_panel_row(row_panels, row_path)

    summary_path = args.output_dir / "confusion_matrix_best_models_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved combined grid: {combined_path}")
    print(f"Saved three-panel row: {row_path}")


if __name__ == "__main__":
    main()
