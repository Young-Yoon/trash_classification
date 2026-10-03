#!/usr/bin/env python3
"""Create dataset sample figure: video -> frame+GT mask, and 15 category masked crops."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parent
DEMO_DIR = PACKAGE_DIR.parent / "demo"
SHARED_DIR = PACKAGE_DIR / "shared"
CLASSIFY_DIR = PACKAGE_DIR / "classify"
for path in (SHARED_DIR, CLASSIFY_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import DATASET_ROOT, META2_CSV, TAXONOMY_JSON
from dataset_paths import resolve_meta_asset_path
from image_utils import get_bbox_from_mask, load_classification_image


def _binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = grown
    return out


def _ring(mask: np.ndarray, outer: int, inner: int) -> np.ndarray:
    return _binary_dilate(mask, outer) & ~_binary_dilate(mask, inner)


def overlay_mask_on_frame(frame: Image.Image, mask_path: Path, *, alpha: float = 0.15) -> Image.Image:
    del alpha  # fill overlay replaced by a dual-color outline
    mask = Image.open(mask_path).convert("L")
    if mask.size != frame.size:
        mask = mask.resize(frame.size, Image.NEAREST)
    mask_np = np.array(mask) > 0
    rgb = np.array(frame.convert("RGB"), dtype=np.uint8)
    rgb[_ring(mask_np, outer=5, inner=0)] = (120, 230, 70)
    return Image.fromarray(rgb)


def normalize_phrase(phrase: str) -> str:
    return re.sub(r"\s+", " ", str(phrase or "").strip().strip(".").lower())


CURATED_PHRASES: dict[str, list[str]] = {
    "Black Plastic Containers": ["Black plastic", "unnumbered black plastic", "Other plastic"],
    "Containers #1/Clamshells": ["Clamshell", "One clamshell", "Number one clamshell"],
    "Containers #2 Colored": ["Two color", "two colors", "It's light purple"],
    "Containers #2 Natural": ["Two natural", "natural", "It looks like all two natural"],
    "Containers #3-7 (Not #5)": ["Three through seven", "three to seven", "Number 4"],
    "Containers #5/PP": ["Number five", "Five", "Number 5"],
    "Deposit #1 Bottles": ["One deposit", "Deposit number one", "deposit number one"],
    "Garbage Bags": ["non -deposit garbage bag", "foil clamshell garbage bag yeah"],
    "Non-Deposit": ["Non -deposit", "other non -deposit", "Other non -deposited"],
    "Other #1 Bottles": ["One non -deposit", "Non -deposit one", "Non -deposit number one"],
    "Other Plastic Film": ["film", "Film plastic", "Plastic film"],
    "Other Plastics": ["Other plastic", "Do we have other plastic?", "One other plastic"],
    "Other Rigid/Bulky Plastics": ["Bulky rigid"],
    "Other Unnumbered Containers & Fragments": ["Unnumbered", "other unnumbered", "One unnumbered"],
    "Plastic": ["Other plastic", "This is plastic", "plastic"],
}


def ranked_phrases_for_category(meta_df: pd.DataFrame, category: str) -> list[str]:
    curated = CURATED_PHRASES.get(category, [])
    if curated:
        return curated
    sub = meta_df[meta_df["category_name"] == category]
    counts: Counter[str] = Counter()
    display: dict[str, Counter[str]] = {}
    for _, row in sub.iterrows():
        phrase = str(row.get("phrase") or "").strip().strip(".")
        if not phrase:
            continue
        key = normalize_phrase(phrase)
        counts[key] += 1
        display.setdefault(key, Counter())[phrase] += 1
    return [display[key].most_common(1)[0][0] for key, _ in counts.most_common()]


def phrase_rank_for_category_index(category_index: int) -> int:
    # Categories 7, 8, 13, 14, 15 use common phrase 1; others use common phrase 2.
    if (category_index + 1) in {7, 8, 13, 14, 15}:
        return 0
    return 1


def pick_category_sample_by_phrase_rank(
    meta_df: pd.DataFrame,
    category: str,
    *,
    phrase_rank: int,
) -> pd.Series | None:
    ranked = ranked_phrases_for_category(meta_df, category)
    if not ranked:
        return None
    target_norm = normalize_phrase(ranked[min(phrase_rank, len(ranked) - 1)])

    for _, row in meta_df[meta_df["category_name"] == category].iterrows():
        if normalize_phrase(str(row.get("phrase") or "")) != target_norm:
            continue
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
        if frame_path and mask_path and frame_path.is_file() and mask_path.is_file():
            bbox = get_bbox_from_mask(mask_path)
            if bbox and (bbox[2] - bbox[0]) >= 20 and (bbox[3] - bbox[1]) >= 20:
                return row
    return pick_category_sample(meta_df, category, skip=phrase_rank)


def pick_category_sample(meta_df: pd.DataFrame, category: str, *, skip: int = 0) -> pd.Series | None:
    sub = meta_df[meta_df["category_name"] == category]
    matches: list[pd.Series] = []
    for _, row in sub.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
        if frame_path and mask_path and frame_path.is_file() and mask_path.is_file():
            bbox = get_bbox_from_mask(mask_path)
            if bbox and (bbox[2] - bbox[0]) >= 20 and (bbox[3] - bbox[1]) >= 20:
                matches.append(row)
    if not matches:
        return None
    index = min(skip, len(matches) - 1)
    return matches[index]


def fit_into_square(image: Image.Image, size: int = 96) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (24, 24, 24))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def short_category_label(category: str) -> str:
    replacements = {
        "Black Plastic Containers": "Black Plastic",
        "Containers #1/Clamshells": "#1 Clamshell",
        "Containers #2 Colored": "#2 Colored",
        "Containers #2 Natural": "#2 Natural",
        "Containers #3-7 (Not #5)": "#3-7",
        "Containers #5/PP": "#5 PP",
        "Deposit #1 Bottles": "Deposit #1",
        "Other #1 Bottles": "Other #1",
        "Other Plastic Film": "Plastic Film",
        "Other Plastics": "Other Plastics",
        "Other Rigid/Bulky Plastics": "Rigid/Bulky",
        "Other Unnumbered Containers & Fragments": "Unnumbered",
        "Non-Deposit": "Non-Deposit",
        "Garbage Bags": "Garbage Bags",
        "Plastic": "Plastic",
    }
    return replacements.get(category, category)


def text_width_in(text: str, fontsize: float, *, bold: bool = False) -> float:
    """Width of a rendered string in inches."""
    if not text:
        return 0.0
    prop = FontProperties(weight="bold" if bold else "normal")
    path = TextPath((0, 0), text, size=fontsize, prop=prop)
    return path.get_extents().width / 72.0


def wrap_to_width(text: str, fontsize: float, max_w_in: float, *, max_lines: int = 2) -> list[str]:
    """Greedy word wrap that measures each candidate line instead of counting characters."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or text_width_in(candidate, fontsize) <= max_w_in:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


def fit_fontsize(labels: list[str], max_w_in: float, start: float, *, bold: bool = False) -> float:
    """Largest font size at or below `start` that keeps every label inside one cell.

    Labels may wrap, so only the longest single word has to fit.
    """
    words = [word for label in labels for word in label.split()] or [""]
    size = start
    while size > 4.0:
        if all(text_width_in(word, size, bold=bold) <= max_w_in for word in words):
            break
        size -= 0.25
    return size


def wrap_phrase_lines(phrase: str, max_chars: int, *, max_lines: int = 2) -> list[str]:
    words = phrase.strip().strip(".").split()
    if not words:
        return ['""']

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(word[:max_chars])
            current = ""
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) == 1:
        return [f'"{lines[0]}"']
    return [f'"{lines[0]}', f'{lines[1]}"']


def phrase_label_text(phrase: str, cell_w: float, fig_w_in: float, *, fontsize: float = 5.5) -> str:
    phrase = phrase.strip().strip(".").strip('"')
    lines = wrap_to_width(phrase, fontsize, cell_w * fig_w_in * 0.96, max_lines=2)
    if not lines:
        return '""'
    if len(lines) == 1:
        return f'"{lines[0]}"'
    return f'"{lines[0]}\n{lines[1]}"'


CATEGORY_LABEL_COLOR = "#1565C0"


def draw_cell_label(
    fig: plt.Figure,
    x: float,
    y_top: float,
    label_h: float,
    phrase: str,
    category: str,
    cell_w: float,
    fig_w_in: float,
    *,
    fontsize: float = 5.5,
    linespacing: float = 0.95,
) -> None:
    label_bottom = y_top - label_h
    max_w_in = cell_w * fig_w_in * 0.96
    phrase_text = phrase_label_text(phrase, cell_w, fig_w_in, fontsize=fontsize)
    if phrase_text:
        fig.text(
            x,
            y_top - 0.001,
            phrase_text,
            ha="center",
            va="top",
            fontsize=fontsize,
            color="#222222",
            linespacing=linespacing,
            clip_on=False,
        )
    category_lines = wrap_to_width(
        short_category_label(category), fontsize, max_w_in, max_lines=2
    )
    fig.text(
        x,
        label_bottom + 0.001,
        "\n".join(category_lines),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=CATEGORY_LABEL_COLOR,
        fontweight="normal",
        linespacing=linespacing,
        clip_on=False,
    )


def draw_subtitle_run(
    fig: plt.Figure,
    x_center: float,
    y: float,
    segments: list[tuple[str, str]],
    *,
    fontsize: float,
    fig_w_in: float,
    min_x: float,
) -> None:
    """Draw colored segments as one centered line, laid out by measured width."""
    widths = [text_width_in(text, fontsize, bold=False) / fig_w_in for text, _ in segments]
    x = max(min_x, x_center - sum(widths) / 2)
    for (text, color), width in zip(segments, widths):
        fig.text(
            x,
            y,
            text,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight="normal",
            color=color,
        )
        x += width


def mask_area_ratio(frame_path: Path, mask_path: Path) -> float | None:
    frame = Image.open(frame_path)
    mask = Image.open(mask_path).convert("L")
    if mask.size != frame.size:
        mask = mask.resize(frame.size, Image.NEAREST)
    mask_np = np.array(mask) > 0
    if not mask_np.any():
        return None
    return float(mask_np.sum() / mask_np.size)


def pick_diverse_rep_samples(meta_df: pd.DataFrame, *, n: int = 5) -> list[pd.Series]:
    """Pick representative frame+mask rows from different gx sequences."""
    target_ratio = 0.055
    max_ratio = 0.10
    min_ratio = 0.015
    by_gx: dict[str, list[tuple[float, pd.Series]]] = {}

    for _, row in meta_df.iterrows():
        gx_id = str(row.get("gx_id") or "")
        if not gx_id:
            continue
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
        if not frame_path or not mask_path or not frame_path.is_file() or not mask_path.is_file():
            continue
        bbox = get_bbox_from_mask(mask_path)
        if not bbox:
            continue
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        if bbox_w < 40 or bbox_h < 40:
            continue
        ratio = mask_area_ratio(frame_path, mask_path)
        if ratio is None or ratio < min_ratio or ratio > max_ratio:
            continue
        score = abs(ratio - target_ratio)
        by_gx.setdefault(gx_id, []).append((score, row))

    picked: list[pd.Series] = []
    for gx_id in sorted(by_gx):
        best = min(by_gx[gx_id], key=lambda item: item[0])[1]
        picked.append(best)
        if len(picked) >= n:
            break
    return picked


def build_category_items(
    meta_df: pd.DataFrame,
    categories: list[str],
) -> list[tuple[int, str, str, Image.Image | None]]:
    category_items: list[tuple[int, str, str, Image.Image | None]] = []
    for idx, category in enumerate(categories):
        phrase_rank = phrase_rank_for_category_index(idx)
        row = pick_category_sample_by_phrase_rank(meta_df, category, phrase_rank=phrase_rank)
        if row is None:
            category_items.append((idx, category, category, None))
            continue
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))
        crop = load_classification_image(frame_path, mask_path, crop_mode="masked")
        phrase = str(row.get("phrase") or category).strip().strip(".")
        category_items.append((idx, category, phrase, fit_into_square(crop)))
    return category_items


def render_pipeline_figure(
    rep_row: pd.Series,
    category_items: list[tuple[int, str, str, Image.Image | None]],
    output_path: Path,
    *,
    label_fontsize: float = 5.5,
    subtitle_fontsize: float = 7.5,
    fig_w_in: float = 12.4,
    fig_h_in: float = 4.15,
    left_panel_max_w: float = 1.0,
) -> tuple[int, float]:
    rep_frame_path = resolve_meta_asset_path(rep_row.get("copied_frame_path"))
    rep_mask_path = resolve_meta_asset_path(rep_row.get("copied_mask_path"))
    assert rep_frame_path and rep_mask_path

    rep_frame = Image.open(rep_frame_path).convert("RGB")
    rep_overlay = overlay_mask_on_frame(rep_frame, rep_mask_path)

    n_cats = sum(1 for item in category_items if item[3] is not None)
    grid_cols = 5
    grid_rows = 3

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")

    content_y0 = 0.018
    content_y1 = 0.875
    right_edge = 0.992
    left_edge = 0.004
    panel_gap = 0.006
    col_gap = 0.0015
    row_gap = 0.010
    label_linespacing = 1.12

    img_aspect = rep_overlay.width / rep_overlay.height
    content_h = content_y1 - content_y0
    left_panel_w = content_h * img_aspect * (fig_h_in / fig_w_in)
    left_panel_h = content_h
    if left_panel_w > left_panel_max_w:
        left_panel_w = left_panel_max_w
        left_panel_h = left_panel_w / img_aspect * (fig_w_in / fig_h_in)
    left_panel_y0 = content_y0 + (content_h - left_panel_h) / 2

    ax_left = fig.add_axes([left_edge, left_panel_y0, left_panel_w, left_panel_h])
    ax_left.imshow(rep_overlay, aspect="equal")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_xmargin(0)
    ax_left.set_ymargin(0)
    for spine in ax_left.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#bbbbbb")

    grid_left = left_edge + left_panel_w + panel_gap
    grid_width = right_edge - grid_left

    # Labels are capped by cell width, so shrink the font until every category name
    # fits, then recompute the cells for the label block that font needs.
    cat_labels = [short_category_label(item[1]) for item in category_items]
    phrases = [item[2] for item in category_items]
    label_fs = label_fontsize
    label_h = 0.058
    cell_w = 0.0
    label_lines = 3.0
    for _ in range(5):
        label_h = max(0.058, (label_lines + 0.3) * label_fs * label_linespacing / 72.0 / fig_h_in)
        cell_h_max = (content_h - grid_rows * label_h - (grid_rows - 1) * row_gap) / grid_rows
        cell_w = min(
            (grid_width - (grid_cols - 1) * col_gap) / grid_cols,
            cell_h_max * (fig_h_in / fig_w_in),
        )
        max_w_in = cell_w * fig_w_in * 0.96
        fitted = fit_fontsize(cat_labels, max_w_in, label_fontsize)
        needed = max(
            len(wrap_to_width(phrase, fitted, max_w_in, max_lines=2))
            + len(wrap_to_width(cat, fitted, max_w_in, max_lines=2))
            for phrase, cat in zip(phrases, cat_labels)
        )
        if abs(fitted - label_fs) < 0.2 and needed == label_lines:
            break
        label_fs, label_lines = fitted, needed
    cell_h = cell_w * (fig_w_in / fig_h_in)
    row_pitch = label_h + cell_h + row_gap

    # Height-limited cells leave a gap on the right; give it back to the frame panel.
    leftover = grid_width - (grid_cols * cell_w + (grid_cols - 1) * col_gap)
    if leftover > 0.005:
        grid_left += leftover
        grid_width -= leftover
        panel_h = min(
            content_h,
            (left_panel_w + leftover) / img_aspect * (fig_w_in / fig_h_in),
        )
        panel_w = panel_h * img_aspect * (fig_h_in / fig_w_in)
        ax_left.set_position(
            [left_edge, content_y0 + (content_h - panel_h) / 2, panel_w, panel_h]
        )

    for idx, category, phrase, crop_img in category_items:
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        x0 = grid_left + col_idx * (cell_w + col_gap)
        y_top = content_y1 - row_idx * row_pitch
        y0 = y_top - label_h - cell_h
        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        if crop_img is None:
            ax.set_facecolor("#181818")
        else:
            ax.imshow(crop_img, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")
        draw_cell_label(
            fig,
            x0 + cell_w / 2,
            y_top,
            label_h,
            phrase,
            category,
            cell_w,
            fig_w_in,
            fontsize=label_fs,
            linespacing=label_linespacing,
        )

    left_pos = ax_left.get_position()
    subtitle_y = content_y1 + 0.012
    gx_id = str(rep_row["gx_id"])
    cat_label = short_category_label(str(rep_row["category_name"]))
    draw_subtitle_run(
        fig,
        left_pos.x0 + left_pos.width / 2,
        subtitle_y,
        [
            (f"Frame + GT mask · {gx_id} · ", "#000000"),
            (cat_label, CATEGORY_LABEL_COLOR),
        ],
        fontsize=subtitle_fontsize,
        fig_w_in=fig_w_in,
        min_x=left_edge,
    )
    draw_subtitle_run(
        fig,
        grid_left + grid_width / 2,
        subtitle_y,
        [
            ("common phrase · ", "#333333"),
            ("category_name", CATEGORY_LABEL_COLOR),
            (" + GT masked crop", "#333333"),
        ],
        fontsize=subtitle_fontsize,
        fig_w_in=fig_w_in,
        min_x=grid_left,
    )

    divider_x = left_pos.x1 + panel_gap / 2
    fig.add_artist(
        plt.Line2D(
            [divider_x, divider_x],
            [content_y0, content_y1 + 0.012],
            transform=fig.transFigure,
            color="#cccccc",
            linewidth=1.0,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
    return n_cats, label_fs


def measured_text_width_in(fig: plt.Figure, text: str, fontsize: float, *, bold: bool = False) -> float:
    """Width of a rendered string in inches, measured against the figure's renderer."""
    if not text:
        return 0.0
    artist = fig.text(
        0.0, 0.0, text, fontsize=fontsize, fontweight="bold" if bold else "normal"
    )
    width_px = artist.get_window_extent(renderer=fig.canvas.get_renderer()).width
    artist.remove()
    return width_px / fig.dpi


def split_word_pieces(
    fig: plt.Figure,
    word: str,
    fontsize: float,
    max_w_in: float,
    *,
    bold: bool = False,
) -> list[str]:
    """Cut a word too wide for one cell, preferring hyphen and slash boundaries."""
    if measured_text_width_in(fig, word, fontsize, bold=bold) <= max_w_in:
        return [word]
    chunks = [chunk for chunk in re.split(r"(?<=[-/])", word) if chunk]
    if len(chunks) > 1:
        pieces: list[str] = []
        for chunk in chunks:
            pieces.extend(split_word_pieces(fig, chunk, fontsize, max_w_in, bold=bold))
        return pieces
    pieces = []
    rest = word
    while len(rest) > 2 and measured_text_width_in(fig, rest, fontsize, bold=bold) > max_w_in:
        cut = len(rest) - 1
        while cut > 2 and measured_text_width_in(fig, f"{rest[:cut]}-", fontsize, bold=bold) > max_w_in:
            cut -= 1
        pieces.append(f"{rest[:cut]}-")
        rest = rest[cut:]
    pieces.append(rest)
    return pieces


def wrap_measured(
    fig: plt.Figure,
    text: str,
    fontsize: float,
    max_w_in: float,
    *,
    bold: bool = False,
    max_lines: int = 2,
) -> list[str]:
    """Greedy word wrap measured against the cell width; long words are broken, not clipped."""
    tokens: list[tuple[str, str]] = []
    for word in text.split():
        pieces = split_word_pieces(fig, word, fontsize, max_w_in, bold=bold)
        tokens.append((pieces[0], " " if tokens else ""))
        tokens.extend((piece, "") for piece in pieces[1:])
    if not tokens:
        return []

    lines: list[str] = []
    current = ""
    for piece, join in tokens:
        candidate = f"{current}{join}{piece}" if current else piece
        if not current or measured_text_width_in(fig, candidate, fontsize, bold=bold) <= max_w_in:
            current = candidate
            continue
        lines.append(current)
        current = piece
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


def draw_centered_run(
    fig: plt.Figure,
    x_center: float,
    y: float,
    segments: list[tuple[str, str]],
    *,
    fontsize: float,
    fig_w_in: float,
) -> None:
    widths = [measured_text_width_in(fig, text, fontsize, bold=False) / fig_w_in for text, _ in segments]
    x = max(0.0, x_center - sum(widths) / 2)
    for (text, color), width in zip(segments, widths):
        fig.text(
            x,
            y,
            text,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight="normal",
            color=color,
        )
        x += width


def render_pipeline_figure_stacked(
    rep_row: pd.Series,
    category_items: list[tuple[int, str, str, Image.Image | None]],
    output_path: Path,
    *,
    label_fontsize: float = 5.5,
    subtitle_fontsize: float = 7.5,
    fig_w_in: float = 3.4,
    grid_cols: int = 8,
) -> tuple[int, float]:
    """Single-column figure: frame + GT mask on top, category crops in a grid below."""
    rep_frame_path = resolve_meta_asset_path(rep_row.get("copied_frame_path"))
    rep_mask_path = resolve_meta_asset_path(rep_row.get("copied_mask_path"))
    assert rep_frame_path and rep_mask_path

    rep_frame = Image.open(rep_frame_path).convert("RGB")
    rep_overlay = overlay_mask_on_frame(rep_frame, rep_mask_path)
    img_aspect = rep_overlay.width / rep_overlay.height

    n_cats = sum(1 for item in category_items if item[3] is not None)
    grid_rows = -(-len(category_items) // grid_cols)

    margin_x = 0.02
    pad_top = 0.015
    pad_bottom = 0.02
    col_gap = 0.012
    row_gap = 0.030
    panel_gap = 0.055
    label_linespacing = 1.12
    subtitle_h = subtitle_fontsize * 1.5 / 72.0

    content_w = fig_w_in - 2 * margin_x
    panel_h = content_w / img_aspect
    cell_w = (content_w - (grid_cols - 1) * col_gap) / grid_cols
    cell_h = cell_w
    cat_max_w = cell_w + col_gap * 0.9

    probe = plt.figure(figsize=(fig_w_in, 1.0))
    label_lines = 1
    for _, category, _, _ in category_items:
        cat_lines = wrap_measured(
            probe, short_category_label(category), label_fontsize, cat_max_w, bold=False, max_lines=3
        )
        label_lines = max(label_lines, len(cat_lines))
    plt.close(probe)
    label_h = (label_lines + 0.25) * label_fontsize * label_linespacing / 72.0

    fig_h_in = (
        pad_top
        + subtitle_h
        + panel_h
        + panel_gap
        + subtitle_h
        + grid_rows * (label_h + cell_h)
        + (grid_rows - 1) * row_gap
        + pad_bottom
    )

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")

    def frac_y(y_in: float) -> float:
        return 1.0 - y_in / fig_h_in

    cursor = pad_top
    draw_centered_run(
        fig,
        0.5,
        frac_y(cursor + subtitle_h) + 0.2 * subtitle_h / fig_h_in,
        [
            (f"Frame + GT mask · {rep_row['gx_id']} · ", "#000000"),
            (short_category_label(str(rep_row["category_name"])), CATEGORY_LABEL_COLOR),
        ],
        fontsize=subtitle_fontsize,
        fig_w_in=fig_w_in,
    )
    cursor += subtitle_h

    ax_top = fig.add_axes(
        [margin_x / fig_w_in, frac_y(cursor + panel_h), content_w / fig_w_in, panel_h / fig_h_in]
    )
    ax_top.imshow(rep_overlay, aspect="equal")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_xmargin(0)
    ax_top.set_ymargin(0)
    for spine in ax_top.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#bbbbbb")
    cursor += panel_h + panel_gap

    draw_centered_run(
        fig,
        0.5,
        frac_y(cursor + subtitle_h) + 0.2 * subtitle_h / fig_h_in,
        [
            ("category_name", CATEGORY_LABEL_COLOR),
            (" + GT masked crop", "#333333"),
        ],
        fontsize=subtitle_fontsize,
        fig_w_in=fig_w_in,
    )
    cursor += subtitle_h

    grid_top = cursor
    row_pitch = label_h + cell_h + row_gap
    for idx, category, _phrase, crop_img in category_items:
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        x0 = margin_x + col_idx * (cell_w + col_gap)
        y_cell_top = grid_top + row_idx * row_pitch
        ax = fig.add_axes(
            [x0 / fig_w_in, frac_y(y_cell_top + cell_h), cell_w / fig_w_in, cell_h / fig_h_in]
        )
        if crop_img is None:
            ax.set_facecolor("#181818")
        else:
            ax.imshow(crop_img, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

        cat_lines = wrap_measured(
            fig, short_category_label(category), label_fontsize, cat_max_w, bold=False, max_lines=3
        )
        fig.text(
            (x0 + cell_w / 2) / fig_w_in,
            frac_y(y_cell_top + cell_h + 0.006),
            "\n".join(cat_lines),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            color=CATEGORY_LABEL_COLOR,
            fontweight="normal",
            linespacing=label_linespacing,
            clip_on=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, facecolor="white")
    plt.close(fig)
    return n_cats, label_fontsize


def render_category_crops_figure(
    category_items: list[tuple[int, str, str, Image.Image | None]],
    output_path: Path,
    *,
    label_fontsize: float = 6.05,
    fig_w_in: float = 3.4,
    grid_cols: int = 8,
) -> tuple[int, float]:
    """8x2 crop grid used as Fig. 1(b)."""
    n_cats = sum(1 for item in category_items if item[3] is not None)
    grid_rows = -(-len(category_items) // grid_cols)

    margin_x = 0.02
    pad_top = 0.01
    pad_bottom = 0.02
    col_gap = 0.012
    row_gap = 0.030
    label_linespacing = 1.12

    content_w = fig_w_in - 2 * margin_x
    cell_w = (content_w - (grid_cols - 1) * col_gap) / grid_cols
    cell_h = cell_w
    cat_max_w = cell_w + col_gap * 0.9

    probe = plt.figure(figsize=(fig_w_in, 1.0))
    label_lines = 1
    for _, category, _, _ in category_items:
        cat_lines = wrap_measured(
            probe, short_category_label(category), label_fontsize, cat_max_w, bold=False, max_lines=2
        )
        label_lines = max(label_lines, len(cat_lines))
    plt.close(probe)
    label_h = (label_lines + 0.25) * label_fontsize * label_linespacing / 72.0

    fig_h_in = (
        pad_top
        + grid_rows * (label_h + cell_h)
        + (grid_rows - 1) * row_gap
        + pad_bottom
    )
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")

    def frac_y(y_in: float) -> float:
        return 1.0 - y_in / fig_h_in

    row_pitch = label_h + cell_h + row_gap
    for idx, category, _phrase, crop_img in category_items:
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        x0 = margin_x + col_idx * (cell_w + col_gap)
        y_cell_top = pad_top + row_idx * row_pitch
        ax = fig.add_axes(
            [x0 / fig_w_in, frac_y(y_cell_top + cell_h), cell_w / fig_w_in, cell_h / fig_h_in]
        )
        if crop_img is None:
            ax.set_facecolor("#181818")
        else:
            ax.imshow(crop_img, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

        cat_lines = wrap_measured(
            fig, short_category_label(category), label_fontsize, cat_max_w, bold=False, max_lines=2
        )
        fig.text(
            (x0 + cell_w / 2) / fig_w_in,
            frac_y(y_cell_top + cell_h + 0.006),
            "\n".join(cat_lines),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            color=CATEGORY_LABEL_COLOR,
            fontweight="normal",
            linespacing=label_linespacing,
            clip_on=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, facecolor="white")
    plt.close(fig)
    return n_cats, label_fontsize


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset sample pipeline figures.")
    parser.add_argument("--variants", type=int, default=5, help="Number of sample variants to generate.")
    parser.add_argument(
        "--layout",
        choices=("stacked", "wide"),
        default="stacked",
        help="stacked: single-column frame on top of an 8x2 crop grid; wide: two-column side-by-side.",
    )
    args = parser.parse_args()

    meta_df = pd.read_csv(META2_CSV)
    with TAXONOMY_JSON.open() as handle:
        taxonomy = json.load(handle)
    categories = [item["name"] for item in taxonomy["categories"]]

    rep_rows = pick_diverse_rep_samples(meta_df, n=args.variants)
    if len(rep_rows) < args.variants:
        raise RuntimeError(f"Need {args.variants} representative samples, found {len(rep_rows)}")

    category_items = build_category_items(meta_df, categories)

    for variant_idx, rep_row in enumerate(rep_rows, start=1):
        if variant_idx == 1:
            output_path = DEMO_DIR / "dataset_sample_pipeline.png"
        else:
            output_path = DEMO_DIR / f"dataset_sample_pipeline_v{variant_idx}.png"
        renderer = (
            render_pipeline_figure_stacked if args.layout == "stacked" else render_pipeline_figure
        )
        n_cats, _ = renderer(
            rep_row,
            category_items,
            output_path,
        )
        phrase = str(rep_row.get("phrase") or "").strip()
        print(
            f"Saved: {output_path} | rep={rep_row['gx_id']} "
            f"({short_category_label(str(rep_row['category_name']))}, \"{phrase}\") | "
            f"categories={n_cats}/{len(categories)}"
        )


if __name__ == "__main__":
    main()
