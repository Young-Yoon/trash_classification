#!/usr/bin/env python3
"""Draw detection + classification pipeline stage diagram.

Note: paper/fig/pipeline_detection_classification_diagram.png is exported from
poster.pptx (model_diagram.png). Do not overwrite it from this script; outputs
belong under demo/ only unless explicitly requested for local experiments.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import PROJECT_ROOT

DEMO_DIR = PROJECT_ROOT / "demo"

PHASE_COLORS = {
    "Input": "#ECECEC",
    "Audio": "#E8A87C",
    "AudioPre": "#F0BC96",
    "AudioTrans": "#D98850",
    "AudioMap": "#C07040",
    "Detection": "#4C72B0",
    "MaskSelect": "#55A868",
    "Crop": "#CCB974",
    "Classification": "#64B5CD",
    "Evaluation": "#937860",
}

VISION_STAGES = [
    {
        "title": "Input",
        "phase": "Input",
        "items": ["Camera frame (JPG)", "GT mask & label", "Timestamp group meta"],
        "artifact": "camera1_plastics_meta2.csv",
    },
    {
        "title": "1. Detection",
        "phase": "Detection",
        "items": ["SAM3 segmentation", "Hand ROI or full image", "Top-K candidates"],
        "artifact": "segmentation results",
    },
    {
        "title": "2. Mask Select",
        "phase": "MaskSelect",
        "items": [
            "Ph.1: detection baselines",
            "Ph.2: mask selection",
            "Ph.3: rerank pool",
            "Ph.4: margin + speech rerank",
        ],
        "artifact": "predicted_masks/",
    },
    {
        "title": "3. Crop",
        "phase": "Crop",
        "items": ["BBox crop (default)", "Optional masked crop", "15% bbox padding"],
        "artifact": "crop image tensor",
    },
    {
        "title": "4. Classification",
        "phase": "Classification",
        "items": [
            "CLIP / SigLIP / EN-B0",
            "Late fusion (image+audio)",
            "Rerank best candidate",
        ],
        "artifact": "classification JSON",
    },
    {
        "title": "5. Evaluation",
        "phase": "Evaluation",
        "items": ["Mask/bbox IoU", "Top-1 accuracy", "Frame & group metrics"],
        "artifact": "pipeline_end_to_end_*.csv",
    },
]

AUDIO_STAGES = [
    {
        "title": "Audio clip",
        "phase": "Audio",
        "items": ["Timestamp mp4/wav", "From meta2 clip path", "Per group segment"],
        "artifact": "audio_video_clips/",
    },
    {
        "title": "Preprocess",
        "phase": "AudioPre",
        "items": ["Extract wav", "Denoise (optional)", "Normalize volume"],
        "artifact": "audio/cleaned/",
    },
    {
        "title": "Transcribe",
        "phase": "AudioTrans",
        "items": ["Whisper LoRA", "Fine-tuned on phrases", "Save transcript text"],
        "artifact": "transcribe/lora/*.txt",
    },
    {
        "title": "Text → Category",
        "phase": "AudioMap",
        "items": ["Match common phrases", "Map to plastic class", "Feed late fusion"],
        "artifact": "category_common_phrases",
    },
]

ITEM_WRAP = 21
ARTIFACT_WRAP = 22
TITLE_FONTSIZE = 9.8
ITEM_FONTSIZE = 7.2
ARTIFACT_FONTSIZE = 6.4
LINE_H = 0.27
PAD_X = 0.11
PAD_TOP = 0.36
PAD_BOTTOM = 0.22
ARTIFACT_GAP = 0.14


def wrap_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(text.replace("\n", " "), width=width, break_long_words=False) or [""]


def box_height_for_stage(stage: dict) -> float:
    title_h = len(wrap_lines(stage["title"], 16)) * LINE_H
    item_h = sum(len(wrap_lines(item, ITEM_WRAP)) for item in stage["items"]) * (LINE_H - 0.02)
    artifact_h = len(wrap_lines(stage["artifact"], ARTIFACT_WRAP)) * (LINE_H - 0.04)
    return PAD_TOP + title_h + 0.08 + item_h + ARTIFACT_GAP + artifact_h + PAD_BOTTOM


def row_height(stages: list[dict]) -> float:
    return max(box_height_for_stage(stage) for stage in stages)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    items: list[str],
    artifact: str,
    facecolor: str,
) -> tuple[float, float, float, float]:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#333333",
        facecolor=facecolor,
        alpha=0.93,
        zorder=2,
    )
    ax.add_patch(box)

    artifact_lines = wrap_lines(artifact, ARTIFACT_WRAP)
    artifact_block_h = len(artifact_lines) * (LINE_H - 0.04)
    artifact_top = y + PAD_BOTTOM + artifact_block_h

    ay = y + PAD_BOTTOM
    for line in artifact_lines:
        ax.text(
            x + width / 2,
            ay,
            line,
            ha="center",
            va="bottom",
            fontsize=ARTIFACT_FONTSIZE,
            color="#444444",
            fontstyle="italic",
            zorder=3,
        )
        ay += LINE_H - 0.04

    cursor_y = y + height - PAD_TOP
    for line in wrap_lines(title, 16):
        ax.text(
            x + width / 2,
            cursor_y,
            line,
            ha="center",
            va="top",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            color="#1a1a1a",
            zorder=3,
        )
        cursor_y -= LINE_H

    cursor_y -= 0.06
    item_floor = artifact_top + ARTIFACT_GAP
    for item in items:
        for line in wrap_lines(item, ITEM_WRAP):
            if cursor_y - (LINE_H - 0.02) < item_floor:
                break
            ax.text(
                x + PAD_X,
                cursor_y,
                f"• {line}",
                ha="left",
                va="top",
                fontsize=ITEM_FONTSIZE,
                color="#222222",
                zorder=3,
            )
            cursor_y -= LINE_H - 0.02

    return (x, y, width, height)


def draw_arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    dashed: bool = False,
    color: str | None = None,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color or ("#555555" if not dashed else "#C06030"),
            linestyle="--" if dashed else "-",
            zorder=1,
        )
    )


def draw_row(
    ax: plt.Axes,
    stages: list[dict],
    *,
    start_x: float,
    y: float,
    box_w: float,
    box_h: float,
    gap: float,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for idx, stage in enumerate(stages):
        x = start_x + idx * (box_w + gap)
        box = draw_box(
            ax,
            x,
            y,
            box_w,
            box_h,
            title=stage["title"],
            items=stage["items"],
            artifact=stage["artifact"],
            facecolor=PHASE_COLORS.get(stage["phase"], "#DDDDDD"),
        )
        boxes.append(box)
        if idx > 0:
            prev = boxes[idx - 1]
            draw_arrow(
                ax,
                prev[0] + prev[2] + 0.02,
                y + box_h / 2,
                x - 0.02,
                y + box_h / 2,
            )
    return boxes




SIMPLE_VISION_STAGES = [
    {"title": "Input", "items": ["Egocentric frame", "Ground-truth mask", "Clip-level label"]},
    {"title": "Detection", "items": ["SAM3 segmentation", "Hand ROI or full image", "Top-K candidates"]},
    {"title": "Mask selection", "items": ["Four-phase reranking", "Candidate pool expansion", "Speech-aligned rerank"]},
    {"title": "Crop", "items": ["Bounding-box crop", "Optional masked crop", "Resize for classifier"]},
    {"title": "Classification", "items": ["CLIP / SigLIP / EffNet", "15-way softmax", "Late fusion with audio"]},
    {"title": "Evaluation", "items": ["Mask IoU", "Frame Top-1", "Group any-frame accuracy"]},
]

SIMPLE_AUDIO_STAGES = [
    {"title": "Audio clip", "items": ["Synchronized speech", "Per deposit event", "Worker phrase"]},
    {"title": "Transcription", "items": ["Whisper large-v2", "LoRA fine-tuning", "Text hypothesis"]},
    {"title": "Phrase mapping", "items": ["Sentence-transformer retrieval", "Common-phrase lexicon", "Plastic category"]},
]


def _box_height_simple(stage: dict) -> float:
    return PAD_TOP + LINE_H + 0.06 + len(stage["items"]) * (LINE_H - 0.02) + PAD_BOTTOM


def _draw_simple_box(ax, x, y, width, height, *, title, items, fill):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="black", facecolor=fill, zorder=2,
    )
    ax.add_patch(box)
    cursor_y = y + height - PAD_TOP
    ax.text(x + width / 2, cursor_y, title, ha="center", va="top",
            fontsize=TITLE_FONTSIZE, fontweight="bold", color="black", zorder=3)
    cursor_y -= LINE_H + 0.04
    for item in items:
        ax.text(x + PAD_X, cursor_y, f"- {item}", ha="left", va="top",
                fontsize=ITEM_FONTSIZE, color="#222222", zorder=3)
        cursor_y -= LINE_H - 0.02
    return (x, y, width, height)


def plot_pipeline_diagram_simple(output_path: Path) -> None:
    box_w, gap, start_x, row_gap = 2.05, 0.38, 0.55, 0.95
    vision_h = max(_box_height_simple(s) for s in SIMPLE_VISION_STAGES)
    audio_h = max(_box_height_simple(s) for s in SIMPLE_AUDIO_STAGES)
    y_audio, y_vision = 0.55, 0.55 + audio_h + row_gap
    n_vision = len(SIMPLE_VISION_STAGES)
    fig_w = start_x + 0.3 + n_vision * box_w + (n_vision - 1) * gap + 0.3
    fig_h = y_vision + vision_h + 0.55

    fig, ax = plt.subplots(figsize=(fig_w * 1.02, fig_h * 1.02))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")
    ax.text(0.1, y_vision + vision_h / 2, "Vision", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="black")
    ax.text(0.1, y_audio + audio_h / 2, "Audio", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="black")

    vision_boxes = []
    for idx, stage in enumerate(SIMPLE_VISION_STAGES):
        x = start_x + idx * (box_w + gap)
        fill = "white" if idx % 2 == 0 else "#F2F2F2"
        box = _draw_simple_box(ax, x, y_vision, box_w, vision_h, title=stage["title"], items=stage["items"], fill=fill)
        vision_boxes.append(box)
        if idx:
            prev = vision_boxes[idx - 1]
            draw_arrow(ax, prev[0]+prev[2]+0.02, y_vision+vision_h/2, x-0.02, y_vision+vision_h/2, color="#333333")

    audio_boxes = []
    for idx, stage in enumerate(SIMPLE_AUDIO_STAGES):
        x = start_x + idx * (box_w + gap)
        fill = "white" if idx % 2 == 0 else "#F2F2F2"
        box = _draw_simple_box(ax, x, y_audio, box_w, audio_h, title=stage["title"], items=stage["items"], fill=fill)
        audio_boxes.append(box)
        if idx:
            prev = audio_boxes[idx - 1]
            draw_arrow(ax, prev[0]+prev[2]+0.02, y_audio+audio_h/2, x-0.02, y_audio+audio_h/2, color="#333333")

    input_box, audio_first = vision_boxes[0], audio_boxes[0]
    draw_arrow(ax, input_box[0]+input_box[2]/2, input_box[1]-0.02,
               audio_first[0]+audio_first[2]/2, audio_first[1]+audio_first[3]+0.02,
               dashed=True, color="#333333")

    cls_box, audio_last = vision_boxes[4], audio_boxes[-1]
    bridge_y = y_audio + audio_h + row_gap * 0.48
    draw_arrow(ax, audio_last[0]+audio_last[2]/2, audio_last[1]+audio_last[3]+0.02,
               audio_last[0]+audio_last[2]/2, bridge_y, dashed=True, color="#333333")
    draw_arrow(ax, audio_last[0]+audio_last[2]/2, bridge_y,
               cls_box[0]+cls_box[2]/2, bridge_y, dashed=True, color="#333333")
    draw_arrow(ax, cls_box[0]+cls_box[2]/2, bridge_y,
               cls_box[0]+cls_box[2]/2, cls_box[1]-0.02, dashed=True, color="#333333")
    ax.text((audio_last[0]+cls_box[0]+audio_last[2]+cls_box[2])/4, bridge_y+0.1,
            "late fusion", ha="center", va="bottom", fontsize=7.5, color="#333333", fontstyle="italic")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)

def plot_pipeline_diagram(output_path: Path) -> None:
    box_w = 2.15
    gap = 0.45
    start_x = 0.62
    row_gap = 1.05

    vision_h = row_height(VISION_STAGES)
    audio_h = row_height(AUDIO_STAGES)

    bottom_margin = 0.75
    y_audio = bottom_margin
    y_vision = y_audio + audio_h + row_gap

    top_margin = 0.85
    fig_h = y_vision + vision_h + top_margin
    n_vision = len(VISION_STAGES)
    fig_w = start_x + 0.35 + n_vision * box_w + (n_vision - 1) * gap + 0.35

    fig, ax = plt.subplots(figsize=(fig_w * 1.05, fig_h * 1.05))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(
        fig_w / 2,
        fig_h - 0.42,
        "Camera1 Plastics — Detection & Classification Pipeline",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(0.12, y_vision + vision_h / 2, "Vision", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333")
    ax.text(0.12, y_audio + audio_h / 2, "Audio", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#A04818")

    vision_boxes = draw_row(
        ax, VISION_STAGES, start_x=start_x, y=y_vision, box_w=box_w, box_h=vision_h, gap=gap
    )
    audio_boxes = draw_row(
        ax, AUDIO_STAGES, start_x=start_x, y=y_audio, box_w=box_w, box_h=audio_h, gap=gap
    )

    input_box = vision_boxes[0]
    audio_first = audio_boxes[0]
    mid_y = (input_box[1] + audio_first[1] + audio_first[3]) / 2
    draw_arrow(
        ax,
        input_box[0] + input_box[2] / 2,
        input_box[1] - 0.02,
        input_box[0] + input_box[2] / 2,
        mid_y + 0.15,
        dashed=True,
        color="#888888",
    )
    draw_arrow(
        ax,
        audio_first[0] + audio_first[2] / 2,
        mid_y - 0.15,
        audio_first[0] + audio_first[2] / 2,
        audio_first[1] + audio_first[3] + 0.02,
        dashed=True,
        color="#888888",
    )

    cls_box = vision_boxes[4]
    audio_last = audio_boxes[-1]
    bridge_y = y_audio + audio_h + row_gap * 0.52
    draw_arrow(
        ax,
        audio_last[0] + audio_last[2] / 2,
        audio_last[1] + audio_last[3] + 0.02,
        audio_last[0] + audio_last[2] / 2,
        bridge_y,
        dashed=True,
        color="#C06030",
    )
    draw_arrow(
        ax,
        audio_last[0] + audio_last[2] / 2,
        bridge_y,
        cls_box[0] + cls_box[2] / 2,
        bridge_y,
        dashed=True,
        color="#C06030",
    )
    draw_arrow(
        ax,
        cls_box[0] + cls_box[2] / 2,
        bridge_y,
        cls_box[0] + cls_box[2] / 2,
        cls_box[1] - 0.02,
        dashed=True,
        color="#C06030",
    )
    ax.text(
        (audio_last[0] + cls_box[0] + audio_last[2] + cls_box[2]) / 4,
        bridge_y + 0.12,
        "late fusion / speech-aligned rerank",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#A04818",
        fontstyle="italic",
    )

    ax.text(
        fig_w / 2,
        0.22,
        "Best: Fusion-EN (EfficientNet + Whisper)  |  Fusion-CLIP baseline",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )

    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)


# System overview drawn at the width of a two-column IEEE figure, in inches.
OVERVIEW_FIGSIZE = (7.16, 2.35)

OVERVIEW_FONT = {
    "row_label": 6.4,
    "box_title": 5.8,
    "box_item": 4.8,
    "edge": 4.6,
}

OVERVIEW_VISION = [
    ("Egocentric frame", ["GoPro chest view", "worker holds the item"]),
    ("SAM3 detection", ["full-image Top-5 masks", "hand ROI as fallback"]),
    ("Mask selection & rerank", ["up to 5 candidates", "hand overlap, SAM score", "top-1 vs top-2 margin"]),
    ("Image classifier", ["CLIP / SigLIP probe", "EfficientNet-B0", "15-way scores"]),
]

OVERVIEW_AUDIO = [
    ("Clip soundtrack", ["wav at the toss", "volume normalized"]),
    ("Whisper large-v2 + LoRA", ["rank 8 on Q/V", "transcript text"]),
    ("Phrase match", ["MiniLM retrieval", "text rules", "speech class"]),
]


def _overview_box(ax, x, y, w, h, title, items, *, facecolor="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.6,
            edgecolor="#333333",
            facecolor=facecolor,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.09,
        title,
        ha="center",
        va="center",
        fontsize=OVERVIEW_FONT["box_title"],
        fontweight="bold",
        color="#1a1a1a",
        zorder=3,
    )
    cursor = y + h - 0.21
    for item in items:
        ax.text(
            x + 0.06,
            cursor,
            item,
            ha="left",
            va="center",
            fontsize=OVERVIEW_FONT["box_item"],
            color="#333333",
            zorder=3,
        )
        cursor -= 0.115


def _overview_arrow(ax, p1, p2, *, dashed=False, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=5,
            linewidth=0.6,
            color=color,
            linestyle=(0, (2.5, 1.5)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def plot_system_overview(output_path: Path) -> None:
    fig_w, fig_h = OVERVIEW_FIGSIZE
    fig = plt.figure(figsize=OVERVIEW_FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    box_w, gap, start_x = 1.28, 0.17, 0.30
    vision_y, vision_h = 1.46, 0.72
    audio_y, audio_h = 0.50, 0.62

    vision_boxes = []
    for idx, (title, items) in enumerate(OVERVIEW_VISION):
        x = start_x + idx * (box_w + gap)
        _overview_box(ax, x, vision_y, box_w, vision_h, title, items)
        vision_boxes.append((x, vision_y, box_w, vision_h))
        if idx:
            prev = vision_boxes[idx - 1]
            _overview_arrow(
                ax,
                (prev[0] + prev[2], vision_y + vision_h / 2),
                (x, vision_y + vision_h / 2),
            )

    audio_boxes = []
    for idx, (title, items) in enumerate(OVERVIEW_AUDIO):
        x = start_x + idx * (box_w + gap)
        _overview_box(ax, x, audio_y, box_w, audio_h, title, items, facecolor="#F4F4F4")
        audio_boxes.append((x, audio_y, box_w, audio_h))
        if idx:
            prev = audio_boxes[idx - 1]
            _overview_arrow(
                ax,
                (prev[0] + prev[2], audio_y + audio_h / 2),
                (x, audio_y + audio_h / 2),
            )

    for label, (_, y, _, h) in (("Vision", vision_boxes[0]), ("Audio", audio_boxes[0])):
        ax.text(
            0.14,
            y + h / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=OVERVIEW_FONT["row_label"],
            fontweight="bold",
            color="#1a1a1a",
        )

    # Same clip feeds both branches.
    frame = vision_boxes[0]
    clip = audio_boxes[0]
    _overview_arrow(
        ax,
        (frame[0] + frame[2] / 2, frame[1]),
        (clip[0] + clip[2] / 2, clip[1] + clip[3]),
        dashed=True,
    )
    ax.text(
        frame[0] + frame[2] / 2 + 0.04,
        (frame[1] + clip[1] + clip[3]) / 2,
        "same clip",
        ha="left",
        va="center",
        fontsize=OVERVIEW_FONT["edge"],
        color="#555555",
        fontstyle="italic",
    )

    # Speech agreement feeds back into reranking.
    rerank = vision_boxes[2]
    phrase = audio_boxes[2]
    bridge_x = phrase[0] + phrase[2] * 0.62
    _overview_arrow(
        ax,
        (bridge_x, phrase[1] + phrase[3]),
        (bridge_x, rerank[1]),
        dashed=True,
    )
    ax.text(
        bridge_x + 0.04,
        (phrase[1] + phrase[3] + rerank[1]) / 2,
        "agreement bonus",
        ha="left",
        va="center",
        fontsize=OVERVIEW_FONT["edge"],
        color="#555555",
        fontstyle="italic",
    )

    fusion_x = start_x + len(OVERVIEW_VISION) * (box_w + gap)
    fusion_y, fusion_h = 0.78, 1.10
    fusion_w = fig_w - fusion_x - 0.08
    _overview_box(
        ax,
        fusion_x,
        fusion_y,
        fusion_w,
        fusion_h,
        "Late fusion",
        [
            "15-D image scores",
            "+ 15-D speech one-hot",
            "logistic regression",
        ],
        facecolor="#EAF0F7",
    )

    classifier = vision_boxes[3]
    _overview_arrow(
        ax,
        (classifier[0] + classifier[2], classifier[1] + classifier[3] / 2),
        (fusion_x, fusion_y + fusion_h * 0.72),
    )
    _overview_arrow(
        ax,
        (phrase[0] + phrase[2], phrase[1] + phrase[3] / 2),
        (fusion_x, fusion_y + fusion_h * 0.25),
    )
    _overview_arrow(
        ax,
        (fusion_x + fusion_w / 2, fusion_y),
        (fusion_x + fusion_w / 2, fusion_y - 0.22),
    )
    ax.text(
        fusion_x + fusion_w / 2,
        fusion_y - 0.32,
        "Final prediction",
        ha="center",
        va="center",
        fontsize=OVERVIEW_FONT["box_title"],
        fontweight="bold",
        color="#1a1a1a",
    )

    fig.savefig(output_path, dpi=600, facecolor="white")
    plt.close(fig)
    _crop_whitespace(output_path, pad=12)


def _crop_whitespace(path: Path, pad: int = 8) -> None:
    from PIL import Image
    import numpy as np

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    ink = np.where((arr < 250).any(axis=2))
    if ink[0].size == 0:
        return
    y0, y1 = int(ink[0].min()), int(ink[0].max()) + 1
    x0, x1 = int(ink[1].min()), int(ink[1].max()) + 1
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(im.size[1], y1 + pad), min(im.size[0], x1 + pad)
    im.crop((x0, y0, x1, y1)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot detection/classification pipeline diagram.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEMO_DIR / "pipeline_detection_classification_diagram.png",
    )
    parser.add_argument(
        "--style",
        choices=("color", "simple", "overview"),
        default="color",
        help="color: demo layout; simple: grayscale paper figure; overview: two-column system overview",
    )
    args = parser.parse_args()
    if args.style == "simple":
        plot_pipeline_diagram_simple(args.output)
    elif args.style == "overview":
        plot_system_overview(args.output)
    else:
        plot_pipeline_diagram(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
