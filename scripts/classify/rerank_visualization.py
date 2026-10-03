"""Visualize rerank candidate bboxes with before/after selection."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
SEGMENT_DIR_PKG = PACKAGE_DIR.parent / "segment"
for path in (SHARED_DIR, PACKAGE_DIR, SEGMENT_DIR_PKG):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from config import CLASSIFY_CHECKPOINT_DIR
from geometry import compute_bbox_iou
from image_utils import expand_bbox, get_bbox_from_mask
from infer_trained import pick_fusion_rerank_candidate_index
from presets import TRAINED_PRESETS
from segment_masks import RERANK_SEGMENT_VERSIONS, RERANK_STRATEGIES, SEGMENT_BBOX_PADDING, SegmentVersion


CANDIDATE_COLORS = [
    (0.55, 0.55, 0.55),
    (0.45, 0.45, 0.45),
    (0.65, 0.65, 0.65),
    (0.50, 0.50, 0.50),
    (0.60, 0.60, 0.60),
]
SELECTED_COLOR = (1.0, 0.15, 0.15)
GT_MASK_COLOR = (0, 0.78, 0)


def overlay_mask_on_frame(frame: Image.Image, mask_path: Path, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.array(frame.convert("RGB"), dtype=np.float32)
    if mask_path.is_file():
        mask = Image.open(mask_path).convert("L")
        if mask.size != frame.size:
            mask = mask.resize(frame.size, Image.NEAREST)
        mask_arr = np.array(mask) > 0
        alpha = 0.40
        for channel in range(3):
            rgb[:, :, channel] = np.where(
                mask_arr,
                rgb[:, :, channel] * (1 - alpha) + color[channel] * 255 * alpha,
                rgb[:, :, channel],
            )
    return rgb.astype(np.uint8)


def candidate_bboxes(
    candidate_mask_paths: list[Path],
    frame_size: tuple[int, int],
    *,
    padding_ratio: float,
) -> list[tuple[int, int, int, int] | None]:
    bboxes: list[tuple[int, int, int, int] | None] = []
    for mask_path in candidate_mask_paths:
        bbox = get_bbox_from_mask(mask_path)
        if bbox is None:
            bboxes.append(None)
            continue
        bboxes.append(expand_bbox(bbox, frame_size, padding_ratio=padding_ratio))
    return bboxes


def bbox_iou_pair(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    return float(compute_bbox_iou(list(left), list(right)))


def compute_after_candidate_index(
    sample,
    base_preset: str,
    version: SegmentVersion,
    *,
    device: torch.device,
) -> int:
    preset = TRAINED_PRESETS[base_preset]
    model_type = preset.get("model_type")
    checkpoint_dir = CLASSIFY_CHECKPOINT_DIR / preset["checkpoint_name"]
    strategy = RERANK_STRATEGIES[version]

    if model_type == "fusion":
        return pick_fusion_rerank_candidate_index(
            checkpoint_dir,
            sample,
            device=device,
            rerank_strategy=strategy,
        )

    from fusion_image import predict_crop_probabilities
    from infer_trained import pick_best_candidate_index
    from segment_rerank import load_candidate_crops
    from training_data import load_text_category_map

    crops = load_candidate_crops(sample)
    if not crops:
        return 0
    probs, id2label = predict_crop_probabilities(checkpoint_dir, crops, device=device)
    clip_to_category = load_text_category_map()
    text_category = clip_to_category.get(sample.clip_id or "", "")
    return pick_best_candidate_index(
        probs,
        strategy=strategy,
        candidate_scores=sample.candidate_scores,
        candidate_hand_ious=sample.candidate_hand_ious,
        id2label=id2label,
        text_category=text_category,
    )


def draw_candidate_bboxes(
    ax: plt.Axes,
    frame_rgb: np.ndarray,
    bboxes: list[tuple[int, int, int, int] | None],
    *,
    selected_idx: int,
    selected_color: tuple[float, float, float],
    title: str,
    scores: list[float] | None = None,
) -> None:
    ax.imshow(frame_rgb)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    for idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        is_selected = idx == selected_idx
        color = selected_color if is_selected else CANDIDATE_COLORS[idx % len(CANDIDATE_COLORS)]
        linewidth = 3.0 if is_selected else 1.2
        linestyle = "-" if is_selected else "--"
        rect = Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        score_text = ""
        if scores and idx < len(scores):
            score_text = f" s={scores[idx]:.2f}"
        label = f"#{idx}{score_text}"
        ax.text(
            xmin,
            max(0, ymin - 4),
            label,
            color=color,
            fontsize=8,
            fontweight="bold" if is_selected else "normal",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1},
        )


def render_rerank_before_after(
    sample,
    *,
    version: SegmentVersion,
    base_preset: str,
    gt_mask_path: Path,
    gt_label: str,
    before_idx: int,
    after_idx: int,
    output_path: Path,
    device: torch.device,
    pred_label: str = "",
) -> dict:
    frame = Image.open(sample.frame_path).convert("RGB")
    padding = SEGMENT_BBOX_PADDING[version]
    bboxes = candidate_bboxes(
        sample.candidate_mask_paths,
        frame.size,
        padding_ratio=padding,
    )
    gt_bbox = get_bbox_from_mask(gt_mask_path)
    if gt_bbox is not None:
        gt_bbox = expand_bbox(gt_bbox, frame.size, padding_ratio=padding)

    before_iou = bbox_iou_pair(bboxes[before_idx] if before_idx < len(bboxes) else None, gt_bbox)
    after_iou = bbox_iou_pair(bboxes[after_idx] if after_idx < len(bboxes) else None, gt_bbox)
    meta = {
        "before_idx": before_idx,
        "after_idx": after_idx,
        "before_bbox_iou": round(before_iou, 4),
        "after_bbox_iou": round(after_iou, 4),
        "rerank_changed": before_idx != after_idx,
        "num_candidates": len(bboxes),
    }

    panel = overlay_mask_on_frame(frame, gt_mask_path, GT_MASK_COLOR)

    img_w, img_h = frame.size
    fig_w = 10.0
    title_frac = 0.065
    legend_frac = 0.045
    axes_h = fig_w * (img_h / img_w)
    fig_h = axes_h / (1.0 - title_frac - legend_frac)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0, right=1, top=1.0 - title_frac, bottom=legend_frac)
    draw_candidate_bboxes(
        ax,
        panel,
        bboxes,
        selected_idx=after_idx,
        selected_color=SELECTED_COLOR,
        title="",
        scores=sample.candidate_scores,
    )
    ax.margins(0)
    ax.set_aspect("equal", adjustable="box")

    pred_text = f" | Pred: {pred_label}" if pred_label else ""
    fig.suptitle(
        f"{sample.gx_id} | GT: {gt_label}{pred_text}\n"
        f"After rerank: cand #{after_idx} / top-{len(bboxes)} | bbox IoU={after_iou:.3f}",
        fontsize=10,
        y=1.0 - title_frac * 0.35,
    )

    legend_handles = [
        Patch(facecolor=GT_MASK_COLOR, label="GT mask"),
        Patch(edgecolor=SELECTED_COLOR, facecolor="none", label="Selected (after rerank)"),
        Patch(edgecolor=(0.55, 0.55, 0.55), facecolor="none", linestyle="--", label="Other candidates"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        columnspacing=1.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, pad_inches=0.02)
    plt.close(fig)

    return meta


def summarize_rerank_selection(
    sample,
    *,
    version: SegmentVersion,
    gt_mask_path: Path,
    before_idx: int,
    after_idx: int,
) -> dict:
    frame = Image.open(sample.frame_path).convert("RGB")
    padding = SEGMENT_BBOX_PADDING[version]
    bboxes = candidate_bboxes(
        sample.candidate_mask_paths,
        frame.size,
        padding_ratio=padding,
    )
    gt_bbox = get_bbox_from_mask(gt_mask_path)
    if gt_bbox is not None:
        gt_bbox = expand_bbox(gt_bbox, frame.size, padding_ratio=padding)
    before_iou = bbox_iou_pair(bboxes[before_idx] if before_idx < len(bboxes) else None, gt_bbox)
    after_iou = bbox_iou_pair(bboxes[after_idx] if after_idx < len(bboxes) else None, gt_bbox)
    return {
        "before_idx": before_idx,
        "after_idx": after_idx,
        "before_bbox_iou": round(before_iou, 4),
        "after_bbox_iou": round(after_iou, 4),
        "rerank_changed": before_idx != after_idx,
        "num_candidates": len(bboxes),
    }


def is_rerank_version(version: str) -> bool:
    return version in RERANK_SEGMENT_VERSIONS
