"""Image cropping helpers for mask-guided classification."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image

CropMode = Literal["bbox", "masked"]


def get_bbox_from_mask(mask_path: str | Path) -> Optional[tuple[int, int, int, int]]:
    try:
        mask = Image.open(mask_path).convert("L")
    except FileNotFoundError:
        return None

    mask_array = np.array(mask)
    active_pixels = np.argwhere(mask_array > 0)
    if active_pixels.size == 0:
        return None

    min_y, min_x = active_pixels.min(axis=0)
    max_y, max_x = active_pixels.max(axis=0)
    return int(min_x), int(min_y), int(max_x + 1), int(max_y + 1)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    padding_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    if padding_ratio <= 0:
        return bbox

    xmin, ymin, xmax, ymax = bbox
    width = max(1, xmax - xmin)
    height = max(1, ymax - ymin)
    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))
    image_width, image_height = image_size
    return (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(image_width, xmax + pad_x),
        min(image_height, ymax + pad_y),
    )


def apply_mask_to_image(image: Image.Image, mask_path: str | Path) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    rgba = image.convert("RGBA")
    mask_array = np.array(mask)
    alpha = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    composited = Image.alpha_composite(background, rgba)
    return composited.convert("RGB")


def load_classification_image(
    frame_path: str | Path,
    mask_path: str | Path | None,
    *,
    crop_mode: CropMode = "bbox",
    bbox_padding_ratio: float = 0.0,
) -> Image.Image:
    image = Image.open(frame_path).convert("RGB")
    if mask_path and Path(mask_path).is_file():
        if crop_mode == "masked":
            masked = apply_mask_to_image(image, mask_path)
            bbox = get_bbox_from_mask(mask_path)
            if bbox:
                bbox = expand_bbox(bbox, image.size, padding_ratio=bbox_padding_ratio)
                return masked.crop(bbox)
            return masked

        bbox = get_bbox_from_mask(mask_path)
        if bbox:
            bbox = expand_bbox(bbox, image.size, padding_ratio=bbox_padding_ratio)
            return image.crop(bbox)
    return image
