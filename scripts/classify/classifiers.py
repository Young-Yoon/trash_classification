"""Plastic object classifiers from imageClassify.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from image_utils import CropMode, load_classification_image


def normalize_top_k_results(results: list[dict], top_k: int = 5) -> list[dict]:
    normalized = []
    for item in results[:top_k]:
        normalized.append(
            {
                "label": item["label"],
                "score": float(item["score"]),
            }
        )
    return normalized


def plastic_classify_clip(
    frame_path: str | Path,
    mask_path: str | Path | None,
    classifier: Any,
    candidate_labels: list[str],
    *,
    top_k: int = 5,
    crop_mode: CropMode = "bbox",
) -> list[dict] | dict:
    try:
        cropped_image = load_classification_image(frame_path, mask_path, crop_mode=crop_mode)
        results = classifier(cropped_image, candidate_labels=candidate_labels)
        return normalize_top_k_results(results, top_k=top_k)
    except FileNotFoundError:
        return {"error": f"Image file not found: {frame_path}"}
    except Exception as exc:
        return {"error": f"An error occurred during CLIP classification for {frame_path}: {exc}"}


def plastic_classify_siglip(
    frame_path: str | Path,
    mask_path: str | Path | None,
    classifier: Any,
    candidate_labels: list[str],
    *,
    top_k: int = 5,
    crop_mode: CropMode = "bbox",
) -> list[dict] | dict:
    try:
        cropped_image = load_classification_image(frame_path, mask_path, crop_mode=crop_mode)
        results = classifier(cropped_image, candidate_labels=candidate_labels)
        return normalize_top_k_results(results, top_k=top_k)
    except FileNotFoundError:
        return {"error": f"Image file not found: {frame_path}"}
    except Exception as exc:
        return {"error": f"An error occurred during SigLIP classification for {frame_path}: {exc}"}


def plastic_classify_gemini(
    frame_path: str | Path,
    mask_path: str | Path | None,
    gemini_model: Any,
    candidate_labels: list[str],
    *,
    crop_mode: CropMode = "bbox",
) -> dict:
    try:
        cropped_image = load_classification_image(frame_path, mask_path, crop_mode=crop_mode)
        labels_str = ", ".join(f'"{label}"' for label in candidate_labels)
        prompt = (
            "Classify the object in this image into one of the following categories: "
            f"{labels_str}. Respond only with the chosen category name."
        )
        response = gemini_model.generate_content([prompt, cropped_image])
        predicted_label = response.text.strip()

        if predicted_label in candidate_labels:
            return {"label": predicted_label, "score": 1.0}

        for label in candidate_labels:
            if label in predicted_label or predicted_label in label:
                return {"label": label, "score": 1.0}

        return {"label": predicted_label, "score": 0.0, "note": "Label not in candidate list"}
    except FileNotFoundError:
        return {"error": f"Image file not found: {frame_path}"}
    except Exception as exc:
        return {"error": f"An error occurred during Gemini classification for {frame_path}: {exc}"}
