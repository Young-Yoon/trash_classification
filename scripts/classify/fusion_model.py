"""Late fusion of image probabilities and Whisper LoRA text categories."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from training_data import ClassificationSample, load_text_category_map


def build_fusion_features(
    image_probs: np.ndarray,
    text_category: str | None,
    label2id: dict[str, int],
) -> np.ndarray:
    text_vec = np.zeros(len(label2id), dtype=np.float32)
    if text_category and text_category in label2id:
        text_vec[label2id[text_category]] = 1.0
    return np.concatenate([image_probs.astype(np.float32), text_vec], axis=0)


def train_fusion_classifier(
    samples: list[ClassificationSample],
    image_prob_matrix: np.ndarray,
    label2id: dict[str, int],
    clip_to_category: dict[str, str],
) -> LogisticRegression:
    features = []
    labels = []
    for sample, probs in zip(samples, image_prob_matrix):
        text_category = clip_to_category.get(sample.clip_id or "")
        features.append(build_fusion_features(probs, text_category, label2id))
        labels.append(label2id[sample.category_name])

    x_train = np.stack(features)
    y_train = np.array(labels)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model


def save_fusion_checkpoint(
    fusion_model: LogisticRegression,
    label2id: dict[str, int],
    image_checkpoint: str,
    output_dir: Path,
    *,
    use_mask: bool,
    image_model_type: str = "vision",
    model_name: str = "late_fusion",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fusion.pkl").open("wb") as handle:
        pickle.dump(fusion_model, handle)
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label2id, handle, indent=2, ensure_ascii=False)
    config = {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "model_name": model_name,
        "image_checkpoint": image_checkpoint,
        "image_model_type": image_model_type,
        "use_mask": use_mask,
        "num_classes": len(label2id),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_fusion_checkpoint(checkpoint_dir: Path) -> tuple[LogisticRegression, dict[str, int], dict]:
    with (checkpoint_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (checkpoint_dir / "label_map.json").open("r", encoding="utf-8") as handle:
        label2id = json.load(handle)
    with (checkpoint_dir / "fusion.pkl").open("rb") as handle:
        fusion_model = pickle.load(handle)
    return fusion_model, label2id, config
