"""Load vision or EfficientNet checkpoints for late fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn_model import efficientnet_transform, load_cnn_checkpoint
from training_data import ClassificationSample, PlasticImageDataset
from vision_model import load_vision_checkpoint


def encode_crops(crops: list, *, processor, transform) -> torch.Tensor:
    if processor is not None:
        pixel_values = [
            processor(images=crop, return_tensors="pt")["pixel_values"].squeeze(0) for crop in crops
        ]
        return torch.stack(pixel_values, dim=0)
    if transform is not None:
        return torch.stack([transform(crop) for crop in crops], dim=0)
    raise ValueError("Either processor or transform must be provided for rerank inference.")


def detect_image_model_type(image_checkpoint: Path) -> str:
    config_path = image_checkpoint / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("model_type") == "efficientnet":
            return "efficientnet"
    return "vision"


@torch.no_grad()
def predict_image_probabilities(
    image_checkpoint: Path,
    samples: list[ClassificationSample],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int]]:
    from train import collate_batch

    image_model_type = detect_image_model_type(image_checkpoint)
    if image_model_type == "efficientnet":
        model, label2id, config = load_cnn_checkpoint(image_checkpoint, device)
        dataset = PlasticImageDataset(
            samples,
            label2id,
            use_mask=config["use_mask"],
            transform=efficientnet_transform(),
        )
    else:
        model, processor, label2id, config = load_vision_checkpoint(image_checkpoint, device)
        dataset = PlasticImageDataset(
            samples,
            label2id,
            use_mask=config["use_mask"],
            processor=processor,
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model.eval()
    all_probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"{image_checkpoint.name} probs"):
        pixel_values = batch["pixel_values"].to(device)
        probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0), label2id


@torch.no_grad()
def predict_crop_probabilities(
    image_checkpoint: Path,
    crops: list,
    *,
    device: torch.device,
) -> tuple[np.ndarray, dict[int, str]]:
    image_model_type = detect_image_model_type(image_checkpoint)
    if image_model_type == "efficientnet":
        model, label2id, _config = load_cnn_checkpoint(image_checkpoint, device)
        id2label = {idx: name for name, idx in label2id.items()}
        transform = efficientnet_transform()
        pixel_values = encode_crops(crops, processor=None, transform=transform).to(device)
    else:
        model, processor, label2id, _config = load_vision_checkpoint(image_checkpoint, device)
        id2label = {idx: name for name, idx in label2id.items()}
        pixel_values = encode_crops(crops, processor=processor, transform=None).to(device)

    probs = torch.softmax(model(pixel_values), dim=-1).cpu().numpy()
    return probs, id2label
