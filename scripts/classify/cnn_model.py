"""EfficientNet image classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet(num_classes: int) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def efficientnet_transform() -> transforms.Compose:
    weights = EfficientNet_B0_Weights.DEFAULT
    return weights.transforms()


def save_cnn_checkpoint(
    model: nn.Module,
    label2id: dict[str, int],
    output_dir: Path,
    *,
    use_mask: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label2id, handle, indent=2, ensure_ascii=False)
    config = {
        "model_type": "efficientnet",
        "method": "efficientnet_b0",
        "backbone": "efficientnet_b0",
        "model_name": "efficientnet_b0",
        "use_mask": use_mask,
        "num_classes": len(label2id),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_cnn_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[nn.Module, dict[str, int], dict]:
    with (checkpoint_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (checkpoint_dir / "label_map.json").open("r", encoding="utf-8") as handle:
        label2id = json.load(handle)

    model = build_efficientnet(config["num_classes"])
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, label2id, config
