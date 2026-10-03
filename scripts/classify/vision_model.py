"""Vision backbone classifiers with linear probe or LoRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, CLIPProcessor, CLIPVisionModel, SiglipVisionModel

BackboneName = Literal["clip", "siglip"]

MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


def load_processor(backbone: BackboneName):
    model_name = MODEL_NAMES[backbone]
    if backbone == "clip":
        return CLIPProcessor.from_pretrained(model_name)
    return AutoProcessor.from_pretrained(model_name)


def load_vision_backbone(backbone: BackboneName):
    model_name = MODEL_NAMES[backbone]
    if backbone == "clip":
        return CLIPVisionModel.from_pretrained(model_name)
    return SiglipVisionModel.from_pretrained(model_name)


def pool_vision_features(backbone: BackboneName, outputs) -> torch.Tensor:
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0, :]


class VisionClassifier(nn.Module):
    def __init__(
        self,
        backbone: BackboneName,
        num_classes: int,
        *,
        method: Literal["linear_probe", "lora"] = "linear_probe",
        lora_r: int = 8,
        lora_alpha: int = 32,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.method = method
        self.model_name = MODEL_NAMES[backbone]
        vision = load_vision_backbone(backbone)

        if method == "linear_probe":
            for param in vision.parameters():
                param.requires_grad = False
            self.backbone = vision
        else:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.backbone = get_peft_model(vision, lora_config)

        hidden_size = vision.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = pool_vision_features(self.backbone_name, outputs)
        return self.classifier(pooled)

    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.forward(pixel_values)
        return torch.softmax(logits, dim=-1)


def save_vision_checkpoint(
    model: VisionClassifier,
    processor,
    label2id: dict[str, int],
    output_dir: Path,
    *,
    use_mask: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    processor.save_pretrained(output_dir / "processor")
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label2id, handle, indent=2, ensure_ascii=False)
    config = {
        "model_type": "vision",
        "method": model.method,
        "backbone": model.backbone_name,
        "model_name": model.model_name,
        "use_mask": use_mask,
        "num_classes": len(label2id),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_vision_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[VisionClassifier, object, dict[str, int], dict]:
    with (checkpoint_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (checkpoint_dir / "label_map.json").open("r", encoding="utf-8") as handle:
        label2id = json.load(handle)

    model = VisionClassifier(
        config["backbone"],
        config["num_classes"],
        method=config["method"],
    )
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    processor_dir = checkpoint_dir / "processor"
    if config["backbone"] == "clip":
        processor = CLIPProcessor.from_pretrained(processor_dir)
    else:
        processor = AutoProcessor.from_pretrained(processor_dir)

    return model, processor, label2id, config
