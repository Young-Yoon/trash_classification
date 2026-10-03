#!/usr/bin/env python3
"""Train supervised image classifiers (linear probe, LoRA, EfficientNet, fusion)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from cnn_model import build_efficientnet, efficientnet_transform, save_cnn_checkpoint
from config import CLASSIFY_CHECKPOINT_DIR, META2_CSV
from fusion_image import detect_image_model_type, predict_image_probabilities
from fusion_model import build_fusion_features, save_fusion_checkpoint, train_fusion_classifier
from segment_masks import RERANK_SEGMENT_VERSIONS, load_segment_classification_samples
from segment_rerank import load_rerank_classification_samples
from training_data import (
    BalancedEpochSampler,
    ClassificationSample,
    PlasticImageDataset,
    build_label_maps,
    count_samples_by_class,
    inverse_frequency_weights,
    load_classification_samples,
    load_text_category_map,
    split_samples,
)
from vision_model import VisionClassifier, load_processor, save_vision_checkpoint


def collate_batch(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "gx_id": [item["gx_id"] for item in batch],
        "frame_path": [item["frame_path"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
    }


class FocalLoss(nn.Module):
    def __init__(self, *, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        batches += 1
    if batches == 0:
        return 0.0, 0.0
    return total_loss / batches, total_acc / batches


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    criterion: nn.Module | None = None,
    train_sampler: Sampler[int] | None = None,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch - 1)
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        batches = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs} train"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(logits, labels)
            batches += 1

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        train_loss /= max(batches, 1)
        train_acc /= max(batches, 1)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_balance_training(
    train_samples: list[ClassificationSample],
    label2id: dict[str, int],
    device: torch.device,
    *,
    balance_mode: str,
    class_weight_max: float,
    oversample_min_per_class: int,
    oversample_max_multiplier: float,
    focal_gamma: float,
) -> tuple[nn.Module | None, Sampler[int] | None, dict]:
    counts = count_samples_by_class(train_samples, label2id)
    class_weights = inverse_frequency_weights(counts, w_max=class_weight_max)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    use_reweight = balance_mode in {"reweight", "combined"}
    use_oversample = balance_mode in {"oversample", "combined"}
    use_focal = balance_mode == "combined"

    criterion: nn.Module | None = None
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, weight=weight_tensor if use_reweight else None)
    elif use_reweight:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_sampler: Sampler[int] | None = None
    if use_oversample:
        train_sampler = BalancedEpochSampler(
            train_samples,
            label2id,
            min_per_class=oversample_min_per_class,
            max_multiplier=oversample_max_multiplier,
        )

    config = {
        "balance_mode": balance_mode,
        "class_counts": counts.tolist(),
        "class_weights": class_weights.tolist(),
        "oversample_min_per_class": oversample_min_per_class,
        "oversample_max_multiplier": oversample_max_multiplier,
        "focal_gamma": focal_gamma if use_focal else None,
        "train_epoch_size": len(train_sampler) if train_sampler is not None else len(train_samples),
    }
    return criterion, train_sampler, config


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict"):
        pixel_values = batch["pixel_values"].to(device)
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def load_training_samples(
    *,
    meta_csv: Path,
    use_mask: bool,
    segment_version: str | None,
) -> list[ClassificationSample]:
    if not segment_version:
        return load_classification_samples(meta_csv=meta_csv, use_mask=use_mask)

    if segment_version in RERANK_SEGMENT_VERSIONS:
        rerank_samples = load_rerank_classification_samples(
            segment_version,  # type: ignore[arg-type]
            meta_csv=meta_csv,
        )
        samples: list[ClassificationSample] = []
        for sample in rerank_samples:
            mask_path = sample.candidate_mask_paths[0] if sample.candidate_mask_paths else sample.mask_path
            samples.append(
                ClassificationSample(
                    frame_path=sample.frame_path,
                    mask_path=mask_path,
                    category_name=sample.category_name,
                    gx_id=sample.gx_id,
                    clip_id=sample.clip_id,
                    crop_mode=sample.crop_mode,
                    bbox_padding_ratio=sample.bbox_padding_ratio,
                )
            )
        return samples

    return load_segment_classification_samples(
        segment_version,  # type: ignore[arg-type]
        meta_csv=meta_csv,
        export_if_missing=True,
    )


def maybe_load_init_checkpoint(model: nn.Module, init_checkpoint: Path | None, device: torch.device) -> None:
    if init_checkpoint is None:
        return
    from cnn_model import load_cnn_checkpoint
    from vision_model import load_vision_checkpoint

    if init_checkpoint.is_dir():
        try:
            partial_model, _, _ = load_cnn_checkpoint(init_checkpoint, device)
            model.load_state_dict(partial_model.state_dict(), strict=False)
            print(f"Initialized weights from CNN checkpoint: {init_checkpoint}")
            return
        except Exception:
            partial_model, _, _, _ = load_vision_checkpoint(init_checkpoint, device)
            model.load_state_dict(partial_model.state_dict(), strict=False)
            print(f"Initialized weights from vision checkpoint: {init_checkpoint}")


def default_output_dir(method: str, backbone: str, *, use_mask: bool, fusion_image: str = "clip") -> Path:
    mask_tag = "crop" if use_mask else "full"
    if method == "efficientnet":
        name = f"efficientnet_b0_{mask_tag}"
    elif method == "fusion":
        if fusion_image == "efficientnet":
            name = f"late_fusion_efficientnet_whisper_{mask_tag}"
        else:
            name = f"late_fusion_clip_whisper_{mask_tag}"
    else:
        name = f"{backbone}_{method}_{mask_tag}"
    return CLASSIFY_CHECKPOINT_DIR / name


def train_vision(
    *,
    method: str,
    backbone: str,
    use_mask: bool,
    epochs: int,
    batch_size: int,
    val_ratio: float,
    learning_rate: float | None,
    output_dir: Path,
    device: torch.device,
    meta_csv: Path = META2_CSV,
    segment_version: str | None = None,
    init_checkpoint: Path | None = None,
) -> Path:
    samples = load_training_samples(
        meta_csv=meta_csv,
        use_mask=use_mask,
        segment_version=segment_version,
    )
    label2id, _ = build_label_maps(samples)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio)

    processor = load_processor(backbone)
    train_ds = PlasticImageDataset(train_samples, label2id, use_mask=use_mask, processor=processor)
    val_ds = PlasticImageDataset(val_samples, label2id, use_mask=use_mask, processor=processor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = VisionClassifier(backbone, len(label2id), method=method)
    maybe_load_init_checkpoint(model, init_checkpoint, device)
    lr = learning_rate or (1e-3 if method == "linear_probe" else 5e-5)
    model = train_torch_model(model, train_loader, val_loader, device, epochs=epochs, learning_rate=lr)

    save_vision_checkpoint(model, processor, label2id, output_dir, use_mask=use_mask)
    metrics = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "num_classes": len(label2id),
        "method": method,
        "backbone": backbone,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved checkpoint: {output_dir}")
    return output_dir


def train_efficientnet(
    *,
    use_mask: bool,
    epochs: int,
    batch_size: int,
    val_ratio: float,
    learning_rate: float | None,
    output_dir: Path,
    device: torch.device,
    meta_csv: Path = META2_CSV,
    segment_version: str | None = None,
    init_checkpoint: Path | None = None,
    balance_mode: str = "none",
    class_weight_max: float = 10.0,
    oversample_min_per_class: int = 80,
    oversample_max_multiplier: float = 4.0,
    focal_gamma: float = 2.0,
) -> Path:
    samples = load_training_samples(
        meta_csv=meta_csv,
        use_mask=use_mask,
        segment_version=segment_version,
    )
    label2id, _ = build_label_maps(samples)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio)
    transform = efficientnet_transform()

    train_ds = PlasticImageDataset(train_samples, label2id, use_mask=use_mask, transform=transform)
    val_ds = PlasticImageDataset(val_samples, label2id, use_mask=use_mask, transform=transform)

    criterion, train_sampler, balance_config = build_balance_training(
        train_samples,
        label2id,
        device,
        balance_mode=balance_mode,
        class_weight_max=class_weight_max,
        oversample_min_per_class=oversample_min_per_class,
        oversample_max_multiplier=oversample_max_multiplier,
        focal_gamma=focal_gamma,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_efficientnet(len(label2id))
    maybe_load_init_checkpoint(model, init_checkpoint, device)
    lr = learning_rate or 1e-4
    model = train_torch_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        learning_rate=lr,
        criterion=criterion,
        train_sampler=train_sampler,
    )
    save_cnn_checkpoint(model, label2id, output_dir, use_mask=use_mask)
    with (output_dir / "balance_config.json").open("w", encoding="utf-8") as handle:
        json.dump(balance_config, handle, indent=2)
    print(f"Saved checkpoint: {output_dir}")
    return output_dir


def train_fusion(
    *,
    image_checkpoint: Path,
    use_mask: bool,
    batch_size: int,
    val_ratio: float,
    output_dir: Path,
    device: torch.device,
) -> Path:
    samples = load_classification_samples(use_mask=use_mask)
    label2id, _ = build_label_maps(samples)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio)
    clip_to_category = load_text_category_map()
    image_model_type = detect_image_model_type(image_checkpoint)

    train_probs, image_label2id = predict_image_probabilities(
        image_checkpoint,
        train_samples,
        batch_size=batch_size,
        device=device,
    )
    val_probs, _ = predict_image_probabilities(
        image_checkpoint,
        val_samples,
        batch_size=batch_size,
        device=device,
    )
    if image_label2id != label2id:
        raise SystemExit("Image checkpoint label map does not match current dataset labels.")

    fusion_model = train_fusion_classifier(train_samples, train_probs, label2id, clip_to_category)
    train_preds = fusion_model.predict(
        np.stack(
            [
                build_fusion_features(
                    probs,
                    clip_to_category.get(sample.clip_id or ""),
                    label2id,
                )
                for sample, probs in zip(train_samples, train_probs)
            ]
        )
    )
    val_preds = fusion_model.predict(
        np.stack(
            [
                build_fusion_features(
                    probs,
                    clip_to_category.get(sample.clip_id or ""),
                    label2id,
                )
                for sample, probs in zip(val_samples, val_probs)
            ]
        )
    )
    train_acc = np.mean([label2id[s.category_name] == p for s, p in zip(train_samples, train_preds)])
    val_acc = np.mean([label2id[s.category_name] == p for s, p in zip(val_samples, val_preds)])
    print(f"fusion train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    model_name = (
        "late_fusion_efficientnet_whisper_crop"
        if image_model_type == "efficientnet"
        else "late_fusion_clip_whisper_crop"
    )
    save_fusion_checkpoint(
        fusion_model,
        label2id,
        str(image_checkpoint),
        output_dir,
        use_mask=use_mask,
        image_model_type=image_model_type,
        model_name=model_name,
    )
    print(f"Saved fusion checkpoint: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised plastic image classifiers.")
    parser.add_argument(
        "--method",
        choices=["linear_probe", "lora", "efficientnet", "fusion"],
        required=True,
    )
    parser.add_argument("--backbone", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--meta-csv", type=Path, default=META2_CSV)
    parser.add_argument("--use-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-version", type=str, default=None, help="Train on predicted segment crops.")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Fine-tune from existing checkpoint.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--balance-mode",
        choices=["none", "reweight", "oversample", "combined"],
        default="none",
        help="A0=none, A1=reweight, A2=oversample, A3=combined (reweight+oversample+focal).",
    )
    parser.add_argument("--class-weight-max", type=float, default=10.0)
    parser.add_argument("--oversample-min-per-class", type=int, default=80)
    parser.add_argument("--oversample-max-multiplier", type=float, default=4.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fusion-image",
        choices=["clip", "efficientnet"],
        default="clip",
        help="Image backbone for fusion training.",
    )
    parser.add_argument(
        "--image-checkpoint",
        type=Path,
        default=None,
        help="Required for fusion; defaults depend on --fusion-image.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir or default_output_dir(
        args.method,
        args.backbone,
        use_mask=args.use_mask,
        fusion_image=args.fusion_image,
    )

    if args.method in {"linear_probe", "lora"}:
        train_vision(
            method=args.method,
            backbone=args.backbone,
            use_mask=args.use_mask,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            device=device,
            meta_csv=args.meta_csv,
            segment_version=args.segment_version,
            init_checkpoint=args.init_checkpoint,
        )
    elif args.method == "efficientnet":
        train_efficientnet(
            use_mask=args.use_mask,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            device=device,
            meta_csv=args.meta_csv,
            segment_version=args.segment_version,
            init_checkpoint=args.init_checkpoint,
            balance_mode=args.balance_mode,
            class_weight_max=args.class_weight_max,
            oversample_min_per_class=args.oversample_min_per_class,
            oversample_max_multiplier=args.oversample_max_multiplier,
            focal_gamma=args.focal_gamma,
        )
    else:
        if args.fusion_image == "efficientnet":
            image_checkpoint = args.image_checkpoint or (
                CLASSIFY_CHECKPOINT_DIR / "efficientnet_b0_crop"
            )
        else:
            image_checkpoint = args.image_checkpoint or (
                CLASSIFY_CHECKPOINT_DIR / "clip_linear_probe_crop"
            )
        if not image_checkpoint.is_dir():
            raise SystemExit(
                f"Image checkpoint not found: {image_checkpoint}. "
                "Train the image classifier first or pass --image-checkpoint."
            )
        train_fusion(
            image_checkpoint=image_checkpoint,
            use_mask=args.use_mask,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            output_dir=output_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
