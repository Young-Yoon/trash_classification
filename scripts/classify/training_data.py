"""Dataset loading for supervised image classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from config import META2_CSV, TRANSCRIBE_LORA_DIR
from dataset_paths import resolve_meta_asset_path
from image_utils import CropMode, load_classification_image
from paths import clip_id_from_path


@dataclass
class ClassificationSample:
    frame_path: Path
    mask_path: Optional[Path]
    category_name: str
    gx_id: str
    clip_id: Optional[str]
    crop_mode: CropMode = "bbox"
    bbox_padding_ratio: float = 0.0


def load_classification_samples(
    meta_csv: Path = META2_CSV,
    *,
    use_mask: bool = True,
) -> list[ClassificationSample]:
    meta_df = pd.read_csv(meta_csv)
    samples: list[ClassificationSample] = []

    for _, row in meta_df.iterrows():
        frame_path = resolve_meta_asset_path(row.get("copied_frame_path"))
        if frame_path is None or not frame_path.is_file():
            continue

        mask_path = None
        if use_mask:
            mask_path = resolve_meta_asset_path(row.get("copied_mask_path"))

        clip_id = clip_id_from_path(row.get("timestamp_clip_path"))
        samples.append(
            ClassificationSample(
                frame_path=frame_path,
                mask_path=mask_path,
                category_name=str(row["category_name"]),
                gx_id=str(row["gx_id"]),
                clip_id=clip_id,
            )
        )

    return samples


def build_label_maps(samples: list[ClassificationSample]) -> tuple[dict[str, int], dict[int, str]]:
    categories = sorted({sample.category_name for sample in samples})
    label2id = {name: idx for idx, name in enumerate(categories)}
    id2label = {idx: name for name, idx in label2id.items()}
    return label2id, id2label


def save_label_maps(label2id: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(label2id, handle, indent=2, ensure_ascii=False)


def load_label_maps(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        label2id = json.load(handle)
    id2label = {int(idx): name for name, idx in label2id.items()}
    return label2id, id2label



def count_samples_by_class(
    samples: list[ClassificationSample],
    label2id: dict[str, int],
) -> np.ndarray:
    counts = np.zeros(len(label2id), dtype=np.int64)
    for sample in samples:
        counts[label2id[sample.category_name]] += 1
    return counts


def inverse_frequency_weights(
    counts: np.ndarray,
    *,
    w_max: float = 10.0,
) -> np.ndarray:
    mean_count = float(counts.mean()) if len(counts) else 1.0
    weights = mean_count / np.maximum(counts.astype(np.float64), 1.0)
    return np.minimum(weights, w_max)


def build_oversampled_indices(
    samples: list[ClassificationSample],
    label2id: dict[str, int],
    *,
    min_per_class: int = 80,
    max_multiplier: float = 4.0,
    seed: int = 0,
) -> list[int]:
    """Resample indices so each class appears at least min_per_class times per epoch."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {class_id: [] for class_id in range(len(label2id))}
    for index, sample in enumerate(samples):
        by_class[label2id[sample.category_name]].append(index)

    indices: list[int] = []
    for class_indices in by_class.values():
        if not class_indices:
            continue
        class_count = len(class_indices)
        target = min(max(min_per_class, class_count), int(max_multiplier * class_count))
        if target <= class_count:
            indices.extend(class_indices)
            continue
        extra = rng.choice(class_indices, size=target - class_count, replace=True)
        indices.extend(class_indices)
        indices.extend(int(value) for value in extra)

    rng.shuffle(indices)
    return indices


class BalancedEpochSampler(Sampler[int]):
    """Per-epoch oversampling sampler for tail-class balancing (A2/A3)."""

    def __init__(
        self,
        samples: list[ClassificationSample],
        label2id: dict[str, int],
        *,
        min_per_class: int = 80,
        max_multiplier: float = 4.0,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.label2id = label2id
        self.min_per_class = min_per_class
        self.max_multiplier = max_multiplier
        self.seed = seed
        self.epoch = 0
        self._length = len(
            build_oversampled_indices(
                samples,
                label2id,
                min_per_class=min_per_class,
                max_multiplier=max_multiplier,
                seed=seed,
            )
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        return iter(
            build_oversampled_indices(
                self.samples,
                self.label2id,
                min_per_class=self.min_per_class,
                max_multiplier=self.max_multiplier,
                seed=self.seed + self.epoch,
            )
        )

    def __len__(self) -> int:
        return self._length


def split_samples(
    samples: list[ClassificationSample],
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[ClassificationSample], list[ClassificationSample]]:
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=rng).tolist()
    split_at = max(1, int(len(samples) * (1 - val_ratio)))
    train_idx = indices[:split_at]
    val_idx = indices[split_at:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx] if val_idx else train_samples[: max(1, len(train_samples) // 10)]
    return train_samples, val_samples


class PlasticImageDataset(Dataset):
    def __init__(
        self,
        samples: list[ClassificationSample],
        label2id: dict[str, int],
        *,
        use_mask: bool = True,
        transform=None,
        processor=None,
        crop_mode: CropMode = "bbox",
    ) -> None:
        self.samples = samples
        self.label2id = label2id
        self.use_mask = use_mask
        self.transform = transform
        self.processor = processor
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        mask_path = sample.mask_path if self.use_mask else None
        crop_mode = sample.crop_mode if self.use_mask else "bbox"
        image = load_classification_image(
            sample.frame_path,
            mask_path,
            crop_mode=crop_mode,
            bbox_padding_ratio=sample.bbox_padding_ratio,
        )

        if self.processor is not None:
            encoded = self.processor(images=image, return_tensors="pt")
            pixel_values = encoded["pixel_values"].squeeze(0)
        elif self.transform is not None:
            pixel_values = self.transform(image)
        else:
            pixel_values = transforms.ToTensor()(image)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(self.label2id[sample.category_name], dtype=torch.long),
            "gx_id": sample.gx_id,
            "frame_path": str(sample.frame_path),
            "clip_id": sample.clip_id or "",
        }


def load_text_category_map(
    transcribe_dir: Path = TRANSCRIBE_LORA_DIR,
) -> dict[str, str]:
    """Map audio clip_id to predicted category from LoRA transcriptions."""
    sys_path_added = False
    import sys

    whisper_dir = Path(__file__).resolve().parent.parent / "whisper"
    if str(whisper_dir) not in sys.path:
        sys.path.insert(0, str(whisper_dir))
        sys_path_added = True

    try:
        from category_mapping import SemanticCategoryMapper, map_transcript_to_category
    finally:
        if sys_path_added:
            sys.path.remove(str(whisper_dir))

    mapper = SemanticCategoryMapper()
    clip_to_category: dict[str, str] = {}

    if not transcribe_dir.is_dir():
        return clip_to_category

    for txt_path in transcribe_dir.glob("*.txt"):
        clip_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        rule_category = map_transcript_to_category(text)
        semantic_category = mapper.predict(text)
        clip_to_category[clip_id] = semantic_category or rule_category

    return clip_to_category
