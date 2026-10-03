"""Path normalization helpers for dataset metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATASET_ASSET_ROOT, DATASET_META_ROOT, DATASET_ROOT, LEGACY_COLAB_PREFIX, LEGACY_EXTERNAL_ROOT


def clip_id_from_path(path_value: Optional[str]) -> Optional[str]:
    """Extract clip id (filename without extension) from any stored path format."""
    if pd.isna(path_value) or not path_value:
        return None

    normalized = str(path_value)
    if normalized.startswith(LEGACY_EXTERNAL_ROOT):
        normalized = normalized.replace(LEGACY_EXTERNAL_ROOT, LEGACY_COLAB_PREFIX)

    dataset_root_str = str(DATASET_ROOT)
    if normalized.startswith(dataset_root_str):
        relative = normalized[len(dataset_root_str) :].lstrip("/\\")
        return Path(relative).stem

    return Path(normalized).stem


def resolve_dataset_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a metadata path to a local file under DATASET_ROOT when possible."""
    if pd.isna(path_value) or not path_value:
        return None

    normalized = str(path_value)
    marker = "camera1_plastics_shareable_dataset/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        for root in (DATASET_ASSET_ROOT, DATASET_META_ROOT):
            candidate = root / relative
            if candidate.is_file():
                return candidate
        return DATASET_ASSET_ROOT / relative

    if normalized.startswith(LEGACY_EXTERNAL_ROOT):
        marker = "camera1_plastics_shareable_dataset/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            for root in (DATASET_ASSET_ROOT, DATASET_META_ROOT):
                candidate = root / relative
                if candidate.is_file():
                    return candidate
            return DATASET_ASSET_ROOT / relative
        return DATASET_ASSET_ROOT / Path(normalized).name

    path = Path(normalized)
    if path.is_file():
        return path

    if normalized.startswith(str(DATASET_META_ROOT)):
        return Path(normalized)

    for root in (DATASET_ASSET_ROOT, DATASET_META_ROOT):
        candidate = root / normalized
        if candidate.is_file():
            return candidate

    return DATASET_ASSET_ROOT / normalized


def find_audio_file(clip_id: str, audio_dirs: list[Path]) -> Optional[Path]:
    """Find an audio file for a clip id across known audio directories."""
    for audio_dir in audio_dirs:
        candidate = audio_dir / f"{clip_id}.mp3"
        if candidate.is_file():
            return candidate
        candidate = audio_dir / f"{clip_id}.wav"
        if candidate.is_file():
            return candidate
    return None


def list_audio_files(audio_dir: Path) -> list[Path]:
    """List audio files, including nested audio/audio/ if present."""
    dirs = [audio_dir, audio_dir / "audio"]
    files: list[Path] = []
    seen: set[str] = set()

    for directory in dirs:
        if not directory.is_dir():
            continue
        for pattern in ("*.mp3", "*.wav"):
            for path in directory.glob(pattern):
                if path.name not in seen:
                    seen.add(path.name)
                    files.append(path)

    return sorted(files, key=lambda p: p.name)
