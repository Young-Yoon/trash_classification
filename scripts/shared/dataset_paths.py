"""Resolve dataset asset paths from meta2 CSV values."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATASET_ROOT, LEGACY_COLAB_PREFIX, LEGACY_EXTERNAL_ROOT
from paths import resolve_dataset_path


def resolve_meta_asset_path(path_value: Optional[str], dataset_root: Path = DATASET_ROOT) -> Optional[Path]:
    if pd.isna(path_value) or not path_value:
        return None

    resolved = resolve_dataset_path(path_value)
    if resolved is not None and resolved.is_file():
        return resolved

    normalized = str(path_value)
    if normalized.startswith(LEGACY_EXTERNAL_ROOT):
        normalized = normalized.replace(LEGACY_EXTERNAL_ROOT, LEGACY_COLAB_PREFIX)

    dataset_root_str = str(dataset_root)
    if normalized.startswith(dataset_root_str):
        relative_part = normalized[len(dataset_root_str) :].lstrip("/\\")
        candidate = dataset_root / relative_part
        if candidate.is_file():
            return candidate

    candidate = dataset_root / Path(normalized).name
    if candidate.is_file():
        return candidate

    return resolved


def frame_result_id(gx_id: str, frame_path: Path) -> str:
    frame_base_name = frame_path.name.replace(".jpg", "").replace(".jpeg", "")
    return f"{gx_id}_{frame_base_name}"
