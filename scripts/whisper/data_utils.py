"""Shared data loading utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from config import (
    AUDIO_CLEANED_DIR,
    AUDIO_DIR,
    META2_CSV,
    TIMESTAMP_GROUPS_CSV,
    TRAINING_METADATA_CSV,
    TRAINING_METADATA_DENOISED_CSV,
)
from paths import clip_id_from_path, find_audio_file, list_audio_files


def load_meta_df() -> pd.DataFrame:
    return pd.read_csv(META2_CSV)


def load_timestamp_groups_df() -> pd.DataFrame:
    df = pd.read_csv(TIMESTAMP_GROUPS_CSV)
    df["clip_id"] = df["clip_path"].apply(clip_id_from_path)
    return df


def build_training_metadata(
    output_csv: Path = TRAINING_METADATA_CSV,
    use_denoised: bool = False,
) -> pd.DataFrame:
    """Create training metadata CSV mapping audio files to ground-truth phrases."""
    timestamp_groups_df = load_timestamp_groups_df()
    if use_denoised:
        audio_dirs = [AUDIO_CLEANED_DIR]
    else:
        audio_dirs = [AUDIO_DIR, AUDIO_DIR / "audio"]

    rows = []
    for _, row in timestamp_groups_df.iterrows():
        clip_id = row["clip_id"]
        if not clip_id or pd.isna(row.get("meta2_phrase")):
            continue
        audio_path = find_audio_file(clip_id, audio_dirs)
        if audio_path is None:
            continue
        rows.append(
            {
                "clip_id": clip_id,
                "file_name": str(audio_path),
                "sentence": row["meta2_phrase"],
                "category_names": row.get("category_names"),
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved training metadata: {output_csv} ({len(df)} rows)")
    return df


def load_transcriptions_from_dir(
    transcribe_dir: Path,
    audio_dir: Path = AUDIO_DIR,
    version_label: str = "unknown",
) -> list[dict]:
    """Reconstruct transcription results from saved .txt files."""
    if not transcribe_dir.is_dir():
        raise FileNotFoundError(f"Transcription directory not found: {transcribe_dir}")

    results: list[dict] = []
    txt_files = sorted(transcribe_dir.glob("*.txt"))

    for txt_path in tqdm(txt_files, desc=f"Loading {version_label} transcriptions"):
        clip_id = txt_path.stem
        audio_path = find_audio_file(clip_id, [audio_dir, audio_dir / "audio"])
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
            results.append(
                {
                    "audio_filename": f"{clip_id}.mp3",
                    "audio_path": str(audio_path) if audio_path else None,
                    "transcription_path": str(txt_path),
                    "status": "success",
                    "transcription_text": text,
                    "transcription_text_snippet": text[:100] + "..." if len(text) > 100 else text,
                }
            )
        except OSError as exc:
            results.append(
                {
                    "audio_filename": f"{clip_id}.mp3",
                    "audio_path": str(audio_path) if audio_path else None,
                    "transcription_path": str(txt_path),
                    "status": f"error: {exc}",
                    "transcription_text": None,
                    "transcription_text_snippet": None,
                }
            )

    return results


def build_comparison_df(transcription_results: Iterable[dict]) -> pd.DataFrame:
    """Merge transcription results with timestamp group labels."""
    timestamp_groups_df = load_timestamp_groups_df()
    transcription_df = pd.DataFrame(list(transcription_results))
    transcription_df["clip_id"] = transcription_df["audio_filename"].apply(
        lambda name: Path(str(name)).stem
    )

    merged = pd.merge(
        transcription_df,
        timestamp_groups_df[
            [
                "clip_id",
                "meta2_phrase",
                "category_names",
                "supercategory_names",
                "begin_time",
                "end_time",
            ]
        ].rename(
            columns={
                "meta2_phrase": "phrase",
                "supercategory_names": "supercategory_name",
            }
        ),
        on="clip_id",
        how="left",
    )
    return merged


def save_manifest(
    manifest_path: Path,
    whisper_model: str,
    audio_input_dir: Path,
    transcribe_output_dir: Path,
    transcription_results: list[dict],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "whisper_model": whisper_model,
        "audio_input_directory": str(audio_input_dir),
        "transcription_output_directory": str(transcribe_output_dir),
        "transcription_summary": transcription_results,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


def get_available_audio_files(audio_dir: Path = AUDIO_DIR, limit: Optional[int] = None) -> list[Path]:
    files = list_audio_files(audio_dir)
    if limit is not None:
        return files[:limit]
    return files
