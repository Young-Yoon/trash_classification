#!/usr/bin/env python3
"""Extract audio from timestamp video clips listed in meta2.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import AUDIO_DIR, VIDEO_CLIPS_DIR
from data_utils import load_meta_df
from paths import clip_id_from_path, resolve_dataset_path


def extract_audio_from_video(video_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return "skipped_exists"

    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            return "no_audio"
        video.audio.write_audiofile(str(output_path), logger=None)
    return "success"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MP3 audio from dataset video clips.")
    parser.add_argument("--output-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--test-limit", type=int, default=None, help="Process only first N rows.")
    args = parser.parse_args()

    meta_df = load_meta_df()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    rows = meta_df.head(args.test_limit) if args.test_limit else meta_df

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Extracting audio"):
        clip_suffix = row.get("timestamp_clip_path")
        if pd.isna(clip_suffix):
            continue

        video_path = resolve_dataset_path(clip_suffix)
        if video_path is None or not video_path.is_file():
            alt = VIDEO_CLIPS_DIR / Path(str(clip_suffix)).name
            if alt.is_file():
                video_path = alt
            else:
                nested = VIDEO_CLIPS_DIR / clip_id_from_path(clip_suffix)
                nested = nested.with_suffix(".mp4")
                if nested.is_file():
                    video_path = nested
                else:
                    parent = VIDEO_CLIPS_DIR / str(clip_suffix).split("/")[-2] if "/" in str(clip_suffix) else None
                    candidate = None
                    if parent and parent.is_dir():
                        candidate = parent / f"{clip_id_from_path(clip_suffix)}.mp4"
                    if candidate and candidate.is_file():
                        video_path = candidate
                    else:
                        extracted.append(
                            {
                                "clip_id": clip_id_from_path(clip_suffix),
                                "video_path": str(clip_suffix),
                                "status": "video_not_found",
                            }
                        )
                        continue

        clip_id = video_path.stem
        output_path = output_dir / f"{clip_id}.mp3"
        status = extract_audio_from_video(video_path, output_path)
        extracted.append(
            {
                "clip_id": clip_id,
                "video_path": str(video_path),
                "audio_path": str(output_path),
                "status": status,
            }
        )

    success = sum(item["status"] == "success" for item in extracted)
    skipped = sum(item["status"] == "skipped_exists" for item in extracted)
    print(f"Done. success={success}, skipped={skipped}, total={len(extracted)}")


if __name__ == "__main__":
    main()
