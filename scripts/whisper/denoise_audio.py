#!/usr/bin/env python3
"""Denoise audio files using noisereduce (same approach as audio.ipynb)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import noisereduce as nr
import soundfile as sf
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import AUDIO_CLEANED_DIR, AUDIO_DIR
from data_utils import get_available_audio_files


def denoise_file(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data, rate = librosa.load(str(input_path), sr=sample_rate)
    cleaned = nr.reduce_noise(y=data, sr=rate, stationary=False)
    sf.write(str(output_path), cleaned, rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise audio files with noisereduce.")
    parser.add_argument("--input-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUDIO_CLEANED_DIR)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audio_files = get_available_audio_files(args.input_dir, args.test_limit)
    if not audio_files:
        raise SystemExit(f"No audio files found under {args.input_dir}")

    processed = skipped = 0
    for audio_path in tqdm(audio_files, desc="Denoising audio"):
        output_path = args.output_dir / f"{audio_path.stem}.wav"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        denoise_file(audio_path, output_path)
        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, total={len(audio_files)}")
    print(f"Cleaned audio saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
