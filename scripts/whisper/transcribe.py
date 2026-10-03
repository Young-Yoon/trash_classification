#!/usr/bin/env python3
"""Run Whisper transcription inference (base/large/lora)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import whisper
from tqdm import tqdm

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import (
    AUDIO_CLEANED_DIR,
    AUDIO_DIR,
    LORA_CHECKPOINT_DIR,
    LORA_DENOISED_CHECKPOINT_DIR,
    MANIFEST_LORA,
    MANIFEST_LORA_DENOISED,
    MANIFEST_LORA_DENOISED_TRAINED,
    MANIFEST_V1,
    MANIFEST_V1_DENOISED,
    MANIFEST_V2,
    MANIFEST_V2_DENOISED,
    TRANSCRIBE_LORA_DIR,
    TRANSCRIBE_LORA_DENOISED_DIR,
    TRANSCRIBE_LORA_DENOISED_TRAINED_DIR,
    TRANSCRIBE_V1_DIR,
    TRANSCRIBE_V1_DENOISED_DIR,
    TRANSCRIBE_V2_DIR,
    TRANSCRIBE_V2_DENOISED_DIR,
    WHISPER_MODEL_LORA,
    WHISPER_MODEL_V1,
    WHISPER_MODEL_V2,
)
from data_utils import get_available_audio_files, save_manifest


def transcribe_with_openai_whisper(
    model_name: str,
    audio_files: list[Path],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    results: list[dict] = []
    for audio_path in tqdm(audio_files, desc=f"Transcribing ({model_name})"):
        output_txt = output_dir / f"{audio_path.stem}.txt"
        try:
            result = model.transcribe(str(audio_path))
            text = result["text"].strip()
            output_txt.write_text(text, encoding="utf-8")
            results.append(
                {
                    "audio_filename": audio_path.name,
                    "audio_path": str(audio_path),
                    "transcription_path": str(output_txt),
                    "status": "success",
                    "transcription_text_snippet": text[:100] + "..." if len(text) > 100 else text,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "audio_filename": audio_path.name,
                    "audio_path": str(audio_path),
                    "transcription_path": None,
                    "status": f"error: {exc}",
                    "transcription_text_snippet": None,
                }
            )
            print(f"Error transcribing {audio_path.name}: {exc}")

    return results


def transcribe_with_lora(
    checkpoint_dir: Path,
    audio_files: list[Path],
    output_dir: Path,
    base_model: str = WHISPER_MODEL_LORA,
) -> list[dict]:
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {base_model}")
    processor = WhisperProcessor.from_pretrained(base_model, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.to(device)
    model.eval()

    results: list[dict] = []
    for audio_path in tqdm(audio_files, desc="Transcribing (LoRA)"):
        output_txt = output_dir / f"{audio_path.stem}.txt"
        try:
            import librosa

            audio_array, _ = librosa.load(str(audio_path), sr=16000)
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features=input_features)

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            output_txt.write_text(text, encoding="utf-8")
            results.append(
                {
                    "audio_filename": audio_path.name,
                    "audio_path": str(audio_path),
                    "transcription_path": str(output_txt),
                    "status": "success",
                    "transcription_text_snippet": text[:100] + "..." if len(text) > 100 else text,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "audio_filename": audio_path.name,
                    "audio_path": str(audio_path),
                    "transcription_path": None,
                    "status": f"error: {exc}",
                    "transcription_text_snippet": None,
                }
            )
            print(f"Error transcribing {audio_path.name}: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper.")
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "lora"],
        default="v2",
        help="v1=base, v2=large, lora=fine-tuned adapter",
    )
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--denoised", action="store_true", help="Use denoised audio from audio/cleaned/")
    parser.add_argument(
        "--denoised-trained",
        action="store_true",
        help="Use LoRA checkpoint trained on denoised audio (implies --denoised).",
    )
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=LORA_CHECKPOINT_DIR,
        help="LoRA checkpoint directory",
    )
    args = parser.parse_args()

    if args.denoised_trained:
        args.denoised = True
        if args.checkpoint_dir == LORA_CHECKPOINT_DIR:
            args.checkpoint_dir = LORA_DENOISED_CHECKPOINT_DIR

    if args.denoised:
        audio_dir = args.audio_dir or AUDIO_CLEANED_DIR
    else:
        audio_dir = args.audio_dir or AUDIO_DIR

    audio_files = get_available_audio_files(audio_dir, args.test_limit)
    if not audio_files:
        raise SystemExit(f"No audio files found under {audio_dir}")

    if args.version == "v1":
        output_dir = TRANSCRIBE_V1_DENOISED_DIR if args.denoised else TRANSCRIBE_V1_DIR
        manifest = MANIFEST_V1_DENOISED if args.denoised else MANIFEST_V1
        model_name = WHISPER_MODEL_V1
        if args.denoised:
            model_name = f"{model_name}_denoised"
        results = transcribe_with_openai_whisper(WHISPER_MODEL_V1, audio_files, output_dir)
    elif args.version == "v2":
        output_dir = TRANSCRIBE_V2_DENOISED_DIR if args.denoised else TRANSCRIBE_V2_DIR
        manifest = MANIFEST_V2_DENOISED if args.denoised else MANIFEST_V2
        model_name = WHISPER_MODEL_V2
        if args.denoised:
            model_name = f"{model_name}_denoised"
        results = transcribe_with_openai_whisper(WHISPER_MODEL_V2, audio_files, output_dir)
    else:
        if args.denoised_trained:
            output_dir = TRANSCRIBE_LORA_DENOISED_TRAINED_DIR
            manifest = MANIFEST_LORA_DENOISED_TRAINED
            model_name = f"lora_denoised_trained:{args.checkpoint_dir}"
        else:
            output_dir = TRANSCRIBE_LORA_DENOISED_DIR if args.denoised else TRANSCRIBE_LORA_DIR
            manifest = MANIFEST_LORA_DENOISED if args.denoised else MANIFEST_LORA
            model_name = f"lora:{args.checkpoint_dir}"
            if args.denoised:
                model_name = f"{model_name}_denoised"
        results = transcribe_with_lora(args.checkpoint_dir, audio_files, output_dir)

    save_manifest(manifest, model_name, audio_dir, output_dir, results)
    success = sum(item["status"] == "success" for item in results)
    print(f"Completed transcription: success={success}/{len(results)}")


if __name__ == "__main__":
    main()
