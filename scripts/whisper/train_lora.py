#!/usr/bin/env python3
"""Fine-tune Whisper Large V2 with LoRA on local audio + phrase labels."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import librosa
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import (
    LORA_CHECKPOINT_DIR,
    LORA_DENOISED_CHECKPOINT_DIR,
    TRAINING_METADATA_CSV,
    TRAINING_METADATA_DENOISED_CSV,
    WHISPER_MODEL_LORA,
)
from data_utils import build_training_metadata


@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class WhisperLoRATrainer(Seq2SeqTrainer):
    """Avoid PEFT seq2seq wrapper passing input_ids=None into Whisper."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model.base_model(
            input_features=inputs.get("input_features"),
            labels=inputs.get("labels"),
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def build_prepared_dataset(processor: WhisperProcessor, metadata_csv: Path) -> Dataset:
    df = pd.read_csv(metadata_csv)
    records = [
        {"audio": row["file_name"], "sentence": row["sentence"]}
        for _, row in df.iterrows()
        if pd.notna(row["sentence"]) and Path(row["file_name"]).is_file()
    ]
    if not records:
        raise RuntimeError(f"No valid training rows found in {metadata_csv}")

    raw_dataset = Dataset.from_list(records)

    def prepare_dataset(batch):
        audio_array, _ = librosa.load(batch["audio"], sr=16000)
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    return raw_dataset.map(
        prepare_dataset,
        remove_columns=raw_dataset.column_names,
        desc="Preparing dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Whisper LoRA adapter.")
    parser.add_argument("--metadata-csv", type=Path, default=TRAINING_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=LORA_CHECKPOINT_DIR)
    parser.add_argument("--model-name", default=WHISPER_MODEL_LORA)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--rebuild-metadata", action="store_true")
    parser.add_argument(
        "--denoised",
        action="store_true",
        help="Train on denoised audio from audio/cleaned/ and save to a separate checkpoint dir.",
    )
    args = parser.parse_args()

    if args.denoised:
        if args.metadata_csv == TRAINING_METADATA_CSV:
            args.metadata_csv = TRAINING_METADATA_DENOISED_CSV
        if args.output_dir == LORA_CHECKPOINT_DIR:
            args.output_dir = LORA_DENOISED_CHECKPOINT_DIR

    if args.rebuild_metadata or not args.metadata_csv.is_file():
        build_training_metadata(args.metadata_csv, use_denoised=args.denoised)

    use_cuda = torch.cuda.is_available()
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if use_cuda else None

    processor = WhisperProcessor.from_pretrained(args.model_name, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )
    if use_cuda:
        model = prepare_model_for_kbit_training(model)

    prepared_dataset = build_prepared_dataset(processor, args.metadata_csv)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=use_cuda,
        remove_unused_columns=False,
        label_names=["labels"],
        use_cpu=not use_cuda,
        report_to="none",
        save_steps=max(10, args.max_steps // 5),
        logging_steps=max(5, args.max_steps // 10),
    )

    trainer = WhisperLoRATrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
