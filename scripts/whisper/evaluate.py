#!/usr/bin/env python3
"""Evaluate transcription quality and category prediction performance."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SHARED_DIR = PACKAGE_DIR.parent / "shared"
for path in (SHARED_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from category_mapping import SemanticCategoryMapper, check_keyword_match, map_transcript_to_category
from config import (
    EVAL_SUMMARY_CSV,
    RESULTS_DIR,
    TRANSCRIBE_LORA_DIR,
    TRANSCRIBE_LORA_DENOISED_DIR,
    TRANSCRIBE_LORA_DENOISED_TRAINED_DIR,
    TRANSCRIBE_V1_DIR,
    TRANSCRIBE_V1_DENOISED_DIR,
    TRANSCRIBE_V2_DIR,
    TRANSCRIBE_V2_DENOISED_DIR,
)
from data_utils import build_comparison_df, load_timestamp_groups_df, load_transcriptions_from_dir


def _normalize_wer_text(text) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return [tok for tok in cleaned.split() if tok]


def word_error_rate(reference, hypothesis) -> float:
    ref = _normalize_wer_text(reference)
    hyp = _normalize_wer_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def success_rate(series_actual: pd.Series, series_predicted: pd.Series) -> float:
    valid = series_actual.notna() & series_predicted.notna()
    if valid.sum() == 0:
        return 0.0
    return float((series_actual[valid] == series_predicted[valid]).mean() * 100)


def category_summary(df: pd.DataFrame, predicted_col: str) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("category_names"):
        rows.append(
            {
                "category_names": category,
                "success_rate": success_rate(group["category_names"], group[predicted_col]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_version(
    version: str,
    transcribe_dir: Path,
    use_semantic: bool = True,
) -> dict:
    transcription_results = load_transcriptions_from_dir(transcribe_dir, version_label=version)
    comparison_df = build_comparison_df(transcription_results)

    comparison_df["matches_meta_phrase"] = comparison_df.apply(
        lambda row: check_keyword_match(row.get("transcription_text"), row.get("phrase")),
        axis=1,
    )
    comparison_df["predicted_category"] = comparison_df["transcription_text"].apply(
        map_transcript_to_category
    )

    wer_values = [
        word_error_rate(row.get("phrase"), row.get("transcription_text"))
        for _, row in comparison_df.iterrows()
    ]
    metrics = {
        "version": version,
        "num_samples": len(comparison_df),
        "phrase_match_rate": comparison_df["matches_meta_phrase"].mean() * 100,
        "wer": float(sum(wer_values) / len(wer_values) * 100) if wer_values else 0.0,
        "rule_based_success_rate": success_rate(
            comparison_df["category_names"], comparison_df["predicted_category"]
        ),
    }

    if use_semantic:
        mapper = SemanticCategoryMapper()
        comparison_df["predicted_category_semantic"] = comparison_df["transcription_text"].apply(
            mapper.predict
        )
        metrics["semantic_success_rate"] = success_rate(
            comparison_df["category_names"], comparison_df["predicted_category_semantic"]
        )

    return {"metrics": metrics, "comparison_df": comparison_df}


def build_full_summary() -> pd.DataFrame:
    timestamp_groups_df = load_timestamp_groups_df()
    timestamp_groups_df["predicted_category_from_phrase"] = timestamp_groups_df["meta2_phrase"].apply(
        map_transcript_to_category
    )
    phrase_baseline = success_rate(
        timestamp_groups_df["category_names"],
        timestamp_groups_df["predicted_category_from_phrase"],
    )

    rows = []
    version_dirs = {
        "v1": TRANSCRIBE_V1_DIR,
        "v2": TRANSCRIBE_V2_DIR,
        "lora": TRANSCRIBE_LORA_DIR,
        "v1_denoised": TRANSCRIBE_V1_DENOISED_DIR,
        "v2_denoised": TRANSCRIBE_V2_DENOISED_DIR,
        "lora_denoised": TRANSCRIBE_LORA_DENOISED_DIR,
        "lora_denoised_trained": TRANSCRIBE_LORA_DENOISED_TRAINED_DIR,
    }

    for version, directory in version_dirs.items():
        if not directory.is_dir() or not any(directory.glob("*.txt")):
            continue
        result = evaluate_version(version, directory, use_semantic=True)
        metrics = result["metrics"]
        if version == "lora_denoised_trained":
            audio_type = "denoised"
            lora_training = "denoised"
        elif version.endswith("_denoised"):
            audio_type = "denoised"
            lora_training = "original" if version.startswith("lora") else "n/a"
        else:
            audio_type = "original"
            lora_training = "original" if version == "lora" else "n/a"
        rows.append(
            {
                "version": version,
                "audio_type": audio_type,
                "lora_training": lora_training,
                "num_samples": metrics["num_samples"],
                "phrase_match_rate": round(metrics["phrase_match_rate"], 2),
                "wer": round(metrics.get("wer", 0.0), 2),
                "rule_based_success_rate": round(metrics["rule_based_success_rate"], 2),
                "semantic_success_rate": round(metrics.get("semantic_success_rate", 0.0), 2),
                "phrase_baseline_from_meta2": round(phrase_baseline, 2),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Whisper transcription performance.")
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "lora", "all"],
        default="all",
        help="Which transcription version to evaluate.",
    )
    parser.add_argument("--output-csv", type=Path, default=EVAL_SUMMARY_CSV)
    parser.add_argument("--save-details", action="store_true", help="Save per-sample comparison CSV.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.version == "all":
        summary = build_full_summary()
        if summary.empty:
            raise SystemExit("No transcription outputs found. Run scripts/transcribe.py first.")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(summary.to_string(index=False))
        print(f"\nSaved summary: {args.output_csv}")
        return

    version_dirs = {
        "v1": TRANSCRIBE_V1_DIR,
        "v2": TRANSCRIBE_V2_DIR,
        "lora": TRANSCRIBE_LORA_DIR,
        "v1_denoised": TRANSCRIBE_V1_DENOISED_DIR,
        "v2_denoised": TRANSCRIBE_V2_DENOISED_DIR,
        "lora_denoised": TRANSCRIBE_LORA_DENOISED_DIR,
        "lora_denoised_trained": TRANSCRIBE_LORA_DENOISED_TRAINED_DIR,
    }
    transcribe_dir = version_dirs[args.version]
    result = evaluate_version(args.version, transcribe_dir, use_semantic=True)
    metrics = result["metrics"]

    print(f"Version: {metrics['version']}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Phrase keyword match rate: {metrics['phrase_match_rate']:.2f}%")
    print(f"WER vs common phrase: {metrics.get('wer', 0.0):.2f}%")
    print(f"Rule-based category success: {metrics['rule_based_success_rate']:.2f}%")
    if "semantic_success_rate" in metrics:
        print(f"Semantic category success: {metrics['semantic_success_rate']:.2f}%")

    if args.save_details:
        detail_path = RESULTS_DIR / f"comparison_{args.version}.csv"
        result["comparison_df"].to_csv(detail_path, index=False)
        print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
