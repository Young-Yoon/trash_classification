"""Evaluate saved image classification JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import CLASSIFICATION_RESULTS_DIR, CLASSIFICATION_SUMMARY_CSV, META2_CSV
from presets import ALL_PRESETS, PRESETS, TRAINED_PRESETS


def predicted_label_name(raw_label: str) -> str:
    return raw_label.split(".")[0].strip()


def result_json_path(row: pd.Series, model_output_dir: Path) -> Path:
    copied_frame_path = row.get("copied_frame_path")
    if pd.isna(copied_frame_path):
        return model_output_dir / "missing.json"

    frame_name = Path(str(copied_frame_path)).name.replace(".jpg", "").replace(".jpeg", "")
    frame_id = f"{row['gx_id']}_{frame_name}"
    return model_output_dir / f"{frame_id}.json"


def calculate_success_rates(
    model_output_dir: Path,
    meta_df: pd.DataFrame,
    *,
    top_n_values: list[int] | None = None,
) -> dict[str, float]:
    if top_n_values is None:
        top_n_values = [1, 3, 5]

    predicted_top_n: dict[int, list[bool]] = {n: [] for n in top_n_values}

    if not model_output_dir.is_dir():
        return {f"top_{n}": 0.0 for n in top_n_values}

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=model_output_dir.name):
        output_filepath = result_json_path(row, model_output_dir)
        if not output_filepath.is_file():
            continue

        try:
            with output_filepath.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            for n in top_n_values:
                predicted_top_n[n].append(False)
            continue

        if isinstance(payload, dict) and "error" in payload:
            for n in top_n_values:
                predicted_top_n[n].append(False)
            continue

        if isinstance(payload, dict) and "label" in payload:
            all_predicted_labels = [predicted_label_name(payload["label"])]
        elif isinstance(payload, list):
            all_predicted_labels = [
                predicted_label_name(item["label"])
                for item in payload
                if isinstance(item, dict) and "label" in item
            ]
        else:
            for n in top_n_values:
                predicted_top_n[n].append(False)
            continue

        actual_label = row["category_name"]
        for n in top_n_values:
            predicted_top_n[n].append(actual_label in all_predicted_labels[:n])

    return {
        f"top_{n}": (sum(predicted_top_n[n]) / len(predicted_top_n[n]) * 100 if predicted_top_n[n] else 0.0)
        for n in top_n_values
    }


def build_summary(
    preset_names: list[str],
    meta_csv: Path = META2_CSV,
    results_dir: Path = CLASSIFICATION_RESULTS_DIR,
    *,
    preset_map: dict | None = None,
) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_csv)
    preset_lookup = preset_map or ALL_PRESETS
    rows = []
    for preset_name in preset_names:
        preset = preset_lookup[preset_name]
        output_dir_name = preset["output_dir"]
        metrics = calculate_success_rates(results_dir / output_dir_name, meta_df)
        rows.append(
            {
                "preset": preset_name,
                "output_dir": output_dir_name,
                "top_1_success_rate": round(metrics.get("top_1", 0.0), 2),
                "top_3_success_rate": round(metrics.get("top_3", 0.0), 2),
                "top_5_success_rate": round(metrics.get("top_5", 0.0), 2),
            }
        )
    return pd.DataFrame(rows)
