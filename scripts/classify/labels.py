"""Load candidate labels from the plastics taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import TAXONOMY_JSON


def load_label_sets(taxonomy_json: Path = TAXONOMY_JSON) -> tuple[list[str], list[str]]:
    with taxonomy_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        df_labels = pd.DataFrame.from_dict(data, orient="index")
    elif isinstance(data, list):
        df_labels = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported taxonomy JSON format: {type(data)}")

    categories_list = df_labels.loc["categories", 0]
    categories_df = pd.DataFrame(categories_list)

    full_labels = [f"{row['name']}. {row['description']}" for _, row in categories_df.iterrows()]
    concise_labels = [row["name"] for _, row in categories_df.iterrows()]
    return full_labels, concise_labels
