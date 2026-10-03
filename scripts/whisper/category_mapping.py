"""Rule-based and semantic category mapping for transcriptions."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from config import COMMON_PHRASES_CSV


def map_transcript_to_category(transcript_text: Optional[str]) -> str:
    """Map transcription text to a plastic category using notebook rule logic."""
    if pd.isna(transcript_text):
        return "Unknown"

    text = str(transcript_text).lower()

    if text == "non -deposit garbage bag":
        return "Non-Deposit"
    if text == "dissolt.":
        return "Deposit #1 Bottles"
    if text == "fly plastic.":
        return "Black Plastic Containers"
    if text == "under posit one down to positive":
        return "Non-Deposit"
    if text in {"no.", "no"}:
        return "Containers #1/Clamshells"
    if text in {"one deposit.", "one deposit"}:
        return "Deposit #1 Bottles"
    if "other one" in text or "lid" in text:
        return "Containers #1/Clamshells"
    if "five" in text or "5" in text:
        return "Containers #5/PP"
    if text == "unnumbered.":
        return "Other Unnumbered Containers & Fragments"
    if text == "other plastic.":
        return "Other Plastics"
    if "two" in text and "color" in text:
        return "Containers #2 Colored"
    if text == "number one.":
        return "Deposit #1 Bottles"

    if ("one" in text or "1" in text) and "non" in text:
        return "Other #1 Bottles"
    if "non" in text and "deposit" in text:
        return "Non-Deposit"
    if "garbage bag" in text:
        return "Garbage Bags"
    if "black" in text:
        return "Black Plastic Containers"
    if "clam" in text and "shell" in text:
        return "Containers #1/Clamshells"
    if "two natural" in text:
        return "Containers #2 Natural"
    if "two color" in text or "two colored" in text:
        return "Containers #2 Colored"

    if any(token in text for token in ("three", "3", "seven", "7", "four", "4", "six", "6")):
        return "Containers #3-7 (Not #5)"
    if "five" in text or "5" in text:
        return "Containers #5/PP"

    if "number one" in text:
        if "deposit" in text:
            return "Deposit #1 Bottles"
        return "Other #1 Bottles"
    if "deposit" in text:
        return "Deposit #1 Bottles"

    if "plastic film" in text or "film plastic" in text or "film" in text:
        return "Other Plastic Film"
    if "other plastic" in text:
        return "Other Plastics"
    if "unnumbered" in text:
        return "Other Unnumbered Containers & Fragments"
    if "bulky rigid" in text:
        return "Other Rigid/Bulky Plastics"
    if "if you find numbers" in text:
        return "Other Unnumbered Containers & Fragments"
    if "plastic" in text:
        return "Plastic"
    if "one" in text:
        return "Containers #1/Clamshells"

    return "Unknown"


def check_keyword_match(transcription_text: Optional[str], target_phrase: Optional[str]) -> bool:
    if pd.isna(transcription_text) or pd.isna(target_phrase):
        return False
    transcription_words = set(str(transcription_text).lower().split())
    target_words = set(str(target_phrase).lower().split())
    return bool(transcription_words.intersection(target_words))


class SemanticCategoryMapper:
    """Sentence-transformer based category mapper."""

    def __init__(self, common_phrases_csv=COMMON_PHRASES_CSV, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.common_phrases_df = self._load_common_phrases(common_phrases_csv)
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.common_phrases_df["normalized_phrase"].tolist(),
            convert_to_tensor=True,
        )

    @staticmethod
    def _load_common_phrases(path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "example_categories" in df.columns:
            df["parsed_categories"] = df["example_categories"].apply(
                lambda value: json.loads(str(value).replace("'", '"'))
                if isinstance(value, str) and value.startswith("[")
                else value
            )
            df["target_category"] = df["parsed_categories"].apply(
                lambda value: value[0] if isinstance(value, list) and value else value
            )
        else:
            df["target_category"] = None
        return df.dropna(subset=["normalized_phrase", "target_category"]).reset_index(drop=True)

    def predict(self, transcript_text: Optional[str], top_k: int = 1) -> str:
        if pd.isna(transcript_text) or not str(transcript_text).strip():
            return "Unknown"

        transcript_embedding = self.model.encode(str(transcript_text), convert_to_tensor=True)
        cosine_scores = self.util.cos_sim(transcript_embedding, self.embeddings)[0]
        top_results = np.argpartition(-cosine_scores.cpu().numpy(), top_k)[:top_k]
        if len(top_results) == 0:
            return "Unknown"
        return self.common_phrases_df.iloc[int(top_results[0])]["target_category"]
