"""
Stage 1 -- Load and clean the complaint corpus.

Task 1, Project: Data Analysis (CSEBDSEDA02)
Corpus: CFPB consumer complaints (Kaggle: ashwinik/consumer-complaints-financial-products)

This script produces data/corpus_clean.parquet, which vectorise.py reads.

Run:  python pipeline.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- the only block a reader needs to edit.
# ---------------------------------------------------------------------------

# Folder holding Consumer_Complaints.csv (download it from Kaggle first).
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Output folder for intermediate and final artefacts.
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))

RAW_CSV = DATA_DIR / "Consumer_Complaints.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# The free-text column to analyse.
TEXT_COL = "Consumer complaint narrative"

# Minimum content words for a document to be kept. The corpus median is 137
# words and only 0.1% fall below five, so this removes a negligible tail.
MIN_WORDS = 10

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# CFPB redacts names, account numbers and dates as runs of X. Present in
# 85.8% of documents, so stripping them is load-bearing, not cosmetic.
REDACTION_RE = re.compile(r"\bx{2,}\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip redactions/URLs/punctuation, collapse whitespace."""
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = REDACTION_RE.sub(" ", t)
    t = NON_ALPHA_RE.sub(" ", t)
    return WHITESPACE_RE.sub(" ", t).strip()


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(
            f"Input not found: {RAW_CSV}\n"
            f"Download Consumer_Complaints.csv from Kaggle into {DATA_DIR}/ "
            f"or set DATA_DIR."
        )

    print(f"Reading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"  rows in file:        {len(df):,}")

    if TEXT_COL not in df.columns:
        raise SystemExit(
            f"Column {TEXT_COL!r} not found. Available columns:\n"
            + "\n".join(f"  - {c}" for c in df.columns)
        )

    # Only rows carrying a published narrative can be analysed. Publication
    # requires the complainant's consent, which is why ~83% of the file is empty.
    df = df[df[TEXT_COL].notna()].copy()
    print(f"  with a narrative:    {len(df):,}")

    before = len(df)
    df = df.drop_duplicates(subset=TEXT_COL, keep="first").reset_index(drop=True)
    print(f"  duplicates removed:  {before - len(df):,}")

    df["text_clean"] = df[TEXT_COL].map(clean_text)
    df["n_words"] = df["text_clean"].str.split().str.len()

    before = len(df)
    df = df[df["n_words"] >= MIN_WORDS].reset_index(drop=True)
    print(f"  under {MIN_WORDS} words:     {before - len(df):,}")

    out = DATA_DIR / "corpus_clean.parquet"
    df.to_parquet(out)

    print(f"\nAnalysed corpus: {len(df):,} documents")
    print(f"Median length:   {df['n_words'].median():.0f} words")
    print(f"Products:        {df['Product'].nunique()}")
    print(f"Issues:          {df['Issue'].nunique()}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
