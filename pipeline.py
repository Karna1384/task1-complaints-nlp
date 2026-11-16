"""
Stage 1 -- Load, clean, lemmatise and collocate the complaint corpus.

Task 1, Project: Data Analysis (CSEBDSEDA02)
Corpus: CFPB consumer complaints (Kaggle: ashwinik/consumer-complaints-financial-products)

Pipeline within this stage:
  1. load the raw CSV
  2. drop rows with no published narrative, then exact duplicates
  3. clean the text (lowercase, strip URLs, redactions, punctuation)
  4. lemmatise and remove stop words  (spaCy)
  5. merge frequent two-word phrases into single tokens  (scikit-learn)

Step 5 is what turns "student" + "loan" into the single token `student_loan`
that appears in the saved results. Without it the topic term lists would be
built from unigrams only.

Produces data/corpus_clean.parquet, which vectorise.py reads.

Run:  python pipeline.py
      python pipeline.py --sample 3000     (fast smoke test)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

# Lemmatisation: which parts of speech to keep, and the shortest term allowed.
KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ"}
MIN_LEMMA_LEN = 3

# Collocations: a two-word phrase becomes one token when it appears in at
# least this many documents. Frequent bigrams carry topic meaning
# (`credit_report`, `debt_collection`); rare ones are noise.
BIGRAM_MIN_DOCS = 20

SEED = 42

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


# Domain words that are near-universal in this corpus and so carry no
# topical signal: every document is a complaint about an account or a
# company. Removing them sharpens the topics.
DOMAIN_STOPWORDS = {
    "complaint", "complaints", "company", "consumer", "account", "xxxx",
    "xx", "said", "told", "day", "time", "year", "month",
}


def load_spacy():
    """Load the spaCy model, with a clear error if it is missing."""
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is not installed. Run:\n"
            "  pip install -r requirements.txt\n"
            "  python -m spacy download en_core_web_sm"
        ) from exc
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError as exc:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' not found. Run:\n"
            "  python -m spacy download en_core_web_sm"
        ) from exc


def lemmatise(nlp, texts: list[str]) -> list[list[str]]:
    """Tokenise, lemmatise and keep content words only."""
    out: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=256, n_process=1):
        out.append([
            t.lemma_
            for t in doc
            if t.is_alpha
            and not t.is_stop
            and len(t.lemma_) >= MIN_LEMMA_LEN
            and t.lemma_ not in DOMAIN_STOPWORDS
            and t.pos_ in KEEP_POS
        ])
    return out


def merge_collocations(token_lists: list[list[str]]) -> tuple[list[list[str]], dict]:
    """
    Join frequent two-word phrases into single tokens.

    A candidate bigram is rejected when the *whole* phrase already exists as a
    single token in the corpus. Without that guard a second pass produces
    tokens such as `wells_fargo home_mortgage` that never appeared in the
    source text.
    """
    joined = [" ".join(toks) for toks in token_lists]
    token_types = {t for toks in token_lists for t in toks}

    finder = CountVectorizer(
        ngram_range=(2, 2), min_df=BIGRAM_MIN_DOCS, token_pattern=r"(?u)\b\w+\b"
    )
    finder.fit(joined)

    phrases: dict[str, str] = {}
    for phrase in finder.get_feature_names_out():
        merged = phrase.replace(" ", "_")
        if merged in token_types:      # would recreate an existing token
            continue
        phrases[phrase] = merged

    def merge(tokens: list[str]) -> list[str]:
        out, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1:
                key = f"{tokens[i]} {tokens[i + 1]}"
                if key in phrases and phrases[key] not in token_types:
                    out.append(phrases[key])
                    i += 2
                    continue
            out.append(tokens[i])
            i += 1
        return out

    merged_lists = [merge(t) for t in token_lists]
    return merged_lists, phrases


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="Use only N documents (0 = all). For smoke tests.")
    args = ap.parse_args()

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

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=SEED).reset_index(drop=True)
        print(f"  sampled:             {len(df):,}")

    # --- step 4: lemmatise -------------------------------------------------
    nlp = load_spacy()
    print("\nLemmatising ...")
    df["tokens"] = lemmatise(nlp, df["text_clean"].tolist())
    before = len(df)
    df = df[df["tokens"].map(len) >= 5].reset_index(drop=True)
    print(f"  under 5 content words: {before - len(df):,}")
    print(f"  kept:                  {len(df):,}")

    # --- step 5: merge frequent two-word phrases ---------------------------
    print("\nMerging collocations ...")
    df["tokens"], phrases = merge_collocations(df["tokens"].tolist())
    print(f"  bigrams merged:        {len(phrases):,}")
    print(f"  examples:              {', '.join(list(phrases.values())[:8])}")

    df["text_final"] = df["tokens"].map(" ".join)

    vocab = len({t for toks in df["tokens"] for t in toks})
    double_merged = {t for toks in df["tokens"] for t in toks if t.count("_") > 1}
    print(f"  vocabulary:            {vocab:,}")
    print(f"  double-merged tokens:  {len(double_merged)}  (must be 0)")

    out = DATA_DIR / "corpus_clean.parquet"
    df.to_parquet(out)

    print(f"\nAnalysed corpus: {len(df):,} documents")
    print(f"Median length:   {df['n_words'].median():.0f} words")
    print(f"Products:        {df['Product'].nunique()}")
    print(f"Issues:          {df['Issue'].nunique()}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
