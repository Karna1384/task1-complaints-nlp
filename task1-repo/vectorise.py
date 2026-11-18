"""
Stage 2 -- Vectorise the cleaned corpus two ways and compare them.

Task 1, Project: Data Analysis (CSEBDSEDA02)

Reads data/corpus_clean.parquet (written by pipeline.py) and produces:

  outputs/tfidf.npz          sparse TF-IDF matrix
  outputs/embeddings.npy     dense sentence embeddings
  outputs/texts.parquet      the tokenised text plus labels, for stage 3
  outputs/vectorisation.json the agreement score, saved so it is checkable

The two representations are compared by neighbourhood agreement: for a
sample of documents, take the ten nearest neighbours under each method and
measure the overlap. A low score means the corpus contains semantic
similarity that word overlap misses -- which is the argument for carrying
both rather than choosing one.

Run:  python vectorise.py                  (full corpus, ~25 min for embeddings)
      python vectorise.py --sample 5000    (fast smoke test)
      python vectorise.py --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS = DATA_DIR / "corpus_clean.parquet"

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.5      # terms in >50% of documents are boilerplate
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEIGHBOURS = 10
AGREEMENT_SAMPLE = 500
SEED = 42

# The corpus column to vectorise. pipeline.py writes the merged,
# lemmatised text here; `text_clean` is the pre-lemmatisation version.
TEXT_COL = "text_final"


def pick_text_column(df: pd.DataFrame) -> str:
    """Use the lemmatised/merged text when present, else the raw clean text."""
    if TEXT_COL in df.columns:
        return TEXT_COL
    if "text_clean" in df.columns:
        return "text_clean"
    raise SystemExit(
        "No text column found. Expected 'text_final' (written by pipeline.py) "
        "or 'text_clean'."
    )


def build_tfidf(texts: list[str]):
    """Sparse TF-IDF matrix. Fast, interpretable, blind to synonymy."""
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    print(f"TF-IDF: {matrix.shape}")
    return matrix, vec


def build_embeddings(texts: list[str]) -> np.ndarray:
    """Dense sentence embeddings. Captures meaning, costs much more time."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embeddings: {emb.shape}")
    return emb


def neighbourhood_agreement(tfidf, emb, k: int = NEIGHBOURS,
                            sample: int = AGREEMENT_SAMPLE) -> float:
    """Mean Jaccard overlap of each document's k nearest neighbours under both."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(emb), size=min(sample, len(emb)), replace=False)

    sim_t = cosine_similarity(tfidf[idx], tfidf)
    sim_e = emb[idx] @ emb.T
    for r, d in enumerate(idx):
        sim_t[r, d] = -np.inf
        sim_e[r, d] = -np.inf

    nt = np.argsort(-sim_t, axis=1)[:, :k]
    ne = np.argsort(-sim_e, axis=1)[:, :k]
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(nt, ne)]
    return float(np.mean(jac))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="Use only N documents (0 = all). For smoke tests.")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="Build TF-IDF only (no sentence-transformers needed).")
    args = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} not found. Run pipeline.py first.")

    df = pd.read_parquet(CORPUS)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)),
                       random_state=SEED).reset_index(drop=True)

    col = pick_text_column(df)
    texts = df[col].tolist()
    print(f"Documents: {len(texts):,}  (text column: {col})")

    tfidf, vec = build_tfidf(texts)
    sparse.save_npz(OUT_DIR / "tfidf.npz", tfidf)

    result = {
        "documents": int(len(texts)),
        "text_column": col,
        "tfidf_shape": [int(tfidf.shape[0]), int(tfidf.shape[1])],
        "tfidf_vocabulary": int(len(vec.get_feature_names_out())),
    }

    if not args.skip_embeddings:
        emb = build_embeddings(texts)
        np.save(OUT_DIR / "embeddings.npy", emb)
        agreement = neighbourhood_agreement(tfidf, emb)
        print(f"\nMean neighbourhood Jaccard: {agreement:.3f}")
        print("  Low score => much of the similarity is semantic, not lexical.")
        result["agreement"] = {
            "metric": "mean neighbourhood Jaccard overlap",
            "neighbours": NEIGHBOURS,
            "sample": min(AGREEMENT_SAMPLE, len(texts)),
            "embedding_model": EMBEDDING_MODEL,
            "value": round(agreement, 3),
        }
    else:
        result["agreement"] = None
        print("\nSkipped embeddings (--skip-embeddings).")

    (OUT_DIR / "vectorisation.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT_DIR/'vectorisation.json'}")

    # Stage 3 needs the text plus whichever label columns exist.
    keep = [col]
    for c in ("Product", "Issue"):
        if c in df.columns:
            keep.append(c)
    df[keep].to_parquet(OUT_DIR / "texts.parquet")
    print(f"Wrote {OUT_DIR/'texts.parquet'}")


if __name__ == "__main__":
    main()
