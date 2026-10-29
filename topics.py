"""
Stage 3 -- Extract topics by two methods and validate them externally.

Task 1, Project: Data Analysis (CSEBDSEDA02)

Methods:
  LDA  -- probabilistic mixture model on a bag-of-words matrix. The topic
          count is selected by a coherence sweep rather than assumed.
  NMF  -- linear decomposition on the TF-IDF matrix.

Validation:
  * NPMI coherence, implemented directly so the metric is auditable.
  * Raw term frequency (model-free baseline).
  * Cross-check of each topic against the corpus's own Product and Issue
    labels, which no model has seen.

Produces outputs/results_summary.json with every figure quoted in the report.

Run:  python topics.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))
TEXTS = OUT_DIR / "texts.parquet"

TOPIC_CANDIDATES = [5, 8, 10, 12, 15, 20]
N_TOP_WORDS = 10
SEED = 42

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.5

LDA_MAX_ITER = 15
NMF_MAX_ITER = 400


# ---------------------------------------------------------------------------
# Coherence -- normalised PMI, computed directly.
#
# The normalisation matters: an unnormalised co-occurrence count scores a
# pair of very common words highly merely for being common, which saturates
# the metric and leaves the topic-count sweep unable to discriminate.
# ---------------------------------------------------------------------------

def build_postings(doc_sets: list[set]) -> dict[str, set]:
    postings: dict[str, set] = {}
    for i, s in enumerate(doc_sets):
        for w in s:
            postings.setdefault(w, set()).add(i)
    return postings


def npmi(a: str, b: str, postings: dict, n_docs: int) -> float:
    pa, pb = postings.get(a), postings.get(b)
    if not pa or not pb:
        return 0.0
    small, large = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    cooc = sum(1 for d in small if d in large)
    if cooc == 0:
        return 0.0
    p_ij = cooc / n_docs
    denom = -np.log(p_ij)
    if denom <= 0:
        return 0.0
    return max(0.0, float(np.log(p_ij / ((len(pa) / n_docs) * (len(pb) / n_docs))) / denom))


def coherence(term_lists: list[list[str]], postings: dict, n_docs: int) -> float:
    scores = []
    for terms in term_lists:
        vals = [npmi(terms[i], terms[j], postings, n_docs)
                for i in range(len(terms)) for j in range(i + 1, len(terms))]
        scores.append(float(np.mean(vals)) if vals else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def top_terms(components, features, n: int = N_TOP_WORDS) -> list[list[str]]:
    return [[str(features[j]) for j in np.argsort(-c)[:n]] for c in components]


def agreement(a: list[list[str]], b: list[list[str]], n: int = N_TOP_WORDS) -> dict:
    """Greedy best-match Jaccard overlap between two models' topic vocabularies."""
    sa, sb = [set(t[:n]) for t in a], [set(t[:n]) for t in b]
    used, scores = set(), []
    for x in sa:
        best, bj = 0.0, -1
        for j, y in enumerate(sb):
            if j in used:
                continue
            jac = len(x & y) / len(x | y) if (x | y) else 0.0
            if jac > best:
                best, bj = jac, j
        if bj >= 0:
            used.add(bj)
            scores.append(round(best, 3))
    return {
        "mean": round(float(np.mean(scores)), 3) if scores else 0.0,
        "per_topic": scores,
        "matched": len(scores),
    }


def crosstab(df: pd.DataFrame, dominant: np.ndarray, label_col: str, k: int) -> list[dict]:
    """Which human-applied label does each topic concentrate in?"""
    if label_col not in df.columns:
        return []
    labels = df[label_col].astype(str).to_numpy()
    rows = []
    for t in range(k):
        mask = dominant == t
        if mask.sum() == 0:
            continue
        vals, counts = np.unique(labels[mask], return_counts=True)
        rows.append({
            "topic": int(t),
            "documents": int(mask.sum()),
            "top_label": str(vals[np.argmax(counts)]),
            "share": round(float(counts.max() / mask.sum()), 3),
        })
    return rows


# ---------------------------------------------------------------------------

def main() -> None:
    if not TEXTS.exists():
        raise SystemExit(f"{TEXTS} not found. Run vectorise.py first.")

    df = pd.read_parquet(TEXTS)
    texts = df["text_clean"].tolist()
    print(f"Documents: {len(texts):,}")

    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF, ngram_range=(1, 2), sublinear_tf=True,
    )
    tfidf = vec.fit_transform(texts)
    features = vec.get_feature_names_out()
    bow = CountVectorizer(vocabulary=features).fit_transform(texts)
    print(f"TF-IDF: {tfidf.shape}")

    doc_sets = [set(t.split()) for t in texts]
    postings = build_postings(doc_sets)
    n_docs = len(doc_sets)

    # --- sweep the topic count -------------------------------------------
    sweep = []
    for k in TOPIC_CANDIDATES:
        m = LatentDirichletAllocation(
            n_components=k, max_iter=10,
            learning_method="batch", random_state=SEED,
        ).fit(bow)
        s = coherence(top_terms(m.components_, features), postings, n_docs)
        sweep.append({"k": k, "coherence": round(s, 4)})
        print(f"  k={k:2d}  coherence={s:.4f}")

    best_k = max(sweep, key=lambda x: x["coherence"])["k"]
    print(f"Selected k = {best_k}")

    # --- fit both models -------------------------------------------------
    lda = LatentDirichletAllocation(
        n_components=best_k, max_iter=LDA_MAX_ITER,
        learning_method="batch", random_state=SEED,
    ).fit(bow)
    lda_doc = lda.transform(bow)

    nmf = NMF(
        n_components=best_k, init="nndsvda",
        max_iter=NMF_MAX_ITER, random_state=SEED,
    ).fit(tfidf)

    lda_t, nmf_t = top_terms(lda.components_, features), top_terms(nmf.components_, features)

    dominant = np.argmax(lda_doc, axis=1)
    freq = Counter(w for s in doc_sets for w in s)

    result = {
        "documents": int(len(df)),
        "vocabulary": int(len(features)),
        "coherence_sweep": sweep,
        "selected_k": best_k,
        "lda_topics": [
            {"id": i, "terms": t, "size": int((dominant == i).sum())}
            for i, t in enumerate(lda_t)
        ],
        "nmf_topics": [{"id": i, "terms": t} for i, t in enumerate(nmf_t)],
        "topic_agreement": agreement(lda_t, nmf_t),
        "topic_vs_product": crosstab(df, dominant, "Product", best_k),
        "topic_vs_issue": crosstab(df, dominant, "Issue", best_k),
        "top_terms_raw": [[w, c] for w, c in freq.most_common(20)],
        "issue_count": int(df["Issue"].nunique()) if "Issue" in df.columns else 0,
    }

    out = OUT_DIR / "results_summary.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\nModel agreement: {result['topic_agreement']['mean']}")
    print(f"\nLDA topics:")
    for t in result["lda_topics"]:
        print(f"  T{t['id']:>2} ({t['size']:>6}): {', '.join(t['terms'])}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
