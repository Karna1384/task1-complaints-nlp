# Task 1 — NLP Topic Analysis of Consumer Complaint Texts

**Project: Data Analysis (CSEBDSEDA02)**

Extracts the topics consumers most frequently complain about, from a corpus of
free-text complaints filed with the U.S. Consumer Financial Protection Bureau.

---

## What this does

Three scripts, run in order:

| Script | Produces | Typical runtime |
|---|---|---|
| `pipeline.py` | Cleans and deduplicates the corpus → `data/corpus_clean.parquet` | 2–3 min |
| `vectorise.py` | TF-IDF matrix + sentence embeddings → `outputs/` | ~25 min |
| `topics.py` | Coherence sweep, LDA, NMF, label validation → `outputs/results_summary.json` | ~10 min |

The embeddings step is by far the slowest. Every stage writes its output to
disk, so a later stage can be re-run without repeating the ones before it.

---

## Quick start

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Data — see below
mkdir -p data
# place Consumer_Complaints.csv in data/

# 4. Run
python pipeline.py
python vectorise.py
python topics.py
```

**Smoke test first.** `python vectorise.py --sample 5000` runs the whole chain
on 5,000 documents in about a minute, which is the right way to check a change
before committing to the full corpus.

---

## Data

The corpus is **not included in this repository** — it is 255 MB, and it is
third-party data.

Download `Consumer_Complaints.csv` from Kaggle:
<https://www.kaggle.com/datasets/ashwinik/consumer-complaints-financial-products>

Place it at `data/Consumer_Complaints.csv`. If you keep it elsewhere, set the
`DATA_DIR` environment variable instead of editing the code:

```bash
DATA_DIR=/path/to/folder python pipeline.py
```

### Why the corpus is much smaller than the file

The source file holds 670,598 rows, but only 114,704 of them carry a published
complaint narrative. Publication requires the complainant's consent, so
roughly 83 per cent of the file is empty of text. After removing 2,014 exact
duplicates and dropping documents under ten words, **111,108 complaints** remain
for analysis. That filtering is itself a finding, and it is reported in the
write-up rather than presented as routine cleaning.

---

## Configuration

Each script has a `CONFIG` block at the top. The settings that most affect
results:

| Setting | Default | Why it matters |
|---|---|---|
| `MIN_WORDS` | 10 | Corpus median is 137 words, so this removes a negligible tail |
| `TFIDF_MAX_DF` | 0.5 | Terms appearing in >50 % of documents are boilerplate, not topics |
| `TOPIC_CANDIDATES` | 5–20 | Range swept for the topic count |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384-dim, fast enough on CPU for 111 k documents |

Everything is seeded (`SEED = 42`), so repeated runs produce identical results.

---

## Design decisions

**Topic count is selected, not assumed.** LDA is re-fitted across a range of
candidate values and the one with the best coherence wins. The full sweep is
saved, so the choice is auditable.

**Two models, chosen to disagree usefully.** LDA and NMF share bag-of-words
assumptions but decompose differently. Agreement between them is measured as a
Jaccard overlap of topic vocabularies.

**Coherence is implemented directly.** Normalised point-wise mutual information,
about fifteen lines. An unnormalised co-occurrence count scores a pair of very
common words highly merely for being common, which saturates the metric and
leaves the sweep unable to discriminate between topic counts.

**External validation against human labels.** The corpus carries `Product`
(11 classes) and `Issue` (90 classes) labels that no model has seen. A topic
concentrating in one human-applied label is evidence the model recovered real
structure — and this turned out to be the strongest result in the analysis.

**Collocation merging guards against double merges.** Frequent bigrams become
single tokens (`credit_report`), but a candidate is rejected when the whole
bigram already exists as one token. Without that guard, a second merge pass
produces tokens such as `wells_fargo home_mortgage` that never appeared in the
source text.

---

## Results

Run on 111,108 documents at the time of writing:

- **Selected topic count:** 20 (coherence 0.1673)
- **LDA / NMF agreement:** 0.172 across 13 matched pairs
- **Vectorisation agreement:** 0.053 neighbourhood Jaccard
- **Strongest validation:** topic T12 concentrates 72.7 % in *"Loan servicing,
  payments, escrow account"*; T10 65.7 % in *"Loan modification, collection,
  foreclosure"*

**Two limitations worth knowing.** The coherence curve rises to the largest
candidate tested and does not turn over, so twenty is a lower bound rather than
a discovered optimum. And seven of the twenty topics remain diffuse, under 15 %
concentration against their dominant issue — those are general service themes
rather than distinct concerns.

Every figure is written to `outputs/results_summary.json`, which is the source
for the numbers quoted in the written report.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── pipeline.py              # clean + deduplicate
├── vectorise.py             # TF-IDF + embeddings
├── topics.py                # coherence sweep, LDA, NMF, validation
├── data/                    # your download goes here (not committed)
└── outputs/                 # artefacts and results_summary.json
```

---

## Licence and attribution

Code written for coursework (CSEBDSEDA02). The complaint data is published by
the U.S. Consumer Financial Protection Bureau and carries its own terms of use;
check them before redistributing the corpus.
