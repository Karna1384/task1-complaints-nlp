"""Stage 4 - draws the three figures from results_summary.json.

Run after topics.py:  python figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT_DIR = os.environ.get("OUT_DIR", "outputs")
FIG_DIR = os.environ.get("FIG_DIR", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#2F5D9E", "#E07B39", "#3A9D5D", "#8E5EA2", "#6B6B6B"])

with open(os.path.join(OUT_DIR, "results_summary.json")) as fh:
    r = json.load(fh)

ACCENT, HIGHLIGHT = "#2F5D9E", "#E07B39"


def tidy(ax, axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


# ---- figure 1: coherence sweep -------------------------------------------
sw = r["coherence_sweep"]
ks = [d["k"] for d in sw]
co = [d["coherence"] for d in sw]
sel = r["selected_k"]
i = ks.index(sel)

fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained", dpi=200)
ax.plot(ks, co, marker="o", color=ACCENT, linewidth=2, markersize=6)
ax.plot([ks[i]], [co[i]], marker="o", markersize=11, markerfacecolor="none",
        markeredgecolor=HIGHLIGHT, markeredgewidth=2.2)
ax.annotate("selected k = %d  (%.3f)" % (sel, co[i]),
            xy=(ks[i], co[i]), xytext=(ks[i] - 5.5, co[i] + 0.006),
            fontsize=11, color=HIGHLIGHT, fontweight="bold")
ax.set_xticks(ks)
ax.set_xlabel("Number of topics (k)", fontsize=12)
ax.set_ylabel("Topic coherence (NPMI)", fontsize=12)
ax.set_title("Coherence keeps rising: k=%d is the largest value tested, "
             "not a peak" % sel, fontsize=12.5, pad=10)
tidy(ax)
fig.savefig(os.path.join(FIG_DIR, "fig-coherence.png"), dpi=200)
plt.close(fig)

# ---- figure 2: product concentration -------------------------------------
tp = r["topic_vs_product"]
labels = ["T%d" % d["topic"] for d in tp]
shares = [d["share"] for d in tp]
prods = [d["top_label"] for d in tp]
cols = [HIGHLIGHT if s >= 0.80 else ACCENT for s in shares]

fig, ax = plt.subplots(figsize=(6.5, 5.6), layout="constrained", dpi=200)
y = np.arange(len(labels))
ax.barh(y, shares, color=cols, height=0.72)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Share of the topic's documents carrying its dominant "
              "product label", fontsize=11.5)
ax.set_title("Five topics sit above 80%% in one product category", fontsize=12.5,
             pad=10)
for i2, (s, p) in enumerate(zip(shares, prods)):
    ax.text(s + 0.012, i2, "%s  %.0f%%" % (p, s * 100), va="center",
            fontsize=9, color="#333")
tidy(ax, axis="x")
fig.savefig(os.path.join(FIG_DIR, "fig-product-concentration.png"), dpi=200)
plt.close(fig)

# ---- figure 3: issue concentration ---------------------------------------
ti = sorted(r["topic_vs_issue"], key=lambda d: -d["share"])[:12]
labs = ["T%d" % d["topic"] for d in ti]
shr = [d["share"] for d in ti]
iss = [d["top_label"] for d in ti]
short = [s if len(s) <= 42 else s[:39] + "..." for s in iss]
cols = [HIGHLIGHT if s >= 0.40 else ACCENT for s in shr]

fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained", dpi=200)
y = np.arange(len(labs))
ax.barh(y, shr, color=cols, height=0.7)
ax.set_yticks(y)
ax.set_yticklabels(labs, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlim(0, max(shr) * 1.55)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Share of the topic's documents in its dominant issue "
              "category", fontsize=11.5)
ax.set_title("Against 90 issue categories, the strongest topic reaches "
             "%.0f%%" % (max(shr) * 100), fontsize=12.5, pad=10)
for i2, (s, name) in enumerate(zip(shr, short)):
    ax.text(s + 0.012, i2, "%.0f%%   %s" % (s * 100, name), va="center",
            fontsize=8.6, color="#333")
tidy(ax, axis="x")
fig.savefig(os.path.join(FIG_DIR, "fig-issue-concentration.png"), dpi=200)
plt.close(fig)

print("wrote 3 figures to", FIG_DIR)
