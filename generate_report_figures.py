#!/usr/bin/env python3
"""
Generate Report Figures — Single Source of Truth for All Thesis Plots.

Reads from:
  - split_manifest.json
  - benchmark_results_gpu.json
  - DFAT_Hybrid_Fusion/ablation_results.json
  - ViSEC dataset (via HuggingFace)

Outputs all figures to report_images/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# ── Setup ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_images"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})

EMOTION_ORDER = ["angry", "happy", "neutral", "sad"]
EMOTION_LABELS = ["Angry", "Happy", "Neutral", "Sad"]

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading dataset and manifests...")
from datasets import load_dataset

dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
df = dataset.to_pandas()

with open(ROOT / "split_manifest.json", "r") as f:
    manifest = json.load(f)

with open(ROOT / "benchmark_results_gpu.json", "r") as f:
    benchmark = json.load(f)

with open(ROOT / "DFAT_Hybrid_Fusion" / "ablation_results.json", "r") as f:
    ablation = json.load(f)

# Reconstruct splits
df["split"] = "none"
df.loc[manifest["train_indices"], "split"] = "Train"
df.loc[manifest["val_indices"], "split"] = "Val"
df.loc[manifest["test_indices"], "split"] = "Test"

# ── Plot 1: Duration Violin ───────────────────────────────────────────────────
print("  [1/11] Duration violin...")
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df, x="emotion", y="duration", hue="emotion",
               palette="pastel", order=EMOTION_ORDER, ax=ax)
ax.set_xlabel("Emotion")
ax.set_ylabel("Duration (seconds)")
ax.set_title("Utterance Duration by Emotion Class")
fig.tight_layout()
fig.savefig(OUT / "duration_violin.png")
plt.close(fig)

# ── Plot 2: Speaker Histogram ─────────────────────────────────────────────────
print("  [2/11] Speaker histogram...")
fig, ax = plt.subplots(figsize=(10, 5))
for sp, color in zip(["Train", "Val", "Test"], ["#4C72B0", "#55A868", "#C44E52"]):
    sub = df[df["split"] == sp]
    counts = sub.groupby("speaker_id").size()
    ax.hist(counts, bins=20, alpha=0.6, label=sp, color=color, edgecolor="white")
ax.set_xlabel("Utterances per Speaker")
ax.set_ylabel("Number of Speakers")
ax.set_title("Speaker Contribution Distribution by Split")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "speaker_histogram.png")
plt.close(fig)

# ── Plot 3: Class Distribution ────────────────────────────────────────────────
print("  [3/11] Class distribution...")
split_order = ["Train", "Val", "Test"]
fig, ax = plt.subplots(figsize=(8, 5))
cross = pd.crosstab(df["split"], df["emotion"])
cross = cross.reindex(index=split_order, columns=EMOTION_ORDER)
cross.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_xlabel("Split")
ax.set_ylabel("Sample Count")
ax.set_title("Class Distribution Across Splits")
ax.legend(title="Emotion")
ax.set_xticklabels(split_order, rotation=0)
fig.tight_layout()
fig.savefig(OUT / "class_distribution.png")
plt.close(fig)

# ── Plot 4: Samples per Speaker (sorted bar) ─────────────────────────────────
print("  [4/11] Samples per speaker...")
speaker_counts = df.groupby("speaker_id").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(speaker_counts)), speaker_counts.values, color="#4C72B0", width=1.0)
ax.set_xlabel("Speaker (sorted by contribution)")
ax.set_ylabel("Number of Utterances")
ax.set_title("Samples per Speaker (Descending)")
fig.tight_layout()
fig.savefig(OUT / "samples_per_speaker.png")
plt.close(fig)

# ── Plot 5: Main Benchmark Bar Chart with Error Bars ─────────────────────────
print("  [5/11] Main benchmark bar chart...")
ranked = benchmark["ranked_results"]
methods = [r["method"] for r in ranked]
wf1s = [r["f1_weighted_mean"] for r in ranked]
stds = [r["f1_weighted_std"] for r in ranked]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#C44E52", "#4C72B0", "#55A868"]
bars = ax.bar(methods, wf1s, yerr=stds, capsize=5, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, wf1s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Weighted F1")
ax.set_title("Main Benchmark: Weighted F1 on Held-Out Test Set")
ax.set_ylim(0, 0.85)
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png")
plt.close(fig)

# ── Plot 6: Per-class F1 Comparison ──────────────────────────────────────────
print("  [6/11] Per-class F1 comparison...")
ecapa_report = next(r for r in ranked if r["method"].startswith("ECAPA-TDNN"))["representative_run"]["classification_report"]
dfat_report = next(r for r in ranked if r["method"] == "DFAT Late Fusion")["representative_run"]["classification_report"]

ecapa_f1 = [ecapa_report[e]["f1-score"] for e in EMOTION_ORDER]
dfat_f1 = [dfat_report[e]["f1-score"] for e in EMOTION_ORDER]

x = np.arange(len(EMOTION_LABELS))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, ecapa_f1, width, label="ECAPA-TDNN", color="#4C72B0")
ax.bar(x + width / 2, dfat_f1, width, label="DFAT Late Fusion", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(EMOTION_LABELS)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Comparison")
ax.legend()
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig(OUT / "per_class_f1.png")
plt.close(fig)

# ── Plot 7: Confusion Matrices ───────────────────────────────────────────────
print("  [7/11] Confusion matrices...")
for model_name, key in [("ECAPA-TDNN (simplified implementation)", "ecapa"), ("DFAT Late Fusion", "dfat")]:
    result = next(r for r in ranked if r["method"] == model_name)
    cm = np.array(result["representative_run"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_matrix_{key}.png")
    plt.close(fig)

# ── Plot 8: Top Confusion Pairs ──────────────────────────────────────────────
print("  [8/11] Top confusion pairs...")
dfat_cm = np.array(next(r for r in ranked if r["method"] == "DFAT Late Fusion")["representative_run"]["confusion_matrix"])
pairs = []
for i in range(4):
    for j in range(4):
        if i != j:
            pairs.append((f"{EMOTION_LABELS[i]}→{EMOTION_LABELS[j]}", dfat_cm[i][j]))
pairs.sort(key=lambda x: x[1], reverse=True)
top_pairs = pairs[:6]

fig, ax = plt.subplots(figsize=(8, 4))
labels, counts = zip(*top_pairs)
ax.barh(labels[::-1], counts[::-1], color="#C44E52", edgecolor="white")
ax.set_xlabel("Misclassification Count")
ax.set_title("Top Confusion Pairs (DFAT Late Fusion)")
fig.tight_layout()
fig.savefig(OUT / "top_confusion_pairs.png")
plt.close(fig)

# ── Plot 9: Ablation Heatmap ─────────────────────────────────────────────────
print("  [9/11] Ablation heatmap...")
abl_data = ablation["ablation_results"]
configs = [r["config"] for r in abl_data]
metrics = ["wF1", "mF1", "Acc"]
matrix = np.array([[r["ensemble"]["f1_weighted"], r["ensemble"]["f1_macro"], r["ensemble"]["accuracy"]] for r in abl_data])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=metrics,
            yticklabels=[c[:35] for c in configs], ax=ax)
ax.set_title("DFAT Ablation Study: Performance Heatmap")
fig.tight_layout()
fig.savefig(OUT / "ablation_heatmap.png")
plt.close(fig)

# ── Plot 10: Seed Stability (RF multi-seed) ──────────────────────────────────
print("  [10/11] Seed stability plot...")
# RF has 5-seed data; ECAPA and DFAT have 1 seed each (no variance to plot)
rf_result = next(r for r in ranked if r["method"] == "MFCC+RandomForest")
ecapa_result = next(r for r in ranked if r["method"].startswith("ECAPA-TDNN"))
dfat_result = next(r for r in ranked if r["method"] == "DFAT Late Fusion")

model_names = ["MFCC+RF", "ECAPA-TDNN", "DFAT Late"]
means = [rf_result["f1_weighted_mean"], ecapa_result["f1_weighted_mean"], dfat_result["f1_weighted_mean"]]
stds_plot = [rf_result["f1_weighted_std"], ecapa_result["f1_weighted_std"], dfat_result["f1_weighted_std"]]
n_seeds_list = [rf_result["n_seeds"], ecapa_result["n_seeds"], dfat_result["n_seeds"]]

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#55A868", "#4C72B0", "#C44E52"]
for i, (name, mean, std, ns) in enumerate(zip(model_names, means, stds_plot, n_seeds_list)):
    marker = "o" if ns > 1 else "s"
    ax.errorbar(i, mean, yerr=std, fmt=marker, markersize=10, color=colors[i],
                capsize=8, capthick=2, elinewidth=2, label=f"{name} ({ns} seed{'s' if ns > 1 else ''})")
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel("Weighted F1 (mean ± std)")
ax.set_title("Seed Stability: wF1 with Error Bars")
ax.legend(loc="upper left")
ax.set_ylim(0, 0.85)
fig.tight_layout()
fig.savefig(OUT / "seed_stability.png")
plt.close(fig)

# ── Plot 11: Latency vs Quality Trade-off ────────────────────────────────────
print("  [11/11] Latency vs quality scatter...")
# Use parameter count as proxy for compute footprint
model_points = [
    ("MFCC+RF", rf_result["f1_weighted_mean"], 0.5),       # ~0.5M params
    ("ECAPA-TDNN", ecapa_result["f1_weighted_mean"], 20.8), # ~20.8M params
    ("DFAT Late", dfat_result["f1_weighted_mean"], 473),    # ~473M params
]

fig, ax = plt.subplots(figsize=(8, 5))
for name, wf1, params in model_points:
    ax.scatter(params, wf1, s=200, zorder=5)
    ax.annotate(name, (params, wf1), textcoords="offset points", xytext=(10, 5),
                fontsize=11, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("Estimated Parameters (Millions, log scale)")
ax.set_ylabel("Weighted F1")
ax.set_title("Quality vs. Computational Cost Trade-off")
ax.set_ylim(0.3, 0.8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "latency_vs_quality.png")
plt.close(fig)

print(f"\nDone! All {len(list(OUT.glob('*.png')))} figures saved to {OUT}/")
