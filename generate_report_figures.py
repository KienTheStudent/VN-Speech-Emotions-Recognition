import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from pathlib import Path

# Setup
os.makedirs("report_images", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

print("Loading dataset and manifest...")
dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
df = dataset.to_pandas()

with open("split_manifest.json", "r") as f:
    manifest = json.load(f)

# Reconstruct splits
train_idx = manifest["train_indices"]
val_idx = manifest["val_indices"]
test_idx = manifest["test_indices"]

df['split'] = 'none'
df.loc[train_idx, 'split'] = 'train'
df.loc[val_idx, 'split'] = 'val'
df.loc[test_idx, 'split'] = 'test'

print("\n--- UTTERANCE LENGTHS ---")
print(df['duration'].describe(percentiles=[.5, .9, .95]))

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='emotion', y='duration', hue='emotion', palette='pastel', order=['angry', 'happy', 'neutral', 'sad'])
plt.title("Utterance Duration by Emotion")
plt.ylabel("Duration (seconds)")
plt.xlabel("Emotion")
plt.tight_layout()
plt.savefig("report_images/duration_violin.png", dpi=300)
plt.close()

print("\nDone! Duration plot saved.")
