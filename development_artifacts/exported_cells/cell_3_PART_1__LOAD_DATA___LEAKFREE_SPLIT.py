# ==================== PART 1: LOAD DATA & LEAK-FREE SPLIT ====================
print("📥 LOADING VISEC DATASET AND READING LEAK-FREE MANIFEST...")

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

# Add root directory to sys.path for importing
sys.path.append(os.path.abspath("."))

# Load original dataset
print("Loading ViSEC dataset...")
dataset = load_dataset("hustep-lab/ViSEC")
df = dataset['train'].to_pandas()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])
emotion_labels = le.classes_.tolist()
num_labels = len(emotion_labels)

# Read fixed split_manifest.json as the single source of truth
manifest_path = "split_manifest.json"
if not os.path.exists(manifest_path):
    print("split_manifest.json not found, please run generate_splits.py first.")
    raise FileNotFoundError("Missing split_manifest.json")

with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

train_idx = manifest['train_indices']
val_idx = manifest['val_indices']
test_idx = manifest['test_indices']

# Assign train/val/test data variables
X_train = df['path'].iloc[train_idx].values
y_train = df['label'].iloc[train_idx].values
X_val = df['path'].iloc[val_idx].values
y_val = df['label'].iloc[val_idx].values
X_test = df['path'].iloc[test_idx].values
y_test = df['label'].iloc[test_idx].values

# Classical ML merges Train + Val
X_trainval = df['path'].iloc[train_idx + val_idx].values
y_trainval = df['label'].iloc[train_idx + val_idx].values

print(f"✓ Manifest loaded: {manifest['total_samples']} samples")
print(f"  - Train: {len(X_train)} samples")
print(f"  - Val:   {len(X_val)} samples")
print(f"  - Test:  {len(X_test)} samples")
