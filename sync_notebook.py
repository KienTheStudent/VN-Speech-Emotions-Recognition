import json
from pathlib import Path

notebook_path = Path(__file__).parent / "SER.ipynb"

# Define cell contents
cell_intro = """# Vietnamese Speech Emotion Recognition (SER) - Live Inference & Evaluation

This notebook demonstrates the end-to-end inference and evaluation pipeline for the proposed Speech Emotion Recognition models on the ViSEC dataset. It operates strictly on the held-out Test set as defined by the speaker-independent `split_manifest.json`.

**Prerequisites:**
You must have already run `generate_splits.py` to create the manifest, and trained the models using their respective scripts to generate the checkpoints.
"""

cell_setup = """# ==================== 1. SETUP & LOAD MANIFEST ====================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Load original dataset
print("Loading ViSEC dataset...")
dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
df = dataset.to_pandas()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])
emotion_labels = le.classes_.tolist()
num_labels = len(emotion_labels)

# Read speaker-independent split_manifest.json
manifest_path = "split_manifest.json"
if not os.path.exists(manifest_path):
    raise FileNotFoundError("Missing split_manifest.json. Run generate_splits.py first.")

with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

test_idx = manifest['test_indices']
X_test = df['path'].iloc[test_idx].values
y_test = df['label'].iloc[test_idx].values

print(f"✓ Manifest loaded: {manifest['total_samples']} samples total")
print(f"✓ Test set strictly isolated: {len(X_test)} samples")
"""

cell_ecapa = """# ==================== 2. EVALUATE ECAPA-TDNN ====================
import sys
sys.path.append(os.path.abspath("./ECAPA"))
from predict_emotion import EmotionClassifier
from train_emotion_model import prepare_features, AudioFeaturesDataset, collate_fn, evaluate
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecapa_model = EmotionClassifier(num_labels).to(device)

checkpoint_path = "./ECAPA/emotion_model/best_ecapa_model.pth"
if not os.path.exists(checkpoint_path):
    print(f"❌ ECAPA checkpoint not found at {checkpoint_path}. Please retrain.")
else:
    print(f"✨ Loading ECAPA-TDNN model from {checkpoint_path}...")
    ecapa_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    ecapa_model.eval()
    
    print("Extracting Mel-spectrogram features for Test set (this takes a moment)...")
    X_test_feat, y_test_clean = prepare_features(X_test, y_test, "Test")
    test_dataset = AudioFeaturesDataset(X_test_feat, y_test_clean)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print("Running Inference...")
    test_preds, test_labels_out = evaluate(ecapa_model, test_loader, device)
    
    ecapa_f1_weighted = f1_score(test_labels_out, test_preds, average='weighted')
    ecapa_acc = accuracy_score(test_labels_out, test_preds)
    
    print(f"\\n✓ ECAPA-TDNN Test Accuracy: {ecapa_acc:.4f}")
    print(f"✓ ECAPA-TDNN Test F1 (weighted): {ecapa_f1_weighted:.4f}")
    
    cm = confusion_matrix(test_labels_out, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title("ECAPA-TDNN Confusion Matrix (Test Set)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
"""

cell_dfat = """# ==================== 3. EVALUATE DFAT HYBRID FUSION ====================
sys.path.append(os.path.abspath("./DFAT Hybrid Fusion"))
import pickle
from transformers import AutoFeatureExtractor, AutoModel, WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer

model_dir = Path("./DFAT Hybrid Fusion/dualstream_model")
if not (model_dir / "metadata.json").exists():
    print("❌ DFAT models not found. Please run train_dualstream.py first.")
else:
    print("✨ Loading Ensemble Models and Weights...")
    with open(model_dir / "lr_model.pkl", "rb") as f: lr_model = pickle.load(f)
    with open(model_dir / "rf_model.pkl", "rb") as f: rf_model = pickle.load(f)
    with open(model_dir / "xgb_model.pkl", "rb") as f: xgb_model = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open(model_dir / "metadata.json", "r") as f: metadata = json.load(f)
    
    w_xgb, w_rf, w_lr = metadata["ensemble_weights"]["xgb"], metadata["ensemble_weights"]["rf"], metadata["ensemble_weights"]["lr"]
    print(f"Ensemble Weights -> XGB: {w_xgb:.3f}, RF: {w_rf:.3f}, LR: {w_lr:.3f}")
    
    print("\\nLoading Extractors (WavLM, Whisper, PhoBERT)...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()
    
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()
    
    from train_dualstream import extract_features_for_split, load_transcript_cache, save_transcript_cache
    
    print("\\nExtracting Dual-Stream Features for Test Set (this will take several minutes)...")
    # This invokes audio loading, Whisper ASR, Underthesea word segmentation, and PhoBERT embedding.
    # It automatically uses transcript_cache.json if available.
    cache = load_transcript_cache()
    test_fused, test_labels_ext = extract_features_for_split(
        X_test, y_test, wavlm_model, wavlm_processor, whisper_model, whisper_processor, 
        phobert_model, phobert_tokenizer, device, "Test", cache
    )
    save_transcript_cache(cache)
    
    # Scale features
    test_fused_scaled = scaler.transform(test_fused)
    
    print("\\nRunning Late Fusion Ensemble Inference...")
    lr_proba = lr_model.predict_proba(test_fused_scaled)
    rf_proba = rf_model.predict_proba(test_fused_scaled)
    xgb_proba = xgb_model.predict_proba(test_fused_scaled)
    
    ensemble_proba = w_xgb * xgb_proba + w_rf * rf_proba + w_lr * lr_proba
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    dfat_f1_weighted = f1_score(test_labels_ext, ensemble_pred, average="weighted")
    dfat_acc = accuracy_score(test_labels_ext, ensemble_pred)
    
    print(f"\\n✓ DFAT Hybrid Fusion Test Accuracy: {dfat_acc:.4f}")
    print(f"✓ DFAT Hybrid Fusion Test F1 (weighted): {dfat_f1_weighted:.4f}")
    
    cm = confusion_matrix(test_labels_ext, ensemble_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title("DFAT Hybrid Fusion Confusion Matrix (Test Set)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
"""

def split_lines(text):
    return [line + "\n" for line in text.split("\n")]

# Build clean notebook structure
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": split_lines(cell_intro)[:-1]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_setup)[:-1]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_ecapa)[:-1]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_dfat)[:-1]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook synchronized successfully to {notebook_path}")
