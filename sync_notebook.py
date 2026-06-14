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
sys.path.append(os.path.abspath("./DFAT_Hybrid_Fusion"))
import pickle
from transformers import AutoFeatureExtractor, AutoModel, WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer

model_dir = Path("./DFAT_Hybrid_Fusion/dualstream_model")
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

cell_barchart = """# ==================== 4. MODEL COMPARISON CHART ====================
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

benchmark_path = Path("benchmark_results_gpu.json")
if benchmark_path.exists():
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    primary = ["MFCC+RandomForest", "ECAPA-TDNN", "DFAT Late Fusion"]
    methods = []
    wf1s = []
    colors = []
    
    for r in data.get("ranked_results", []):
        methods.append(r["method"])
        wf1s.append(r.get("f1_weighted_mean", 0))
        colors.append("#1f77b4" if r["method"] in primary else "#b0c4de")
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=wf1s, y=methods, palette=colors)
    plt.title("Model Comparison - Weighted F1 Score", pad=20)
    plt.xlabel("Weighted F1 Score")
    plt.xlim(0, 1.0)
    plt.axvline(x=0.7176, color='red', linestyle='--', alpha=0.5, label='DFAT Best')
    plt.legend()
    plt.tight_layout()
    
    # Save to report_images
    Path("report_images").mkdir(exist_ok=True)
    plt.savefig("report_images/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("benchmark_results_gpu.json not found. Run benchmark_methods_gpu.py first.")
"""

cell_ablation = """# ==================== 5. ABLATION STUDY RESULTS ====================
import pandas as pd

ablation_path = Path("DFAT_Hybrid_Fusion/ablation_results.json")
if ablation_path.exists():
    with open(ablation_path, 'r') as f:
        data = json.load(f)
        
    records = []
    for r in data.get("ablation_results", []):
        ens = r.get("ensemble", {})
        records.append({
            "Configuration": r["config"],
            "Weighted F1": ens.get("f1_weighted", 0),
            "Macro F1": ens.get("f1_macro", 0),
            "Accuracy": ens.get("accuracy", 0)
        })
        
    df_ablation = pd.DataFrame(records)
    display(df_ablation.style.background_gradient(cmap='Blues', subset=['Weighted F1']))
else:
    print("Ablation results not found. Run ablation_study.py first.")
"""

cell_demo = """# ==================== 6. LIVE INFERENCE DEMO ====================
demo_audio = "sample_visec.wav"

if not Path(demo_audio).exists():
    print(f"Demo file {demo_audio} not found. Please provide an audio file.")
else:
    print(f"Running Inference on {demo_audio}\\n")
    
    # ECAPA Inference
    from predict_emotion import extract_features as ecapa_extract
    feat = ecapa_extract(demo_audio)
    feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ecapa_out = ecapa_model(feat_tensor)
        ecapa_probs = torch.softmax(ecapa_out, dim=1).squeeze().cpu().numpy()
        
    # DFAT Inference
    ac_feat = wavlm_model(**{k: v.to(device) for k, v in wavlm_processor(
        load_audio(demo_audio), sampling_rate=16000, return_tensors="pt", padding=True).items()
    }).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    inputs = whisper_processor(load_audio(demo_audio), sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        ids = whisper_model.generate(inputs, max_new_tokens=100)
        text = whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]
    
    from underthesea import word_tokenize
    seg_text = word_tokenize(text, format="text")
    inputs = phobert_tokenizer(seg_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        lg_feat = phobert_model(**inputs).pooler_output.squeeze().cpu().numpy()
        
    fused = np.concatenate([[ac_feat], [lg_feat]], axis=1)
    fused_scaled = scaler.transform(fused)
    
    dfat_proba = (w_xgb * xgb_model.predict_proba(fused_scaled) + 
                  w_rf * rf_model.predict_proba(fused_scaled) + 
                  w_lr * lr_model.predict_proba(fused_scaled))[0]
                  
    # Print Side-by-Side
    print(f"{'Emotion':<10} | {'ECAPA Prob':<15} | {'DFAT Prob':<15}")
    print("-" * 45)
    for i, label in enumerate(emotion_labels):
        print(f"{label:<10} | {ecapa_probs[i]:<15.4f} | {dfat_proba[i]:<15.4f}")
    
    print(f"\\nTranscribed Text: {text}")
    print(f"ECAPA Prediction: {emotion_labels[np.argmax(ecapa_probs)]}")
    print(f"DFAT Prediction:  {emotion_labels[np.argmax(dfat_proba)]}")
"""


def split_lines(text):
    return [line + "\n" for line in text.split("\n")]


# Build clean notebook structure
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": split_lines(cell_intro)[:-1]},
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_setup)[:-1],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_ecapa)[:-1],
    },
    {"cell_type": "markdown", "metadata": {}, "source": split_lines("## Results Summary\n\nBelow are the final evaluation outputs for our baseline and proposed models.")[:-1]},
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_dfat)[:-1],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_barchart)[:-1],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_ablation)[:-1],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_lines(cell_demo)[:-1],
    },
]

nb = {
    "cells": cells,
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 2,
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook synchronized successfully to {notebook_path}")
