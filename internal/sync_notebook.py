import os
import json
from pathlib import Path

notebook_path = Path(__file__).resolve().parent / "SER.ipynb"

# Define cell contents
cell_intro = """# Vietnamese Speech Emotion Recognition (SER) - Live Inference & Evaluation

This notebook serves as the top-level demonstration and visualization interface.
It avoids heavy evaluation logic, delegating full test set evaluation to `benchmark_methods_gpu.py`.

**Contents:**
1. Setup & Load Artifacts
2. Result Visualization
3. Demo Inference
"""

cell_setup = """# ==================== 1. SETUP & LOAD ARTIFACTS ====================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import librosa
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Use absolute paths using src.config.paths
import sys
sys.path.append(os.path.abspath('..'))
from src.config.paths import BENCHMARK_RESULTS_PATH, ECAPA_METADATA, DFAT_METADATA, ROOT_DIR

print("Loading Benchmark Results...")
if BENCHMARK_RESULTS_PATH.exists():
    with open(BENCHMARK_RESULTS_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    emotion_labels = benchmark_data['emotion_labels']
    print(f"✓ Loaded benchmark results for {len(benchmark_data['ranked_results'])} models.")
else:
    print("❌ benchmark_results_gpu.json not found. Run benchmark_methods_gpu.py first.")
    emotion_labels = ['angry', 'happy', 'neutral', 'sad']
"""

cell_viz = """# ==================== 2. RESULT VISUALIZATION ====================
if BENCHMARK_RESULTS_PATH.exists():
    # 2.1 Benchmark Table
    print("--- Primary Benchmark Results ---")
    records = []
    for r in benchmark_data['ranked_results']:
        records.append({
            "Method": r['method'],
            "wF1 Mean": r['f1_weighted_mean'],
            "wF1 Std": r['f1_weighted_std'],
            "mF1 Mean": r['f1_macro_mean'],
            "Accuracy": r['accuracy_mean']
        })
    df_results = pd.DataFrame(records)
    from IPython.display import display
    display(df_results)
    
    # 2.2 Confusion Matrices
    print("\\n--- Confusion Matrices ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (ax, model_name) in enumerate(zip(axes, ["ECAPA-TDNN (simplified implementation)", "DFAT Late Fusion"])):
        # Find model in benchmark
        model_res = next((r for r in benchmark_data['ranked_results'] if r['method'] == model_name), None)
        if model_res and 'confusion_matrix' in model_res['representative_run']:
            cm = np.array(model_res['representative_run']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=emotion_labels, yticklabels=emotion_labels, ax=ax)
            ax.set_title(model_name)
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
"""

cell_demo = """# ==================== 3. DEMO INFERENCE ====================
# We will demonstrate live inference using the ECAPA-TDNN model on a sample WAV file.
sample_path = ROOT_DIR / "sample_visec.wav"

if not sample_path.exists():
    print(f"❌ Sample audio {sample_path.name} not found.")
else:
    print(f"Playing sample audio: {sample_path.name}")
    import IPython.display as ipd
    from IPython.display import display
    display(ipd.Audio(str(sample_path)))

    # Load ECAPA Model
    from ECAPA.predict_emotion import EmotionClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(emotion_labels)
    
    ecapa_model = EmotionClassifier(num_labels).to(device)
    checkpoint_path = ROOT_DIR / "ECAPA" / "emotion_model" / "best_ecapa_model.pth"
    
    if checkpoint_path.exists():
        ecapa_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        ecapa_model.eval()
        
        # Load and preprocess audio
        audio, sr = librosa.load(str(sample_path), sr=16000)
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_values.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = ecapa_model(features)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()
            
        print(f"\\n🎤 Predicted Emotion: **{emotion_labels[pred_idx].upper()}** (Confidence: {conf*100:.1f}%)")
    else:
        print("❌ Model checkpoint not found for inference.")
"""

def split_lines(text):
    return [line + "\\n" for line in text.split("\\n")]

def generate_notebook():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": split_lines(cell_intro)[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_setup)[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_viz)[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_demo)[:-1]},
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

if __name__ == "__main__":
    generate_notebook()
