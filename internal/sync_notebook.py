import os
import json
from pathlib import Path

notebook_path = Path(__file__).resolve().parent / "SER.ipynb"

# Define cell contents
cell_intro = """# Vietnamese Speech Emotion Recognition (SER) - Live Inference & Evaluation

This notebook demonstrates the end-to-end inference and evaluation pipeline for the proposed Speech Emotion Recognition models on the ViSEC dataset. It operates strictly on the held-out Test set as defined by the speaker-independent `split_manifest.json`.

**Prerequisites:**
You must have already run `generate_splits.py` to create the manifest, and trained the models using their respective scripts to generate the checkpoints.
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

# Move to root to access all paths correctly as originally designed
if os.getcwd().endswith('internal'):
    os.chdir('..')

# Use absolute paths using src.config.paths
import sys
sys.path.append(os.path.abspath('.'))
from src.config.paths import BENCHMARK_RESULTS_PATH, ROOT_DIR

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

cell_viz = """# ==================== 2. CONFUSION MATRICES ====================
if BENCHMARK_RESULTS_PATH.exists():
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

cell_barchart = """# ==================== 3. MODEL COMPARISON CHART ====================
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

benchmark_path = Path("benchmark_results_gpu.json")
if benchmark_path.exists():
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    primary = ["MFCC+RandomForest", "ECAPA-TDNN (simplified implementation)", "DFAT Late Fusion"]
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
    
    best_wf1 = max(wf1s) if wf1s else 0
    plt.axvline(x=best_wf1, color='red', linestyle='--', alpha=0.5, label='DFAT Best')
    plt.legend()
    plt.tight_layout()
    
    # Save to report_images
    Path("report_images").mkdir(exist_ok=True)
    plt.savefig("report_images/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("benchmark_results_gpu.json not found. Run benchmark_methods_gpu.py first.")
"""

cell_ablation = """# ==================== 4. ABLATION STUDY RESULTS ====================
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
    from IPython.display import display
    display(df_ablation.style.background_gradient(cmap='Blues', subset=['Weighted F1']))
else:
    print("Ablation results not found. Run ablation_study.py first.")
"""

cell_demo = """# ==================== 5. LIVE INFERENCE DEMO ====================
demo_audio = "sample_visec.wav"

if not Path(demo_audio).exists():
    print(f"Demo file {demo_audio} not found. Please provide an audio file.")
else:
    print(f"Running Inference on {demo_audio}\\n")
    import IPython.display as ipd
    from IPython.display import display
    display(ipd.Audio(demo_audio))

    # ECAPA Inference
    from ECAPA.predict_emotion import EmotionClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(emotion_labels)
    
    ecapa_model = EmotionClassifier(num_labels).to(device)
    checkpoint_path = Path("ECAPA/emotion_model/best_ecapa_model.pth")
    
    if checkpoint_path.exists():
        ecapa_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        ecapa_model.eval()
        
        audio, sr = librosa.load(demo_audio, sr=16000)
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_values.to(device)
        
        with torch.no_grad():
            outputs = ecapa_model(features)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()
            
        print(f"\\n🎤 ECAPA-TDNN Predicted Emotion: **{emotion_labels[pred_idx].upper()}** (Confidence: {conf*100:.1f}%)")
    else:
        print("❌ Model checkpoint not found for inference.")
"""

def split_lines(text):
    return [line + "\\n" for line in text.split("\\n")]

def generate_notebook():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": split_lines(cell_intro)[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_setup)[:-1]},
        {"cell_type": "markdown", "metadata": {}, "source": split_lines("## Results Summary\\n\\nBelow are the final evaluation outputs for our baseline and proposed models.")[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_viz)[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_barchart)[:-1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_lines(cell_ablation)[:-1]},
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
