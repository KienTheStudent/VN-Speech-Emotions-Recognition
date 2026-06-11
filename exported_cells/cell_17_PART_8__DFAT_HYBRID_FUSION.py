# ==================== PART 8: DFAT HYBRID FUSION ====================
print("🤖 DFAT HYBRID FUSION: WavLM + Whisper")

import sys
sys.path.append(os.path.abspath("./DFAT Hybrid Fusion"))
from train_dualstream import extract_features_for_split
from transformers import AutoFeatureExtractor, AutoModel, WhisperProcessor, WhisperForConditionalGeneration
import pickle

model_dir = "./DFAT Hybrid Fusion/dualstream_model"

if MODE == "demo":
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print(f"✨ DEMO MODE: Found model in {model_dir}, loading model...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        print(f"✓ Ensemble F1 Score (Weighted): {metadata['test_f1_weighted']:.4f}")
        
        cm = np.array(metadata['confusion_matrix'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title("DFAT Hybrid Fusion Confusion Matrix")
        plt.show()
    else:
        print("❌ DFAT model not found. Please switch to MODE = 'retrain'.")
else:
    print("🚀 RETRAIN MODE: Training model from scratch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading WavLM & Whisper...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()
    
    print("Extracting Test features...")
    test_fused, test_labels_ext = extract_features_for_split(
        X_test, y_test, wavlm_model, wavlm_processor, whisper_model, whisper_processor, device, "Test"
    )
    # Shortened training process for notebook, focusing on existing code in the python script
    print("Please refer to train_dualstream.py to see the full leak-free Optuna tuning process.")
