#!/usr/bin/env python3
import argparse
import json
import sys
import warnings
import numpy as np
import librosa
import torch
import pickle
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration

warnings.filterwarnings('ignore')

from underthesea import word_tokenize as vi_word_tokenize

def load_audio(audio_path, sr=16000):
    """Load audio file with error handling."""
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        if audio is None or len(audio) == 0:
            raise ValueError("Loaded audio is empty")
        return audio
    except Exception as e:
        raise ValueError(f"Failed to load audio '{audio_path}': {e}")

def extract_acoustic_features(audio_path, model, processor, device):
    audio = load_audio(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return features

def transcribe_audio(audio_path, model, processor, device):
    """Transcribe audio to text using Whisper."""
    audio = load_audio(audio_path, sr=16000)
    if audio is None:
        return ""
        
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_new_tokens=100)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def segment_text(text):
    """Word-segment Vietnamese text for PhoBERT."""
    if not text.strip():
        return ""
    return vi_word_tokenize(text, format="text")

def extract_textual_features(text, model, tokenizer, device):
    """Extract PhoBERT textual embeddings from word-segmented text."""
    segmented = segment_text(text)
    if not segmented.strip():
        return np.zeros(768, dtype=np.float32)
    inputs = tokenizer(segmented, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.pooler_output.squeeze().cpu().numpy()
        if features.ndim == 0:
            features = np.expand_dims(features, 0)
    return features

def predict_emotion(audio_path, model_dir):
    model_dir = Path(model_dir)
    
    # Load metadata
    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    emotion_labels = metadata['emotion_labels']
    weights = metadata['ensemble_weights']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()
    
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()
    
    # Extract features
    acoustic_features = extract_acoustic_features(audio_path, wavlm_model, wavlm_processor, device)
    
    text = transcribe_audio(audio_path, whisper_model, whisper_processor, device)
    textual_features = extract_textual_features(text, phobert_model, phobert_tokenizer, device)
    
    fused_features = np.concatenate([acoustic_features, textual_features]).reshape(1, -1)
    
    # Load scaler and transform
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    fused_features = scaler.transform(fused_features)
    
    # Load classifiers
    with open(model_dir / 'lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open(model_dir / 'rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open(model_dir / 'xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Predict
    lr_proba = lr_model.predict_proba(fused_features)[0]
    rf_proba = rf_model.predict_proba(fused_features)[0]
    xgb_proba = xgb_model.predict_proba(fused_features)[0]
    
    # Ensemble
    ensemble_proba = weights['xgb'] * xgb_proba + weights['rf'] * rf_proba + weights['lr'] * lr_proba
    
    return {label: float(prob) for label, prob in zip(emotion_labels, ensemble_proba)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, help="Path to single audio file")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_set', action='store_true', help="Evaluate on the full test set")
    args = parser.parse_args()
    
    if args.test_set:
        from datasets import load_dataset
        from sklearn.metrics import classification_report, confusion_matrix
        import os
        
        manifest_path = Path(args.model_dir).parent.parent / "split_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        test_idx = manifest["test_indices"]
        dataset = load_dataset("hustep-lab/ViSEC", trust_remote_code=True)
        df = dataset["train"].to_pandas()[["path", "emotion"]].copy()
        test_paths = df["path"].iloc[test_idx].values
        test_labels_raw = df["emotion"].iloc[test_idx].values
        
        y_true = []
        y_pred = []
        
        print(f"Evaluating on {len(test_paths)} test samples...")
        for i, (p, true_emotion) in enumerate(zip(test_paths, test_labels_raw)):
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_paths)}")
            try:
                res = predict_emotion(p, args.model_dir)
                pred_emotion = max(res, key=res.get)
                y_true.append(true_emotion)
                y_pred.append(pred_emotion)
            except Exception as e:
                print(f"Error processing {p}: {e}")
                
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
    elif args.audio_file:
        result = predict_emotion(args.audio_file, args.model_dir)
        print(json.dumps(result))
    else:
        print("Please provide either --audio_file or --test_set")