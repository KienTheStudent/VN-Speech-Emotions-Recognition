#!/usr/bin/env python3
"""
extract_inference_metadata.py

Runs inference on the test split using the trained DFAT Hybrid Fusion model,
extracts confidence scores, ASR transcripts, and failure flags, and saves
them to per_sample_predictions.json for the Jupyter Notebook visualization.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

warnings.filterwarnings('ignore')

from underthesea import word_tokenize as vi_word_tokenize
import librosa

ROOT_DIR = Path(__file__).parent.parent

def load_audio(audio_path, sr=16000):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=str(ROOT_DIR / "DFAT_Hybrid_Fusion" / "dualstream_model"))
    parser.add_argument('--asr_model', type=str, default="vinai/PhoWhisper-large")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory {model_dir} not found.")
        return

    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
        
    emotion_labels = metadata['emotion_labels']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    whisper_processor = WhisperProcessor.from_pretrained(args.asr_model)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model).to(device).eval()
    
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()

    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    print("Loading dataset...")
    dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
    df = dataset.to_pandas()
    
    manifest_path = ROOT_DIR / "split_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    test_idx = manifest["test_indices"]
    test_paths = df["path"].iloc[test_idx].values
    test_labels_raw = df["emotion"].iloc[test_idx].values
    
    cache_path = ROOT_DIR / "DFAT_Hybrid_Fusion" / f"transcript_cache_{args.asr_model.replace('/', '_')}.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    predictions = []
    
    print(f"Processing {len(test_paths)} test samples...")
    for i, (audio_path, true_emotion) in enumerate(zip(test_paths, test_labels_raw)):
        if i % 10 == 0:
            print(f"  [{i}/{len(test_paths)}]")
            
        path_str = str(audio_path)
        
        # Acoustic
        audio = load_audio(path_str, sr=16000)
        if audio is None:
            continue
            
        inputs_wavlm = wavlm_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs_wavlm = {k: v.to(device) for k, v in inputs_wavlm.items()}
        with torch.no_grad():
            out_wavlm = wavlm_model(**inputs_wavlm)
            acoustic = out_wavlm.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        # Textual
        if path_str in cache:
            text = cache[path_str]
        else:
            inputs_whisper = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs_whisper = inputs_whisper.input_features.to(device)
            with torch.no_grad():
                pred_ids = whisper_model.generate(inputs_whisper, max_new_tokens=100)
                text = whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            cache[path_str] = text

        words = vi_word_tokenize(text, format="text")
        if not words.strip():
            textual = np.zeros(768, dtype=np.float32)
        else:
            inputs_phobert = phobert_tokenizer(words, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out_phobert = phobert_model(**inputs_phobert)
                textual = out_phobert.pooler_output.squeeze().cpu().numpy()
                if textual.ndim == 0:
                    textual = np.expand_dims(textual, 0)
        
        # Predict
        fused = np.concatenate([acoustic, textual]).reshape(1, -1)
        if isinstance(scaler, dict) and "ac" in scaler:
            fused[:, :768] = scaler["ac"].transform(fused[:, :768])
            fused[:, 768:] = scaler["tx"].transform(fused[:, 768:])
        else:
            fused = scaler.transform(fused)
        
        proba = xgb_model.predict_proba(fused)[0]
        pred_idx = np.argmax(proba)
        pred_emotion = emotion_labels[pred_idx]
        confidence = float(proba[pred_idx])
        
        # Flags
        flags = []
        if not words.strip():
            flags.append("empty_transcript")
        if true_emotion != pred_emotion:
            if true_emotion in ['happy', 'sad'] and pred_emotion == 'neutral':
                flags.append("acoustic_ambiguity")
            elif true_emotion == 'angry' and pred_emotion != 'angry':
                flags.append("asr_noise_or_acoustic")
                
        predictions.append({
            "sample_id": Path(path_str).name,
            "true_label": true_emotion,
            "predicted_label": pred_emotion,
            "confidence": confidence,
            "transcript": text,
            "failure_flags": flags
        })
        
    out_path = ROOT_DIR / "per_sample_predictions.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        
    if not cache_path.exists():
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
    print(f"Saved {len(predictions)} predictions to {out_path}")

if __name__ == "__main__":
    main()
