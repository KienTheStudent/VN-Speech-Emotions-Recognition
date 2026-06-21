#!/usr/bin/env python3
"""
Extract Inference Metadata for Test Set

Generates `per_sample_predictions.json` containing:
- sample_id (path relative to dataset)
- true_label
- predicted_label
- transcript
- transcript_length
- latency (ms)
- failure_flags (e.g. empty_transcript, audio_load_failure)

Uses the configured ASR backend (PhoWhisper-large) and matches the
Single XGBoost Hard-Fallback architecture.
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import pickle
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from DFAT_Hybrid_Fusion.train_dualstream import segment_text

def load_audio(path_dict, sr=16000):
    try:
        import io
        if isinstance(path_dict, dict) and "bytes" in path_dict:
            audio, _ = librosa.load(io.BytesIO(path_dict["bytes"]), sr=sr)
        elif isinstance(path_dict, str):
            audio, _ = librosa.load(path_dict, sr=sr)
        elif isinstance(path_dict, dict) and "path" in path_dict:
            audio, _ = librosa.load(path_dict["path"], sr=sr)
        else:
            return None
        return audio
    except Exception:
        return None

def extract_acoustic_features(audio_path, model, processor, device):
    """Extract WavLM acoustic embeddings."""
    audio = load_audio(audio_path, sr=16000)
    if audio is None:
        return None
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

def extract_textual_features(text, model, tokenizer, device):
    """Extract PhoBERT textual embeddings from word-segmented text."""
    segmented = segment_text(text)
    if not segmented.strip():
        return np.zeros(768, dtype=np.float32)
    inputs = tokenizer(
        segmented, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.pooler_output.squeeze().cpu().numpy()
        if features.ndim == 0:
            features = np.expand_dims(features, 0)
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--asr_model", type=str, default="vinai/PhoWhisper-large")
    parser.add_argument("--output", type=str, default="per_sample_predictions.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    emotion_labels = metadata["emotion_labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    print(f"Loading ASR: {args.asr_model}")
    whisper_processor = WhisperProcessor.from_pretrained(args.asr_model)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model).to(device).eval()

    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()

    scaler = None
    if (model_dir / 'scaler.pkl').exists():
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

    with open(model_dir / 'xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    print("Loading manifest and test set...")
    manifest_path = Path(__file__).parent.parent / "split_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    test_idx = manifest["test_indices"]

    dataset = load_dataset("hustep-lab/ViSEC", trust_remote_code=True)
    df = dataset["train"].to_pandas()
    test_paths = df["path"].iloc[test_idx].values
    test_labels_raw = df["emotion"].iloc[test_idx].values

    results = []
    print(f"Running inference on {len(test_paths)} test samples...")
    for i, (p, true_emotion) in enumerate(zip(test_paths, test_labels_raw)):
        if i % 10 == 0:
            print(f"Processed {i}/{len(test_paths)}")

        start_time = time.time()
        failure_flags = []
        
        sample_id = str(p).split('/')[-1] if isinstance(p, str) else str(p)

        try:
            audio = load_audio(p)
            if audio is None:
                failure_flags.append("audio_load_failure")
                # Ensure 100% test-set coverage by writing an error record
                results.append({
                    "sample_id": sample_id,
                    "true_label": true_emotion,
                    "predicted_label": "ERROR",
                    "confidence": 0.0,
                    "transcript": "",
                    "transcript_length": 0,
                    "latency_ms": (time.time() - start_time) * 1000,
                    "failure_flags": failure_flags
                })
                continue

            # Acoustic
            ac_feat = extract_acoustic_features(p, wavlm_model, wavlm_processor, device)
            if ac_feat is None:
                ac_feat = np.zeros(768, dtype=np.float32)
                failure_flags.append("acoustic_extraction_failure")
            
            # Linguistic
            transcript = transcribe_audio(p, whisper_model, whisper_processor, device)
            words = segment_text(transcript).split()
            if len(words) == 0:
                failure_flags.append("empty_transcript")
                txt_feat = np.zeros(768, dtype=np.float32)
            else:
                txt_feat = extract_textual_features(transcript, phobert_model, phobert_tokenizer, device)

            # Fuse
            fused = np.concatenate([ac_feat, txt_feat]).reshape(1, -1)
            
            if scaler is not None:
                ac_scaled = scaler["ac"].transform(fused[:, :768])
                tx_scaled = scaler["tx"].transform(fused[:, 768:])
                fused = np.concatenate([ac_scaled, tx_scaled], axis=1)

            xgb_proba = xgb_model.predict_proba(fused)[0]

            pred_idx = np.argmax(xgb_proba)
            pred_emotion = emotion_labels[pred_idx]
            max_conf = float(np.max(xgb_proba))
            
            latency = (time.time() - start_time) * 1000

            results.append({
                "sample_id": sample_id,
                "true_label": true_emotion,
                "predicted_label": pred_emotion,
                "confidence": max_conf,
                "transcript": transcript,
                "transcript_length": len(words),
                "latency_ms": latency,
                "failure_flags": failure_flags
            })
            
        except Exception as e:
            results.append({
                "sample_id": sample_id,
                "true_label": true_emotion,
                "predicted_label": "ERROR",
                "confidence": 0.0,
                "transcript": "",
                "transcript_length": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "failure_flags": failure_flags + ["inference_exception", str(e)]
            })

    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved inference metadata for {len(results)} samples to {output_path}")

if __name__ == "__main__":
    main()
