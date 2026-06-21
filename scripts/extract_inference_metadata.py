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
- failure_flags (e.g. empty_transcript)

Uses the configured ASR backend (PhoWhisper-large).
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

from DFAT_Hybrid_Fusion.predict_dualstream import (
    extract_acoustic_features, transcribe_audio, extract_textual_features
)


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
    weights = metadata["representative_run"]["ensemble_weights"]

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

    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open(model_dir / 'rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
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
        
        try:
            audio = load_audio(p)
            if audio is None:
                failure_flags.append("audio_load_failure")
                continue

            # Acoustic
            ac_feat = extract_acoustic_features(p, wavlm_model, wavlm_processor, device)
            
            # Linguistic
            transcript = transcribe_audio(p, whisper_model, whisper_processor, device)
            if not transcript.strip():
                failure_flags.append("empty_transcript")

            txt_feat = extract_textual_features(transcript, phobert_model, phobert_tokenizer, device)

            # Fuse & Predict
            fused = np.concatenate([ac_feat, txt_feat]).reshape(1, -1)
            fused_scaled = scaler.transform(fused)

            lr_proba = lr_model.predict_proba(fused_scaled)[0]
            rf_proba = rf_model.predict_proba(fused_scaled)[0]
            xgb_proba = xgb_model.predict_proba(fused_scaled)[0]

            ens_proba = weights['xgb'] * xgb_proba + weights['rf'] * rf_proba + weights['lr'] * lr_proba
            pred_idx = np.argmax(ens_proba)
            pred_emotion = emotion_labels[pred_idx]
            max_conf = float(np.max(ens_proba))
            
            latency = (time.time() - start_time) * 1000

            results.append({
                "sample_id": str(p).split('/')[-1] if isinstance(p, str) else str(p),
                "true_label": true_emotion,
                "predicted_label": pred_emotion,
                "confidence": max_conf,
                "transcript": transcript,
                "transcript_length": len(transcript.split()),
                "latency_ms": latency,
                "failure_flags": failure_flags
            })
            
        except Exception as e:
            results.append({
                "sample_id": str(p).split('/')[-1] if isinstance(p, str) else str(p),
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
    print(f"Saved inference metadata to {output_path}")

if __name__ == "__main__":
    main()
