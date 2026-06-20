#!/usr/bin/env python3
"""
Pre-transcribe all audio files using HuggingFace pipeline with batching.
This dramatically speeds up train_dualstream.py, ablation_study.py, etc.
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline

def get_transcript_cache_file(asr_model):
    safe_name = asr_model.replace("/", "_")
    return Path(__file__).parent.parent / "DFAT_Hybrid_Fusion" / f"transcript_cache_{safe_name}.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="vinai/PhoWhisper-large")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cache_file = get_transcript_cache_file(args.asr_model)
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache = json.load(f)

    print("Loading dataset...")
    dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
    df = dataset.to_pandas()
    all_paths = df["path"].values

    uncached_paths = [p for p in all_paths if str(p) not in cache]
    
    if not uncached_paths:
        print("All files are already cached!")
        return

    print(f"Found {len(uncached_paths)} uncached audio files.")
    print(f"Initializing ASR pipeline for {args.asr_model} with batch_size={args.batch_size}...")
    import torch
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=0, # CUDA
        batch_size=args.batch_size,
        torch_dtype=torch.float16,
    )

    print("Starting batched transcription...")
    import librosa
    import io
    
    def load_audio(path_dict, sr=16000):
        try:
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

    # We yield audio arrays
    def data():
        for p in uncached_paths:
            audio = load_audio(p)
            if audio is not None:
                yield audio
            else:
                yield np.zeros(16000) # dummy

    results = []
    import time
    import numpy as np
    start = time.time()
    
    try:
        for i, out in enumerate(asr_pipeline(data())):
            path = uncached_paths[i]
            text = out["text"]
            cache[str(path)] = text
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"Processed {i+1}/{len(uncached_paths)} - {elapsed:.1f}s")
                with open(cache_file, "w") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
    except KeyboardInterrupt:
        print("Interrupted. Saving cache...")
    finally:
        with open(cache_file, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print("Done pre-transcription.")

if __name__ == "__main__":
    main()
