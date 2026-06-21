#!/usr/bin/env python3
"""
DFAT Hybrid Fusion Training — Leak-Free Protocol.

Single-axis architecture: Raw concat (WavLM + PhoBERT) → XGBoost.
Hard invalid-text fallback: if ASR transcript is empty/failed, the
linguistic stream is zeroed out (acoustic-only fallback).

Reads the fixed Train/Val/Test split from split_manifest.json.
- Train set: used for fitting the XGBoost classifier.
- Val set: used for Optuna hyperparameter tuning.
- Test set: evaluated exactly ONCE at the end for final reporting.
"""

import json
import pickle
import warnings
from pathlib import Path

import librosa
import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import argparse

from underthesea import word_tokenize as vi_word_tokenize

MANIFEST_PATH = Path(__file__).parent.parent / "split_manifest.json"


def get_transcript_cache_file(asr_model):
    safe_name = asr_model.replace("/", "_")
    return Path(__file__).parent / f"transcript_cache_{safe_name}.json"


def load_transcript_cache(asr_model):
    cache_file = get_transcript_cache_file(asr_model)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_transcript_cache(cache, asr_model):
    cache_file = get_transcript_cache_file(asr_model)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==================== UTILITIES ====================


def load_manifest():
    """Load the fixed split manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


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
    """Extract WavLM acoustic embeddings (SEFE stream)."""
    audio = load_audio(audio_path, sr=16000)
    if audio is None:
        return None
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return features


def transcribe_audio(audio_path, model, processor, device, cache):
    """Transcribe audio to text using Whisper."""
    path_str = str(audio_path)
    if path_str in cache:
        return cache[path_str]

    audio = load_audio(audio_path, sr=16000)
    if audio is None:
        print(f"Warning: Failed to load {audio_path} for transcription.")
        return ""

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_new_tokens=100)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]

    cache[path_str] = transcription
    return transcription


def segment_text(text):
    """Word-segment Vietnamese text for PhoBERT.

    PhoBERT was pre-trained on word-segmented data (multi-syllable words
    joined by underscores, e.g. 'Đại_học Quốc_gia'). Raw text must be
    segmented before tokenization for proper embeddings.
    """
    if not text.strip():
        return ""
    return vi_word_tokenize(text, format="text")


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


def extract_features_for_split(
    paths,
    labels,
    wavlm_model,
    wavlm_proc,
    whisper_model,
    whisper_proc,
    phobert_model,
    phobert_tokenizer,
    device,
    split_name,
    cache,
):
    """Extract dual-stream features for a given split.

    Hard invalid-text fallback: if the ASR transcript is empty or
    decode failed, the textual embedding is zeroed out (the model
    falls back to acoustic-only for that sample).
    """
    acoustic_list, textual_list, label_list = [], [], []

    for i, (path, label) in enumerate(zip(paths, labels)):
        if i % 200 == 0:
            print(f"  [{split_name}] {i}/{len(paths)}")

        acoustic = extract_acoustic_features(path, wavlm_model, wavlm_proc, device)
        if acoustic is None:
            print(f"Warning: Skipping {path} due to load failure.")
            continue

        text = transcribe_audio(path, whisper_model, whisper_proc, device, cache)
        textual = extract_textual_features(
            text, phobert_model, phobert_tokenizer, device
        )

        # Hard invalid-text fallback:
        # valid transcript → weight = 1, invalid/empty → weight = 0
        words = segment_text(text).split()
        if len(words) == 0:
            textual = np.zeros(768, dtype=np.float32)

        acoustic_list.append(acoustic)
        textual_list.append(textual)
        label_list.append(label)

    if (i + 1) % 200 == 0 or (i + 1) == len(paths):
        save_transcript_cache(cache, getattr(whisper_model.config, 'name_or_path', 'unknown_asr'))

    fused = np.concatenate([np.array(acoustic_list), np.array(textual_list)], axis=1)
    print(f"  [{split_name}] Valid samples: {len(label_list)} / {len(paths)}")
    return fused, np.array(label_list)


# ==================== MAIN ====================


def main():
    parser = argparse.ArgumentParser(description="DFAT Hybrid Fusion Training")
    parser.add_argument("--asr_model", type=str, default="vinai/PhoWhisper-large", help="ASR model to use (default: vinai/PhoWhisper-large)")
    parser.add_argument("--use_scaler", action="store_true", help="Enable per-stream StandardScaler normalization (ablation)")
    args = parser.parse_args()

    print("=" * 60)
    print("DFAT HYBRID FUSION — SINGLE-AXIS PROTOCOL")
    print(f"ASR Model: {args.asr_model}")
    print(f"Per-Stream Scaler: {args.use_scaler}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load manifest & dataset
    # ------------------------------------------------------------------
    print("\nLoading split manifest...")
    manifest = load_manifest()
    split_checksum = manifest.get("checksum", "unknown")
    print(f"  Train: {len(manifest['train_indices'])}")
    print(f"  Val:   {len(manifest['val_indices'])}")
    print(f"  Test:  {len(manifest['test_indices'])}")

    print("\nLoading ViSEC dataset...")
    dataset = load_dataset("hustep-lab/ViSEC", trust_remote_code=True)
    df = dataset["train"].to_pandas()[["path", "emotion"]].copy()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    emotion_labels = le.classes_.tolist()
    print(f"Emotions: {emotion_labels}")

    # ------------------------------------------------------------------
    # 2. Split data using manifest indices
    # ------------------------------------------------------------------
    train_idx = manifest["train_indices"]
    val_idx = manifest["val_indices"]
    test_idx = manifest["test_indices"]

    train_paths = df["path"].iloc[train_idx].values
    train_labels_raw = df["label"].iloc[train_idx].values
    val_paths = df["path"].iloc[val_idx].values
    val_labels_raw = df["label"].iloc[val_idx].values
    test_paths = df["path"].iloc[test_idx].values
    test_labels_raw = df["label"].iloc[test_idx].values

    print(
        f"Split sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}"
    )

    # ------------------------------------------------------------------
    # 3. Load pretrained models
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("Loading WavLM (SEFE)...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = (
        AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    )

    print(f"Loading Whisper (TEFE — ASR-derived) [{args.asr_model}]...")
    whisper_processor = WhisperProcessor.from_pretrained(args.asr_model)
    whisper_model = (
        WhisperForConditionalGeneration.from_pretrained(args.asr_model)
        .to(device)
        .eval()
    )

    print("Loading PhoBERT (TEFE — linguistic)...")
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()

    cache = load_transcript_cache(args.asr_model)

    # ------------------------------------------------------------------
    # 4. Extract dual-stream features for all splits
    # ------------------------------------------------------------------
    print("\nExtracting dual-stream features...")
    train_fused, train_labels = extract_features_for_split(
        train_paths, train_labels_raw,
        wavlm_model, wavlm_processor,
        whisper_model, whisper_processor,
        phobert_model, phobert_tokenizer,
        device, "Train", cache,
    )
    val_fused, val_labels = extract_features_for_split(
        val_paths, val_labels_raw,
        wavlm_model, wavlm_processor,
        whisper_model, whisper_processor,
        phobert_model, phobert_tokenizer,
        device, "Val", cache,
    )
    test_fused, test_labels = extract_features_for_split(
        test_paths, test_labels_raw,
        wavlm_model, wavlm_processor,
        whisper_model, whisper_processor,
        phobert_model, phobert_tokenizer,
        device, "Test", cache,
    )

    print(f"\nFeatures extracted:")
    print(
        f"  Train: {train_fused.shape}, Val: {val_fused.shape}, Test: {test_fused.shape}"
    )

    # Free GPU memory
    del wavlm_model, whisper_model, phobert_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Optional: Per-Stream Normalization (ablation toggle)
    # ------------------------------------------------------------------
    scaler = None
    if args.use_scaler:
        print("\nApplying Per-Stream StandardScaler...")
        scaler_ac = StandardScaler()
        scaler_tx = StandardScaler()
        train_fused[:, :768] = scaler_ac.fit_transform(train_fused[:, :768])
        train_fused[:, 768:] = scaler_tx.fit_transform(train_fused[:, 768:])
        val_fused[:, :768] = scaler_ac.transform(val_fused[:, :768])
        val_fused[:, 768:] = scaler_tx.transform(val_fused[:, 768:])
        test_fused[:, :768] = scaler_ac.transform(test_fused[:, :768])
        test_fused[:, 768:] = scaler_tx.transform(test_fused[:, 768:])
        scaler = {"ac": scaler_ac, "tx": scaler_tx}
    else:
        print("\nUsing Raw Concat (no scaling).")

    # ------------------------------------------------------------------
    # 6. Train XGBoost with Optuna tuning (5 seeds)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING XGBoost (5 Seeds, Optuna Tuning)")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 2026]
    runs = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")

        def xgb_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": seed,
            }
            model = XGBClassifier(**params, eval_metric="mlogloss")
            model.fit(train_fused, train_labels)
            pred = model.predict(val_fused)
            return f1_score(val_labels, pred, average="weighted")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(xgb_objective, n_trials=30, show_progress_bar=False)

        print(f"    Best Val wF1: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        # Train final model with best params on full train set
        xgb_model = XGBClassifier(**study.best_params, eval_metric="mlogloss")
        xgb_model.fit(train_fused, train_labels)

        # Evaluate on test (exactly once)
        test_pred = xgb_model.predict(test_fused)

        test_f1_weighted = f1_score(test_labels, test_pred, average="weighted")
        test_f1_macro = f1_score(test_labels, test_pred, average="macro")
        test_acc = accuracy_score(test_labels, test_pred)

        report_dict = classification_report(test_labels, test_pred, target_names=emotion_labels, output_dict=True)
        cm = confusion_matrix(test_labels, test_pred).tolist()

        run_data = {
            "seed": seed,
            "f1_weighted": float(test_f1_weighted),
            "f1_macro": float(test_f1_macro),
            "accuracy": float(test_acc),
            "xgboost_best_params": study.best_params,
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "model": xgb_model,
        }
        runs.append(run_data)
        print(f"  -> Seed {seed} Test wF1: {test_f1_weighted:.4f}")

    # ------------------------------------------------------------------
    # 7. Aggregate results and save
    # ------------------------------------------------------------------
    wf1s = [r["f1_weighted"] for r in runs]
    mf1s = [r["f1_macro"] for r in runs]
    accs = [r["accuracy"] for r in runs]

    median_idx = int(np.argsort(wf1s)[len(wf1s) // 2])
    representative = runs[median_idx]

    # Save representative model
    xgb_model = representative.pop("model")
    for r in runs:
        if "model" in r:
            del r["model"]

    output_dir = Path(__file__).parent / "dualstream_model"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    if scaler is not None:
        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        (output_dir / "scaler.pkl").unlink(missing_ok=True)

    # Clean up old model files from previous architecture
    for old_file in ["lr_model.pkl", "rf_model.pkl", "meta_model.pkl"]:
        (output_dir / old_file).unlink(missing_ok=True)

    metadata = {
        "protocol": "Leak-free: Single XGBoost, Val for Optuna tuning, Test evaluated once, 5-seed repeated",
        "architecture": "Raw concat (WavLM 768-d + PhoBERT 768-d) → XGBoost",
        "text_gate": "Hard invalid-text fallback (empty transcript → zero vector)",
        "scaler": "per-stream" if args.use_scaler else "none",
        "split_source": "split_manifest.json",
        "emotion_labels": emotion_labels,
        "split_checksum": split_checksum,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "test_accuracy_mean": float(np.mean(accs)),
        "test_accuracy_std": float(np.std(accs)),
        "test_f1_weighted_mean": float(np.mean(wf1s)),
        "test_f1_weighted_std": float(np.std(wf1s)),
        "test_f1_macro_mean": float(np.mean(mf1s)),
        "test_f1_macro_std": float(np.std(mf1s)),
        "representative_run": representative,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Models saved to: {output_dir}/")
    print(f"✓ Mean test F1 (weighted): {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")


if __name__ == "__main__":
    main()
