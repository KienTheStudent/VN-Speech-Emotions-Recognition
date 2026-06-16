#!/usr/bin/env python3
"""
DFAT Hybrid Fusion Training — Leak-Free Protocol.

Dual-stream Feature Aggregation with Acoustic (WavLM) and
ASR-derived Linguistic (Whisper encoder) features.

Reads the fixed Train/Val/Test split from split_manifest.json.
- Train set: used for fitting classifiers.
- Val set: used for Optuna hyperparameter tuning (XGBoost) and
  ensemble weight optimization (Late Fusion).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

import os
import sys
# Removed sys.path logic to keep import topology clean

from underthesea import word_tokenize as vi_word_tokenize

MANIFEST_PATH = Path(__file__).parent.parent / "split_manifest.json"
TRANSCRIPT_CACHE_FILE = Path(__file__).parent / "transcript_cache.json"


def load_transcript_cache():
    if TRANSCRIPT_CACHE_FILE.exists():
        with open(TRANSCRIPT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_transcript_cache(cache):
    with open(TRANSCRIPT_CACHE_FILE, "w", encoding="utf-8") as f:
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
    """Extract dual-stream features for a given split."""
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

        acoustic_list.append(acoustic)
        textual_list.append(textual)
        label_list.append(label)

    if (i + 1) % 200 == 0 or (i + 1) == len(paths):
        save_transcript_cache(cache)

    fused = np.concatenate([np.array(acoustic_list), np.array(textual_list)], axis=1)
    print(f"  [{split_name}] Valid samples: {len(label_list)} / {len(paths)}")
    return fused, np.array(label_list)


# ==================== MAIN ====================


def main():
    print("=" * 60)
    print("DFAT HYBRID FUSION — LEAK-FREE PROTOCOL")
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

    print("Loading Whisper (TEFE — ASR-derived)...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = (
        WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        .to(device)
        .eval()
    )

    print("Loading PhoBERT (TEFE — linguistic)...")
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()

    cache = load_transcript_cache()

    # ------------------------------------------------------------------
    # 4. Extract dual-stream features for all splits
    # ------------------------------------------------------------------
    print("\nExtracting dual-stream features...")
    train_fused, train_labels = extract_features_for_split(
        train_paths,
        train_labels_raw,
        wavlm_model,
        wavlm_processor,
        whisper_model,
        whisper_processor,
        phobert_model,
        phobert_tokenizer,
        device,
        "Train",
        cache,
    )
    val_fused, val_labels = extract_features_for_split(
        val_paths,
        val_labels_raw,
        wavlm_model,
        wavlm_processor,
        whisper_model,
        whisper_processor,
        phobert_model,
        phobert_tokenizer,
        device,
        "Val",
        cache,
    )
    test_fused, test_labels = extract_features_for_split(
        test_paths,
        test_labels_raw,
        wavlm_model,
        wavlm_processor,
        whisper_model,
        whisper_processor,
        phobert_model,
        phobert_tokenizer,
        device,
        "Test",
        cache,
    )

    print(f"\nFeatures extracted:")
    print(
        f"  Train: {train_fused.shape}, Val: {val_fused.shape}, Test: {test_fused.shape}"
    )

    # Free GPU memory
    del wavlm_model, whisper_model, phobert_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Normalize features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    train_fused = scaler.fit_transform(train_fused)
    val_fused = scaler.transform(val_fused)
    test_fused = scaler.transform(test_fused)

    # ------------------------------------------------------------------
    # 6. Train base classifiers on TRAIN set only (5 seeds)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIERS (5 Seeds)")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 2026]
    runs = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        
        lr_model = LogisticRegression(max_iter=1000, random_state=seed)
        lr_model.fit(train_fused, train_labels)

        rf_model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_model.fit(train_fused, train_labels)

        def xgb_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "random_state": seed,
            }
            model = XGBClassifier(**params, eval_metric="mlogloss")
            model.fit(train_fused, train_labels)
            pred = model.predict(val_fused)
            return f1_score(val_labels, pred, average="weighted")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

        xgb_model = XGBClassifier(**study.best_params, eval_metric="mlogloss")
        xgb_model.fit(train_fused, train_labels)

        # Ensemble
        lr_val_proba = lr_model.predict_proba(val_fused)
        rf_val_proba = rf_model.predict_proba(val_fused)
        xgb_val_proba = xgb_model.predict_proba(val_fused)

        def ensemble_objective(trial):
            w1 = trial.suggest_float("w1", 0, 1)
            w2 = trial.suggest_float("w2", 0, 1)
            w3 = trial.suggest_float("w3", 0, 1)
            total = w1 + w2 + w3
            if total == 0:
                return 0.0
            ensemble_proba = (
                (w1 / total) * xgb_val_proba
                + (w2 / total) * rf_val_proba
                + (w3 / total) * lr_val_proba
            )
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            return f1_score(val_labels, ensemble_pred, average="weighted")

        study2 = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study2.optimize(ensemble_objective, n_trials=50, show_progress_bar=False)

        w1, w2, w3 = study2.best_params["w1"], study2.best_params["w2"], study2.best_params["w3"]
        total = w1 + w2 + w3
        w_xgb, w_rf, w_lr = w1 / total, w2 / total, w3 / total

        # Evaluate on test
        lr_test_proba = lr_model.predict_proba(test_fused)
        rf_test_proba = rf_model.predict_proba(test_fused)
        xgb_test_proba = xgb_model.predict_proba(test_fused)

        ensemble_test_proba = (w_xgb * xgb_test_proba + w_rf * rf_test_proba + w_lr * lr_test_proba)
        ensemble_test_pred = np.argmax(ensemble_test_proba, axis=1)

        test_f1_weighted = f1_score(test_labels, ensemble_test_pred, average="weighted")
        test_f1_macro = f1_score(test_labels, ensemble_test_pred, average="macro")
        test_acc = accuracy_score(test_labels, ensemble_test_pred)

        report_dict = classification_report(test_labels, ensemble_test_pred, target_names=emotion_labels, output_dict=True)
        cm = confusion_matrix(test_labels, ensemble_test_pred).tolist()

        run_data = {
            "seed": seed,
            "f1_weighted": float(test_f1_weighted),
            "f1_macro": float(test_f1_macro),
            "accuracy": float(test_acc),
            "ensemble_weights": {"xgb": float(w_xgb), "rf": float(w_rf), "lr": float(w_lr)},
            "xgboost_best_params": study.best_params,
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "models": (lr_model, rf_model, xgb_model)
        }
        runs.append(run_data)
        print(f"  -> Seed {seed} Test wF1: {test_f1_weighted:.4f}")

    # Calculate stats
    wf1s = [r["f1_weighted"] for r in runs]
    mf1s = [r["f1_macro"] for r in runs]
    accs = [r["accuracy"] for r in runs]

    median_idx = int(np.argsort(wf1s)[len(wf1s) // 2])
    representative = runs[median_idx]
    
    # Save representative models
    lr_model, rf_model, xgb_model = representative.pop("models")
    for r in runs:
        if "models" in r:
            del r["models"]

    output_dir = Path(__file__).parent / "dualstream_model"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(output_dir / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(output_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "protocol": "Leak-free: Val for Optuna tuning & ensemble weights, Test evaluated once, 5-seed repeated",
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
        "representative_run": representative
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Models saved to: {output_dir}/")
    print(f"✓ Mean test F1 (weighted): {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")


if __name__ == "__main__":
    main()
