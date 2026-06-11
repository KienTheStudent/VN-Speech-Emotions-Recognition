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

import os
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
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

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
    inputs = tokenizer(segmented, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.pooler_output.squeeze().cpu().numpy()
        if features.ndim == 0:
            features = np.expand_dims(features, 0)
    return features


def extract_features_for_split(paths, labels, wavlm_model, wavlm_proc,
                                whisper_model, whisper_proc, phobert_model, phobert_tokenizer, device, split_name, cache):
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
        textual = extract_textual_features(text, phobert_model, phobert_tokenizer, device)
        
        acoustic_list.append(acoustic)
        textual_list.append(textual)
        label_list.append(label)

    if i % 200 == 0 or len(paths) > 0:
        save_transcript_cache(cache)

    fused = np.concatenate(
        [np.array(acoustic_list), np.array(textual_list)], axis=1
    )
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

    print(f"Split sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # ------------------------------------------------------------------
    # 3. Load pretrained models
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("Loading WavLM (SEFE)...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    print("Loading Whisper (TEFE — ASR-derived)...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to(device).eval()

    print("Loading PhoBERT (TEFE — linguistic)...")
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()

    cache = load_transcript_cache()

    # ------------------------------------------------------------------
    # 4. Extract dual-stream features for all splits
    # ------------------------------------------------------------------
    print("\nExtracting dual-stream features...")
    train_fused, train_labels = extract_features_for_split(
        train_paths, train_labels_raw,
        wavlm_model, wavlm_processor, whisper_model, whisper_processor, phobert_model, phobert_tokenizer,
        device, "Train", cache
    )
    val_fused, val_labels = extract_features_for_split(
        val_paths, val_labels_raw,
        wavlm_model, wavlm_processor, whisper_model, whisper_processor, phobert_model, phobert_tokenizer,
        device, "Val", cache
    )
    test_fused, test_labels = extract_features_for_split(
        test_paths, test_labels_raw,
        wavlm_model, wavlm_processor, whisper_model, whisper_processor, phobert_model, phobert_tokenizer,
        device, "Test", cache
    )

    print(f"\nFeatures extracted:")
    print(f"  Train: {train_fused.shape}, Val: {val_fused.shape}, Test: {test_fused.shape}")

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
    # 6. Train base classifiers on TRAIN set only
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIERS")
    print("=" * 60)

    print("\n[1/3] Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(train_fused, train_labels)
    lr_val_pred = lr_model.predict(val_fused)
    lr_val_f1 = f1_score(val_labels, lr_val_pred, average="weighted")
    print(f"  Val F1 (weighted): {lr_val_f1:.4f}")

    print("\n[2/3] Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(train_fused, train_labels)
    rf_val_pred = rf_model.predict(val_fused)
    rf_val_f1 = f1_score(val_labels, rf_val_pred, average="weighted")
    print(f"  Val F1 (weighted): {rf_val_f1:.4f}")

    # ------------------------------------------------------------------
    # 7. Optuna-tuned XGBoost — evaluated on VAL set (not test!)
    # ------------------------------------------------------------------
    print("\n[3/3] XGBoost (Optuna tuning on VAL set)...")

    def xgb_objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "random_state": 42,
        }
        model = XGBClassifier(**params, eval_metric="mlogloss")
        model.fit(train_fused, train_labels)
        pred = model.predict(val_fused)
        return f1_score(val_labels, pred, average="weighted")

    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=10, show_progress_bar=True)

    print(f"  Best trial val F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    xgb_model = XGBClassifier(
        **study.best_params, eval_metric="mlogloss"
    )
    xgb_model.fit(train_fused, train_labels)

    # ------------------------------------------------------------------
    # 8. Late Fusion Ensemble — weights optimized on VAL set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OPTIMIZING LATE FUSION WEIGHTS ON VAL SET")
    print("=" * 60)

    # Get probability predictions on VAL from all classifiers
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

    study2 = optuna.create_study(direction="maximize")
    study2.optimize(ensemble_objective, n_trials=100, show_progress_bar=True)

    w1, w2, w3 = study2.best_params["w1"], study2.best_params["w2"], study2.best_params["w3"]
    total = w1 + w2 + w3
    w_xgb, w_rf, w_lr = w1 / total, w2 / total, w3 / total

    print(f"\nBest ensemble weights:")
    print(f"  XGBoost: {w_xgb:.4f}")
    print(f"  Random Forest: {w_rf:.4f}")
    print(f"  Logistic Regression: {w_lr:.4f}")
    print(f"  Best val ensemble F1: {study2.best_value:.4f}")

    # ------------------------------------------------------------------
    # 9. FINAL TEST EVALUATION (exactly once)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (leak-free)")
    print("=" * 60)

    # Individual classifier results on test
    print("\nIndividual classifiers on TEST:")
    for name, clf in [("LR", lr_model), ("RF", rf_model), ("XGB", xgb_model)]:
        pred = clf.predict(test_fused)
        wf1 = f1_score(test_labels, pred, average="weighted")
        mf1 = f1_score(test_labels, pred, average="macro")
        acc = accuracy_score(test_labels, pred)
        print(f"  {name}: wF1={wf1:.4f}  mF1={mf1:.4f}  Acc={acc:.4f}")

    # Ensemble on test
    lr_test_proba = lr_model.predict_proba(test_fused)
    rf_test_proba = rf_model.predict_proba(test_fused)
    xgb_test_proba = xgb_model.predict_proba(test_fused)

    ensemble_test_proba = w_xgb * xgb_test_proba + w_rf * rf_test_proba + w_lr * lr_test_proba
    ensemble_test_pred = np.argmax(ensemble_test_proba, axis=1)

    test_f1_weighted = f1_score(test_labels, ensemble_test_pred, average="weighted")
    test_f1_macro = f1_score(test_labels, ensemble_test_pred, average="macro")
    test_acc = accuracy_score(test_labels, ensemble_test_pred)

    print(f"\nLate Fusion Ensemble on TEST:")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  F1 Weighted: {test_f1_weighted:.4f}")
    print(f"  F1 Macro:    {test_f1_macro:.4f}")

    report_str = classification_report(
        test_labels, ensemble_test_pred, target_names=emotion_labels
    )
    print(f"\nClassification Report:\n{report_str}")

    report_dict = classification_report(
        test_labels, ensemble_test_pred, target_names=emotion_labels, output_dict=True
    )
    cm = confusion_matrix(test_labels, ensemble_test_pred).tolist()
    print("Confusion Matrix:")
    print(np.array(cm))

    # ------------------------------------------------------------------
    # 10. Save models and metadata
    # ------------------------------------------------------------------
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
        "protocol": "Leak-free: Val for Optuna tuning & ensemble weights, Test evaluated once",
        "split_source": "split_manifest.json",
        "emotion_labels": emotion_labels,
        "ensemble_weights": {
            "xgb": float(w_xgb),
            "rf": float(w_rf),
            "lr": float(w_lr),
        },
        "xgboost_best_params": study.best_params,
        "xgboost_best_val_f1": float(study.best_value),
        "ensemble_best_val_f1": float(study2.best_value),
        "test_accuracy": float(test_acc),
        "test_f1_weighted": float(test_f1_weighted),
        "test_f1_macro": float(test_f1_macro),
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Models saved to: {output_dir}/")
    print(f"✓ Final test F1 (weighted): {test_f1_weighted:.4f}")


if __name__ == "__main__":
    main()