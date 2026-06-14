#!/usr/bin/env python3
"""DFAT Ablation Study — Evaluating individual stream contributions.

Ablation configurations:
  1. Acoustic-only   : WavLM embeddings (768-d) + classifiers
  2. Linguistic-only : PhoBERT embeddings via Whisper-small ASR (768-d) + classifiers
  3. Early Fusion    : Concat WavLM + PhoBERT (1536-d) + classifiers
  4. Late Fusion     : Weighted ensemble of acoustic-only and linguistic-only predictions
  5. ASR sensitivity : Repeat linguistic-only and early/late fusion using Whisper-tiny
  6. Stress test     : Synthetic word-level perturbation (10%, 20%, 30%) on Whisper-small text

All ablations use the same Train/Val/Test split from split_manifest.json.
Classifiers: LogReg, RandomForest, XGBoost (Optuna-tuned on Val).
"""

import json
import random
import re
import time
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

from underthesea import word_tokenize as vi_word_tokenize

MANIFEST_PATH = Path(__file__).parent.parent / "split_manifest.json"
TRANSCRIPT_CACHE_FILE = Path(__file__).parent / "transcript_cache.json"
TRANSCRIPT_CACHE_TINY_FILE = Path(__file__).parent / "transcript_cache_tiny.json"

SEEDS = [42, 123, 456, 789, 2026]


# ==================== CACHE UTILITIES ====================

def load_cache(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==================== FEATURE EXTRACTION ====================

def load_audio(path_dict, sr=16000):
    import io as _io
    try:
        if isinstance(path_dict, dict) and "bytes" in path_dict:
            audio, _ = librosa.load(_io.BytesIO(path_dict["bytes"]), sr=sr)
        elif isinstance(path_dict, str):
            audio, _ = librosa.load(path_dict, sr=sr)
        elif isinstance(path_dict, dict) and "path" in path_dict:
            audio, _ = librosa.load(path_dict["path"], sr=sr)
        else:
            return None
        return audio
    except Exception:
        return None


def extract_wavlm(audio, model, processor, device):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def transcribe(audio, whisper_model, whisper_proc, device):
    inputs = whisper_proc(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.input_features.to(device)
    with torch.no_grad():
        ids = whisper_model.generate(inputs, max_new_tokens=100)
        text = whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]
    return text


def extract_phobert(text, phobert_model, phobert_tok, device, use_segmentation=True):
    if use_segmentation:
        segmented = vi_word_tokenize(text, format="text") if text.strip() else ""
    else:
        segmented = text.strip()
        
    if not segmented.strip():
        return np.zeros(768, dtype=np.float32)
    inputs = phobert_tok(segmented, return_tensors="pt", padding=True,
                         truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = phobert_model(**inputs)
        feat = out.pooler_output.squeeze().cpu().numpy()
        if feat.ndim == 0:
            feat = np.expand_dims(feat, 0)
    return feat


def extract_all_features(paths, labels, wavlm_model, wavlm_proc,
                         whisper_model, whisper_proc,
                         phobert_model, phobert_tok,
                         device, split_name, cache):
    """Extract acoustic + linguistic features for a split."""
    acoustic_list, linguistic_list, label_list, valid_paths = [], [], [], []
    for i, (p, y) in enumerate(zip(paths, labels)):
        if i % 200 == 0:
            print(f"  [{split_name}] {i}/{len(paths)}")
        audio = load_audio(p)
        if audio is None:
            continue

        # Acoustic
        ac = extract_wavlm(audio, wavlm_model, wavlm_proc, device)

        # Linguistic (with cache)
        key = str(p)
        if key in cache:
            text = cache[key]
        else:
            text = transcribe(audio, whisper_model, whisper_proc, device)
            cache[key] = text

        lf = extract_phobert(text, phobert_model, phobert_tok, device, use_segmentation=True)

        acoustic_list.append(ac)
        linguistic_list.append(lf)
        label_list.append(y)
        valid_paths.append(str(p))

    print(f"  [{split_name}] {len(label_list)}/{len(paths)} valid")
    return np.array(acoustic_list), np.array(linguistic_list), np.array(label_list), valid_paths


# ==================== TEXT PERTURBATION ====================

def perturb_text(text, error_rate, rng):
    """Simulate ASR errors by randomly deleting/substituting words."""
    words = text.split()
    if len(words) == 0:
        return text
    n_errors = max(1, int(len(words) * error_rate))
    indices = rng.sample(range(len(words)), min(n_errors, len(words)))
    result = []
    for i, w in enumerate(words):
        if i in indices:
            action = rng.choice(["delete", "substitute", "insert"])
            if action == "delete":
                continue
            elif action == "substitute":
                result.append(rng.choice(["uh", "à", "ờ", "hm", "vậy", "thì"]))
            else:
                result.append(rng.choice(["uh", "à", "ờ"]))
                result.append(w)
        else:
            result.append(w)
    return " ".join(result) if result else text


# ==================== EVALUATION ====================

def run_classifiers(x_train, y_train, x_val, y_val, x_test, y_test,
                    label_names, config_name, seed=42, use_scaler=True, use_optuna=True):
    """Train LR, RF, XGB on train; tune XGB on val; report on test."""
    if use_scaler:
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_val_s = scaler.transform(x_val)
        x_test_s = scaler.transform(x_test)
    else:
        x_train_s, x_val_s, x_test_s = x_train, x_val, x_test

    # LR
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(x_train_s, y_train)
    lr_f1 = f1_score(y_test, lr.predict(x_test_s), average="weighted")

    # RF
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(x_train_s, y_train)
    rf_f1 = f1_score(y_test, rf.predict(x_test_s), average="weighted")

    # XGBoost
    if use_optuna:
        def xgb_obj(trial):
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
            m = XGBClassifier(**params, eval_metric="mlogloss")
            m.fit(x_train_s, y_train)
            return f1_score(y_val, m.predict(x_val_s), average="weighted")

        study = optuna.create_study(direction="maximize")
        study.optimize(xgb_obj, n_trials=10, show_progress_bar=False)
        xgb = XGBClassifier(**study.best_params, eval_metric="mlogloss")
    else:
        xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.08, eval_metric="mlogloss", random_state=seed)
        
    xgb.fit(x_train_s, y_train)
    xgb_f1 = f1_score(y_test, xgb.predict(x_test_s), average="weighted")

    # Late fusion ensemble (optimize weights on val)
    lr_val_p = lr.predict_proba(x_val_s)
    rf_val_p = rf.predict_proba(x_val_s)
    xgb_val_p = xgb.predict_proba(x_val_s)

    def ens_obj(trial):
        w1 = trial.suggest_float("w1", 0, 1)
        w2 = trial.suggest_float("w2", 0, 1)
        w3 = trial.suggest_float("w3", 0, 1)
        total = w1 + w2 + w3
        if total == 0:
            return 0.0
        p = (w1/total)*xgb_val_p + (w2/total)*rf_val_p + (w3/total)*lr_val_p
        return f1_score(y_val, np.argmax(p, axis=1), average="weighted")

    study2 = optuna.create_study(direction="maximize")
    study2.optimize(ens_obj, n_trials=50, show_progress_bar=False)

    w1, w2, w3 = study2.best_params["w1"], study2.best_params["w2"], study2.best_params["w3"]
    total = w1 + w2 + w3
    w_xgb, w_rf, w_lr = w1/total, w2/total, w3/total

    lr_test_p = lr.predict_proba(x_test_s)
    rf_test_p = rf.predict_proba(x_test_s)
    xgb_test_p = xgb.predict_proba(x_test_s)
    ens_p = w_xgb * xgb_test_p + w_rf * rf_test_p + w_lr * lr_test_p
    ens_pred = np.argmax(ens_p, axis=1)

    ens_wf1 = f1_score(y_test, ens_pred, average="weighted")
    ens_mf1 = f1_score(y_test, ens_pred, average="macro")
    ens_acc = accuracy_score(y_test, ens_pred)

    report = classification_report(y_test, ens_pred, target_names=label_names, output_dict=True)
    cm = confusion_matrix(y_test, ens_pred).tolist()

    result = {
        "config": config_name,
        "individual_classifiers": {
            "LR_wF1": round(lr_f1, 4),
            "RF_wF1": round(rf_f1, 4),
            "XGB_wF1": round(xgb_f1, 4),
        },
        "ensemble": {
            "weights": {"xgb": round(w_xgb, 4), "rf": round(w_rf, 4), "lr": round(w_lr, 4)},
            "f1_weighted": round(ens_wf1, 4),
            "f1_macro": round(ens_mf1, 4),
            "accuracy": round(ens_acc, 4),
            "classification_report": report,
            "confusion_matrix": cm,
        },
    }
    print(f"    {config_name}: LR={lr_f1:.4f}  RF={rf_f1:.4f}  XGB={xgb_f1:.4f}  Ensemble={ens_wf1:.4f}")
    return result


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("DFAT ABLATION STUDY — Leak-Free Protocol")
    print("=" * 60)

    # 1. Load data
    manifest_path = MANIFEST_PATH
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    ds = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    label_names = le.classes_.tolist()

    train_idx = manifest["train_indices"]
    val_idx = manifest["val_indices"]
    test_idx = manifest["test_indices"]

    train_paths = df["path"].iloc[train_idx].values
    train_labels = df["label"].iloc[train_idx].values
    val_paths = df["path"].iloc[val_idx].values
    val_labels = df["label"].iloc[val_idx].values
    test_paths = df["path"].iloc[test_idx].values
    test_labels = df["label"].iloc[test_idx].values

    print(f"Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # 2. Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading WavLM...")
    wavlm_proc = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    print("Loading Whisper-small...")
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()

    print("Loading PhoBERT...")
    phobert_tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device).eval()

    # 3. Extract features using Whisper-small
    cache_small = load_cache(TRANSCRIPT_CACHE_FILE)

    print("\nExtracting features (Whisper-small)...")
    train_ac, train_ling, train_y, train_vp = extract_all_features(
        train_paths, train_labels, wavlm_model, wavlm_proc,
        whisper_model, whisper_proc, phobert_model, phobert_tok,
        device, "Train", cache_small)
    val_ac, val_ling, val_y, val_vp = extract_all_features(
        val_paths, val_labels, wavlm_model, wavlm_proc,
        whisper_model, whisper_proc, phobert_model, phobert_tok,
        device, "Val", cache_small)
    test_ac, test_ling, test_y, test_vp = extract_all_features(
        test_paths, test_labels, wavlm_model, wavlm_proc,
        whisper_model, whisper_proc, phobert_model, phobert_tok,
        device, "Test", cache_small)

    save_cache(cache_small, TRANSCRIPT_CACHE_FILE)

    # Build acoustic path dictionaries for exact alignment
    ac_dict_train = {p: ac for p, ac in zip(train_vp, train_ac)}
    ac_dict_val = {p: ac for p, ac in zip(val_vp, val_ac)}
    ac_dict_test = {p: ac for p, ac in zip(test_vp, test_ac)}
    ac_dicts = {"train": ac_dict_train, "val": ac_dict_val, "test": ac_dict_test}

    # Fused features
    train_fused = np.concatenate([train_ac, train_ling], axis=1)
    val_fused = np.concatenate([val_ac, val_ling], axis=1)
    test_fused = np.concatenate([test_ac, test_ling], axis=1)

    print(f"\nFeature dims: Acoustic={train_ac.shape[1]}, "
          f"Linguistic={train_ling.shape[1]}, Fused={train_fused.shape[1]}")

    # ------------------------------------------------------------------
    # 4. Run ablation configurations
    # ------------------------------------------------------------------
    ablation_results = []

    print("\n" + "=" * 60)
    print("ABLATION 1: Acoustic-only (WavLM)")
    print("=" * 60)
    ablation_results.append(run_classifiers(
        train_ac, train_y, val_ac, val_y, test_ac, test_y,
        label_names, "Acoustic-only (WavLM)"))

    print("\n" + "=" * 60)
    print("ABLATION 2: Linguistic-only (PhoBERT via Whisper-small)")
    print("=" * 60)
    ablation_results.append(run_classifiers(
        train_ling, train_y, val_ling, val_y, test_ling, test_y,
        label_names, "Linguistic-only (Whisper-small + PhoBERT)"))

    print("\n" + "=" * 60)
    print("ABLATION 3: Early Fusion (Concat)")
    print("=" * 60)
    ablation_results.append(run_classifiers(
        train_fused, train_y, val_fused, val_y, test_fused, test_y,
        label_names, "Early Fusion (Concat 1536-d)"))

    print("\n" + "=" * 60)
    print("ABLATION 3b: Early Fusion (No StandardScaler)")
    print("=" * 60)
    ablation_results.append(run_classifiers(
        train_fused, train_y, val_fused, val_y, test_fused, test_y,
        label_names, "Early Fusion (No StandardScaler)", use_scaler=False))

    print("\n" + "=" * 60)
    print("ABLATION 3c: Early Fusion (No Optuna tuning)")
    print("=" * 60)
    ablation_results.append(run_classifiers(
        train_fused, train_y, val_fused, val_y, test_fused, test_y,
        label_names, "Early Fusion (No Optuna tuning)", use_optuna=False))

    print("\n" + "=" * 60)
    print("ABLATION 3d: Early Fusion (No word segmentation)")
    print("=" * 60)
    
    # Extract PhoBERT without word segmentation
    print("Re-extracting PhoBERT without word segmentation...")
    def get_noseg_ling(paths, labels, cache_ref, ac_dict):
        feat_list, lbl_list, valid_ac_list = [], [], []
        for p, y in zip(paths, labels):
            key = str(p)
            if key not in ac_dict: continue
            text = cache_ref.get(key, "")
            feat = extract_phobert(text, phobert_model, phobert_tok, device, use_segmentation=False)
            feat_list.append(feat)
            lbl_list.append(y)
            valid_ac_list.append(ac_dict[key])
        return np.array(valid_ac_list), np.array(feat_list), np.array(lbl_list)

    train_ac_ns, train_ling_ns, train_y_ns = get_noseg_ling(train_paths, train_labels, cache_small, ac_dict_train)
    val_ac_ns, val_ling_ns, val_y_ns = get_noseg_ling(val_paths, val_labels, cache_small, ac_dict_val)
    test_ac_ns, test_ling_ns, test_y_ns = get_noseg_ling(test_paths, test_labels, cache_small, ac_dict_test)

    train_fused_ns = np.concatenate([train_ac_ns, train_ling_ns], axis=1)
    val_fused_ns = np.concatenate([val_ac_ns, val_ling_ns], axis=1)
    test_fused_ns = np.concatenate([test_ac_ns, test_ling_ns], axis=1)

    ablation_results.append(run_classifiers(
        train_fused_ns, train_y_ns, val_fused_ns, val_y_ns, test_fused_ns, test_y_ns,
        label_names, "Early Fusion (No word segmentation)"))

    # ABLATION 4: Late Fusion of acoustic-only + linguistic-only predictions
    print("\n" + "=" * 60)
    print("ABLATION 4: Late Fusion (Acoustic + Linguistic streams)")
    print("=" * 60)
    # Train separate classifiers on each stream
    scaler_ac = StandardScaler()
    scaler_lg = StandardScaler()
    x_tr_ac = scaler_ac.fit_transform(train_ac)
    x_va_ac = scaler_ac.transform(val_ac)
    x_te_ac = scaler_ac.transform(test_ac)
    x_tr_lg = scaler_lg.fit_transform(train_ling)
    x_va_lg = scaler_lg.transform(val_ling)
    x_te_lg = scaler_lg.transform(test_ling)

    rf_ac = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_ac.fit(x_tr_ac, train_y)
    rf_lg = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_lg.fit(x_tr_lg, train_y)

    # Optimize fusion weight on val
    def late_obj(trial):
        w_a = trial.suggest_float("w_acoustic", 0, 1)
        w_l = 1 - w_a
        p = w_a * rf_ac.predict_proba(x_va_ac) + w_l * rf_lg.predict_proba(x_va_lg)
        return f1_score(val_y, np.argmax(p, axis=1), average="weighted")

    study_late = optuna.create_study(direction="maximize")
    study_late.optimize(late_obj, n_trials=50, show_progress_bar=False)
    w_a = study_late.best_params["w_acoustic"]
    w_l = 1 - w_a

    late_pred = np.argmax(
        w_a * rf_ac.predict_proba(x_te_ac) + w_l * rf_lg.predict_proba(x_te_lg), axis=1
    )
    late_wf1 = f1_score(test_y, late_pred, average="weighted")
    late_mf1 = f1_score(test_y, late_pred, average="macro")
    late_acc = accuracy_score(test_y, late_pred)
    late_report = classification_report(test_y, late_pred, target_names=label_names, output_dict=True)
    late_cm = confusion_matrix(test_y, late_pred).tolist()

    late_result = {
        "config": "Late Fusion (stream-level)",
        "stream_weights": {"acoustic": round(w_a, 4), "linguistic": round(w_l, 4)},
        "ensemble": {
            "f1_weighted": round(late_wf1, 4),
            "f1_macro": round(late_mf1, 4),
            "accuracy": round(late_acc, 4),
            "classification_report": late_report,
            "confusion_matrix": late_cm,
        },
    }
    ablation_results.append(late_result)
    print(f"    Late Fusion: wF1={late_wf1:.4f}  (w_ac={w_a:.3f}, w_lg={w_l:.3f})")

    # ------------------------------------------------------------------
    # 5. ASR Sensitivity: Whisper-tiny
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 5: ASR Sensitivity — Whisper-tiny")
    print("=" * 60)

    # Unload Whisper-small, load Whisper-tiny
    del whisper_model
    torch.cuda.empty_cache()
    print("Loading Whisper-tiny...")
    whisper_tiny_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_tiny_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny").to(device).eval()

    cache_tiny = load_cache(TRANSCRIPT_CACHE_TINY_FILE)

    print("Extracting features (Whisper-tiny)...")
    _, train_ling_tiny, train_y_tiny, train_vp_tiny = extract_all_features(
        train_paths, train_labels, wavlm_model, wavlm_proc,
        whisper_tiny_model, whisper_tiny_proc, phobert_model, phobert_tok,
        device, "Train-tiny", cache_tiny)
    _, val_ling_tiny, val_y_tiny, val_vp_tiny = extract_all_features(
        val_paths, val_labels, wavlm_model, wavlm_proc,
        whisper_tiny_model, whisper_tiny_proc, phobert_model, phobert_tok,
        device, "Val-tiny", cache_tiny)
    _, test_ling_tiny, test_y_tiny, test_vp_tiny = extract_all_features(
        test_paths, test_labels, wavlm_model, wavlm_proc,
        whisper_tiny_model, whisper_tiny_proc, phobert_model, phobert_tok,
        device, "Test-tiny", cache_tiny)

    save_cache(cache_tiny, TRANSCRIPT_CACHE_TINY_FILE)

    # Linguistic-only with Whisper-tiny
    ablation_results.append(run_classifiers(
        train_ling_tiny, train_y_tiny, val_ling_tiny, val_y_tiny,
        test_ling_tiny, test_y_tiny,
        label_names, "Linguistic-only (Whisper-tiny + PhoBERT)"))

    # Early fusion with Whisper-tiny using explicit path alignment
    train_ac_tiny = np.array([ac_dict_train[p] for p in train_vp_tiny])
    val_ac_tiny = np.array([ac_dict_val[p] for p in val_vp_tiny])
    test_ac_tiny = np.array([ac_dict_test[p] for p in test_vp_tiny])

    train_fused_tiny = np.concatenate([train_ac_tiny, train_ling_tiny], axis=1)
    val_fused_tiny = np.concatenate([val_ac_tiny, val_ling_tiny], axis=1)
    test_fused_tiny = np.concatenate([test_ac_tiny, test_ling_tiny], axis=1)

    ablation_results.append(run_classifiers(
        train_fused_tiny, train_y_tiny, val_fused_tiny, val_y_tiny,
        test_fused_tiny, test_y_tiny,
        label_names, "Early Fusion (Whisper-tiny)"))

    del whisper_tiny_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6. Stress test: synthetic word perturbation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STRESS TEST: Synthetic word-level perturbation on Whisper-small text")
    print("=" * 60)

    for error_rate in [0.10, 0.20, 0.30]:
        rng = random.Random(42)
        print(f"\n  Error rate: {error_rate*100:.0f}%")

        # Re-extract PhoBERT features with perturbed text
        perturbed_sets = {}
        for name, paths, labels, cache_ref, ac_dict in [
            ("train", train_paths, train_labels, cache_small, ac_dict_train),
            ("val", val_paths, val_labels, cache_small, ac_dict_val),
            ("test", test_paths, test_labels, cache_small, ac_dict_test),
        ]:
            feat_list, lbl_list, valid_ac_list = [], [], []
            for p, y in zip(paths, labels):
                key = str(p)
                if key not in ac_dict:
                    continue  # Ensure we only keep samples that loaded correctly in acoustics
                text = cache_ref.get(key, "")
                perturbed = perturb_text(text, error_rate, rng)
                feat = extract_phobert(perturbed, phobert_model, phobert_tok, device)
                feat_list.append(feat)
                lbl_list.append(y)
                valid_ac_list.append(ac_dict[key])
            perturbed_sets[name] = (np.array(valid_ac_list), np.array(feat_list), np.array(lbl_list))

        # Early fusion with perturbed linguistic
        p_train_fused = np.concatenate([perturbed_sets["train"][0], perturbed_sets["train"][1]], axis=1)
        p_val_fused = np.concatenate([perturbed_sets["val"][0], perturbed_sets["val"][1]], axis=1)
        p_test_fused = np.concatenate([perturbed_sets["test"][0], perturbed_sets["test"][1]], axis=1)

        ablation_results.append(run_classifiers(
            p_train_fused, perturbed_sets["train"][2],
            p_val_fused, perturbed_sets["val"][2],
            p_test_fused, perturbed_sets["test"][2],
            label_names,
            f"Early Fusion + {error_rate*100:.0f}% synthetic noise"))

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    output = {
        "protocol": "DFAT ablation study — leak-free, speaker-independent split",
        "split_source": "split_manifest.json",
        "emotion_labels": label_names,
        "ablation_results": ablation_results,
    }

    output_path = Path(__file__).parent / "ablation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    for r in ablation_results:
        ens = r.get("ensemble", {})
        wf1 = ens.get("f1_weighted", "N/A")
        print(f"  {r['config']:50s}  wF1={wf1}")

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
