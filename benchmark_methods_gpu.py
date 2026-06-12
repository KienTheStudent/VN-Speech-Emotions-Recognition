#!/usr/bin/env python3
"""Unified SER benchmark — Leak-Free Protocol, 5-seed repeated evaluation.

Evaluates:
  PRIMARY BASELINES (in main thesis body):
    1. MFCC + RandomForest
    2. ECAPA-TDNN           (results read from ECAPA/emotion_model/metadata.json)
    3. DFAT Late Fusion     (results read from DFAT_Hybrid_Fusion/dualstream_model/metadata.json)

  SECONDARY BASELINES (appendix / supplementary):
    4. MFCC + SVM
    5. MFCC + XGBoost
    6. WavLM + LogReg
    7. WavLM + SVM

Latency is reported as:
  - feature_extraction_ms : time to extract features (MFCC or WavLM) per sample
  - classifier_ms         : time to run classifier inference per sample
  - total_e2e_ms          : feature_extraction_ms + classifier_ms

Classical ML classifiers are trained 5× with different random seeds.
Mean and std of weighted-F1, macro-F1, and accuracy are reported.

All methods read the fixed Train/Val/Test split from split_manifest.json.
"""

import io
import json
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
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
from sklearn.svm import SVC
from transformers import AutoFeatureExtractor, AutoModel
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MANIFEST_PATH = Path(__file__).parent / "split_manifest.json"
SEEDS = [42, 123, 456, 789, 2026]


# ==================== UTILITIES ====================


def load_manifest():
    """Load the fixed split manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest loaded: {manifest['total_samples']} total samples")
    print(f"  Train: {len(manifest['train_indices'])}")
    print(f"  Val:   {len(manifest['val_indices'])}")
    print(f"  Test:  {len(manifest['test_indices'])}")
    return manifest


def load_audio(path_dict, sr=16000):
    try:
        if isinstance(path_dict, dict) and "bytes" in path_dict:
            audio, _ = librosa.load(io.BytesIO(path_dict["bytes"]), sr=sr)
            return audio
        if isinstance(path_dict, dict) and "path" in path_dict:
            audio, _ = librosa.load(path_dict["path"], sr=sr)
            return audio
        if isinstance(path_dict, str):
            audio, _ = librosa.load(path_dict, sr=sr)
            return audio
    except Exception:
        return None
    return None


def mfcc_feature(audio, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def wavlm_feature(audio, extractor, model, device):
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def evaluate_single_seed(name, clf_factory, x_train, y_train, x_test, y_test, label_names, seed):
    """Train a classifier with a given seed, evaluate on test set."""
    clf = clf_factory(seed)
    clf.fit(x_train, y_train)

    # Measure classifier inference latency
    start = time.perf_counter()
    y_pred = clf.predict(x_test)
    elapsed = time.perf_counter() - start
    classifier_ms = (elapsed / len(x_test)) * 1000

    w_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    m_f1 = float(f1_score(y_test, y_pred, average="macro"))
    acc = float(accuracy_score(y_test, y_pred))

    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "seed": seed,
        "f1_weighted": w_f1,
        "f1_macro": m_f1,
        "accuracy": acc,
        "classifier_ms_per_sample": round(classifier_ms, 4),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate_multi_seed(name, clf_factory, x_train, y_train, x_test, y_test,
                        label_names, feat_extract_ms):
    """Run 5-seed evaluation and compute summary statistics."""
    print(f"\n{'='*50}")
    print(f"  {name} — 5-seed evaluation")
    print(f"{'='*50}")

    runs = []
    for seed in SEEDS:
        result = evaluate_single_seed(name, clf_factory, x_train, y_train,
                                      x_test, y_test, label_names, seed)
        runs.append(result)
        print(f"  seed={seed}: wF1={result['f1_weighted']:.4f}  "
              f"mF1={result['f1_macro']:.4f}  Acc={result['accuracy']:.4f}")

    wf1s = [r["f1_weighted"] for r in runs]
    mf1s = [r["f1_macro"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    clf_lats = [r["classifier_ms_per_sample"] for r in runs]

    # Pick the median-performing seed as the "representative" run for per-class detail
    median_idx = int(np.argsort(wf1s)[len(wf1s) // 2])
    representative = runs[median_idx]

    summary = {
        "method": name,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "f1_weighted_mean": float(np.mean(wf1s)),
        "f1_weighted_std": float(np.std(wf1s)),
        "f1_macro_mean": float(np.mean(mf1s)),
        "f1_macro_std": float(np.std(mf1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "latency": {
            "feature_extraction_ms_per_sample": round(feat_extract_ms, 4),
            "classifier_ms_per_sample": round(float(np.mean(clf_lats)), 4),
            "total_e2e_ms_per_sample": round(feat_extract_ms + float(np.mean(clf_lats)), 4),
        },
        "representative_run": {
            "seed": representative["seed"],
            "f1_weighted": representative["f1_weighted"],
            "f1_macro": representative["f1_macro"],
            "accuracy": representative["accuracy"],
            "classification_report": representative["classification_report"],
            "confusion_matrix": representative["confusion_matrix"],
        },
    }

    print(f"  → Mean wF1: {summary['f1_weighted_mean']:.4f} ± {summary['f1_weighted_std']:.4f}")
    return summary


# ==================== CLASSIFIER FACTORIES ====================

def make_svm_factory(C=10.0):
    def factory(seed):
        return SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)
    return factory

def make_rf_factory(n_estimators=300):
    def factory(seed):
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    return factory

def make_xgb_factory(n_estimators=300, max_depth=8, lr=0.08):
    def factory(seed):
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            eval_metric="mlogloss",
        )
    return factory

def make_logreg_factory():
    def factory(seed):
        return LogisticRegression(max_iter=1000, random_state=seed)
    return factory

def make_wavlm_svm_factory(C=5.0):
    def factory(seed):
        return SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)
    return factory


# ==================== MAIN ====================


def main():
    print("=" * 60)
    print("BENCHMARK: Full ViSEC — Leak-Free, 5-Seed Repeated Eval")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load manifest & dataset
    # ------------------------------------------------------------------
    manifest = load_manifest()
    print("\nLoading ViSEC dataset...")
    ds = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
    df = ds.to_pandas()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    label_names = le.classes_.tolist()

    train_idx = manifest["train_indices"]
    val_idx = manifest["val_indices"]
    test_idx = manifest["test_indices"]

    # Classical ML: combine train+val (no separate tuning needed)
    trainval_idx = train_idx + val_idx

    trainval_paths = df["path"].iloc[trainval_idx].values
    trainval_labels = df["label"].iloc[trainval_idx].values
    test_paths = df["path"].iloc[test_idx].values
    test_labels = df["label"].iloc[test_idx].values

    print(f"\nTrain+Val: {len(trainval_idx)}, Test: {len(test_idx)}")
    print(f"Emotion labels: {label_names}")

    # ------------------------------------------------------------------
    # 2. Extract MFCC features + measure extraction latency
    # ------------------------------------------------------------------
    print("\nExtracting MFCC features...")
    mfcc_train, mfcc_y_train = [], []
    mfcc_extract_times = []
    for i, (p, y) in enumerate(zip(trainval_paths, trainval_labels)):
        if i % 500 == 0:
            print(f"  MFCC train+val: {i}/{len(trainval_paths)}")
        t0 = time.perf_counter()
        audio = load_audio(p)
        if audio is None:
            continue
        feat = mfcc_feature(audio)
        mfcc_extract_times.append(time.perf_counter() - t0)
        mfcc_train.append(feat)
        mfcc_y_train.append(y)

    mfcc_test, mfcc_y_test = [], []
    for i, (p, y) in enumerate(zip(test_paths, test_labels)):
        if i % 200 == 0:
            print(f"  MFCC test: {i}/{len(test_paths)}")
        t0 = time.perf_counter()
        audio = load_audio(p)
        if audio is None:
            continue
        feat = mfcc_feature(audio)
        mfcc_extract_times.append(time.perf_counter() - t0)
        mfcc_test.append(feat)
        mfcc_y_test.append(y)

    mfcc_feat_ms = (np.mean(mfcc_extract_times)) * 1000
    print(f"  MFCC extraction: {mfcc_feat_ms:.2f} ms/sample avg")

    mfcc_train = np.array(mfcc_train)
    mfcc_test = np.array(mfcc_test)
    mfcc_y_train = np.array(mfcc_y_train)
    mfcc_y_test = np.array(mfcc_y_test)

    mfcc_scaler = StandardScaler()
    mfcc_train = mfcc_scaler.fit_transform(mfcc_train)
    mfcc_test = mfcc_scaler.transform(mfcc_test)

    print(f"  MFCC features: train={len(mfcc_train)}, test={len(mfcc_test)}")

    # ------------------------------------------------------------------
    # 3. Extract WavLM embeddings + measure extraction latency
    # ------------------------------------------------------------------
    print("\nExtracting WavLM embeddings on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    wavlm_train, wavlm_y_train = [], []
    wavlm_extract_times = []
    for i, (p, y) in enumerate(zip(trainval_paths, trainval_labels)):
        if i % 500 == 0:
            print(f"  WavLM train+val: {i}/{len(trainval_paths)}")
        audio = load_audio(p)
        if audio is None:
            continue
        t0 = time.perf_counter()
        wavlm_train.append(wavlm_feature(audio, extractor, wavlm, device))
        wavlm_extract_times.append(time.perf_counter() - t0)
        wavlm_y_train.append(y)

    wavlm_test, wavlm_y_test = [], []
    for i, (p, y) in enumerate(zip(test_paths, test_labels)):
        if i % 200 == 0:
            print(f"  WavLM test: {i}/{len(test_paths)}")
        audio = load_audio(p)
        if audio is None:
            continue
        t0 = time.perf_counter()
        wavlm_test.append(wavlm_feature(audio, extractor, wavlm, device))
        wavlm_extract_times.append(time.perf_counter() - t0)
        wavlm_y_test.append(y)

    wavlm_feat_ms = (np.mean(wavlm_extract_times)) * 1000
    print(f"  WavLM extraction: {wavlm_feat_ms:.2f} ms/sample avg")

    wavlm_train = np.array(wavlm_train)
    wavlm_test = np.array(wavlm_test)
    wavlm_y_train = np.array(wavlm_y_train)
    wavlm_y_test = np.array(wavlm_y_test)

    wavlm_scaler = StandardScaler()
    wavlm_train = wavlm_scaler.fit_transform(wavlm_train)
    wavlm_test = wavlm_scaler.transform(wavlm_test)

    print(f"  WavLM features: train={len(wavlm_train)}, test={len(wavlm_test)}")

    # Free GPU memory
    del wavlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Evaluate all methods with 5 seeds
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING METHODS (5 seeds each)")
    print("=" * 60)

    results = []

    # --- PRIMARY BASELINES ---
    # MFCC + RandomForest (primary baseline)
    results.append(evaluate_multi_seed(
        "MFCC+RandomForest", make_rf_factory(300),
        mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test,
        label_names, mfcc_feat_ms,
    ))

    # --- SECONDARY BASELINES ---
    # MFCC + SVM
    results.append(evaluate_multi_seed(
        "MFCC+SVM", make_svm_factory(10.0),
        mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test,
        label_names, mfcc_feat_ms,
    ))

    # MFCC + XGBoost
    results.append(evaluate_multi_seed(
        "MFCC+XGBoost", make_xgb_factory(300, 8, 0.08),
        mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test,
        label_names, mfcc_feat_ms,
    ))

    # WavLM + LogReg
    results.append(evaluate_multi_seed(
        "WavLM+LogReg", make_logreg_factory(),
        wavlm_train, wavlm_y_train, wavlm_test, wavlm_y_test,
        label_names, wavlm_feat_ms,
    ))

    # WavLM + SVM
    results.append(evaluate_multi_seed(
        "WavLM+SVM", make_wavlm_svm_factory(5.0),
        wavlm_train, wavlm_y_train, wavlm_test, wavlm_y_test,
        label_names, wavlm_feat_ms,
    ))

    # ------------------------------------------------------------------
    # 5. Read ECAPA-TDNN and DFAT results from their metadata
    # ------------------------------------------------------------------
    ecapa_meta_path = Path(__file__).parent / "ECAPA" / "emotion_model" / "metadata.json"
    dfat_meta_path = Path(__file__).parent / "DFAT_Hybrid_Fusion" / "dualstream_model" / "metadata.json"

    if ecapa_meta_path.exists():
        with open(ecapa_meta_path, "r") as f:
            ecapa_meta = json.load(f)
        ecapa_entry = {
            "method": "ECAPA-TDNN",
            "n_seeds": 1,
            "seeds": [42],
            "f1_weighted_mean": ecapa_meta["test_f1_weighted"],
            "f1_weighted_std": 0.0,
            "f1_macro_mean": ecapa_meta["test_f1_macro"],
            "f1_macro_std": 0.0,
            "accuracy_mean": ecapa_meta["test_accuracy"],
            "accuracy_std": 0.0,
            "latency": {
                "feature_extraction_ms_per_sample": None,
                "classifier_ms_per_sample": None,
                "total_e2e_ms_per_sample": None,
                "note": "GPU-bound; requires mel-spectrogram extraction + neural forward pass",
            },
            "representative_run": {
                "seed": 42,
                "f1_weighted": ecapa_meta["test_f1_weighted"],
                "f1_macro": ecapa_meta["test_f1_macro"],
                "accuracy": ecapa_meta["test_accuracy"],
                "classification_report": ecapa_meta["classification_report"],
                "confusion_matrix": ecapa_meta["confusion_matrix"],
            },
            "note": "Single-run result from ECAPA/emotion_model/metadata.json",
        }
        results.append(ecapa_entry)
        print(f"\n  ECAPA-TDNN (from metadata): wF1={ecapa_meta['test_f1_weighted']:.4f}")
    else:
        print("\n  ⚠ ECAPA metadata not found — skipping")

    if dfat_meta_path.exists():
        with open(dfat_meta_path, "r") as f:
            dfat_meta = json.load(f)
        dfat_entry = {
            "method": "DFAT Late Fusion",
            "n_seeds": 1,
            "seeds": [42],
            "f1_weighted_mean": dfat_meta["test_f1_weighted"],
            "f1_weighted_std": 0.0,
            "f1_macro_mean": dfat_meta["test_f1_macro"],
            "f1_macro_std": 0.0,
            "accuracy_mean": dfat_meta["test_accuracy"],
            "accuracy_std": 0.0,
            "latency": {
                "feature_extraction_ms_per_sample": None,
                "classifier_ms_per_sample": None,
                "total_e2e_ms_per_sample": None,
                "note": "GPU-bound; requires WavLM + Whisper ASR + PhoBERT extraction",
            },
            "representative_run": {
                "seed": 42,
                "f1_weighted": dfat_meta["test_f1_weighted"],
                "f1_macro": dfat_meta["test_f1_macro"],
                "accuracy": dfat_meta["test_accuracy"],
                "classification_report": dfat_meta["classification_report"],
                "confusion_matrix": dfat_meta["confusion_matrix"],
            },
            "note": "Single-run result from DFAT_Hybrid_Fusion/dualstream_model/metadata.json",
        }
        results.append(dfat_entry)
        print(f"  DFAT Late Fusion (from metadata): wF1={dfat_meta['test_f1_weighted']:.4f}")
    else:
        print("  ⚠ DFAT metadata not found — skipping")

    # ------------------------------------------------------------------
    # 6. Rank and save
    # ------------------------------------------------------------------
    ranked = sorted(results, key=lambda x: x["f1_weighted_mean"], reverse=True)

    summary = {
        "protocol": "Leak-free benchmark on full ViSEC, 5-seed repeated evaluation",
        "split": "80% Train+Val / 10% Test (from split_manifest.json)",
        "samples_total": int(len(df)),
        "trainval_size": int(len(trainval_idx)),
        "test_size": int(len(test_idx)),
        "seeds": SEEDS,
        "device": str(device),
        "emotion_labels": label_names,
        "primary_models": ["MFCC+RandomForest", "ECAPA-TDNN", "DFAT Late Fusion"],
        "secondary_models": ["MFCC+SVM", "MFCC+XGBoost", "WavLM+LogReg", "WavLM+SVM"],
        "ranked_results": ranked,
        "best_method": ranked[0]["method"],
    }

    output_path = Path(__file__).parent / "benchmark_results_gpu.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (ranked by Mean Weighted F1)")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        std_str = f"±{r['f1_weighted_std']:.4f}" if r['f1_weighted_std'] > 0 else "(single)"
        print(
            f"  {i}. {r['method']:22s}  "
            f"wF1={r['f1_weighted_mean']:.4f} {std_str:>12s}  "
            f"mF1={r['f1_macro_mean']:.4f}  "
            f"Acc={r['accuracy_mean']:.4f}"
        )

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
