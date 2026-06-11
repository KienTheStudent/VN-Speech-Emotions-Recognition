#!/usr/bin/env python3
"""Run multiple SER methods on full ViSEC (5280 samples) and rank by F1.

Methods evaluated:
- MFCC + SVM
- MFCC + RandomForest
- MFCC + XGBoost
- WavLM embedding + LogisticRegression
- WavLM embedding + SVM

All methods read the fixed Train/Val/Test split from split_manifest.json
to ensure fair, leak-free comparison across the entire project.
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


def evaluate_method(name, clf, x_train, y_train, x_test, y_test, label_names):
    """Train a classifier, evaluate on test set, and measure inference latency."""
    clf.fit(x_train, y_train)

    # Measure inference latency (average per sample)
    start = time.perf_counter()
    y_pred = clf.predict(x_test)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / len(x_test)) * 1000  # ms per sample

    w_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    m_f1 = float(f1_score(y_test, y_pred, average="macro"))
    acc = float(accuracy_score(y_test, y_pred))

    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "method": name,
        "f1_weighted": w_f1,
        "f1_macro": m_f1,
        "accuracy": acc,
        "latency_ms_per_sample": round(latency_ms, 4),
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ==================== MAIN ====================


def main():
    print("=" * 60)
    print("BENCHMARK: Full ViSEC (5280 samples) — Leak-Free Protocol")
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

    # For classical ML benchmark we combine train+val for training
    # (no hyperparameter tuning with Optuna here, so val not needed separately)
    # and evaluate on the held-out test set only.
    trainval_idx = train_idx + val_idx

    trainval_paths = df["path"].iloc[trainval_idx].values
    trainval_labels = df["label"].iloc[trainval_idx].values
    test_paths = df["path"].iloc[test_idx].values
    test_labels = df["label"].iloc[test_idx].values

    print(f"\nTrain+Val: {len(trainval_idx)}, Test: {len(test_idx)}")
    print(f"Emotion labels: {label_names}")

    # ------------------------------------------------------------------
    # 2. Extract MFCC features
    # ------------------------------------------------------------------
    print("\nExtracting MFCC features...")
    mfcc_train, mfcc_y_train = [], []
    for i, (p, y) in enumerate(zip(trainval_paths, trainval_labels)):
        if i % 500 == 0:
            print(f"  MFCC train+val: {i}/{len(trainval_paths)}")
        audio = load_audio(p)
        if audio is None:
            print(f"  Warning: Skipping train+val sample {i} (load failed)")
            continue
        mfcc_train.append(mfcc_feature(audio))
        mfcc_y_train.append(y)

    mfcc_test, mfcc_y_test = [], []
    for i, (p, y) in enumerate(zip(test_paths, test_labels)):
        if i % 200 == 0:
            print(f"  MFCC test: {i}/{len(test_paths)}")
        audio = load_audio(p)
        if audio is None:
            print(f"  Warning: Skipping test sample {i} (load failed)")
            continue
        mfcc_test.append(mfcc_feature(audio))
        mfcc_y_test.append(y)

    mfcc_train = np.array(mfcc_train)
    mfcc_test = np.array(mfcc_test)
    mfcc_y_train = np.array(mfcc_y_train)
    mfcc_y_test = np.array(mfcc_y_test)

    mfcc_scaler = StandardScaler()
    mfcc_train = mfcc_scaler.fit_transform(mfcc_train)
    mfcc_test = mfcc_scaler.transform(mfcc_test)

    print(f"  MFCC features extracted: train={len(mfcc_train)}, test={len(mfcc_test)}")

    # ------------------------------------------------------------------
    # 3. Extract WavLM embeddings
    # ------------------------------------------------------------------
    print("\nExtracting WavLM embeddings on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    wavlm_train, wavlm_y_train = [], []
    for i, (p, y) in enumerate(zip(trainval_paths, trainval_labels)):
        if i % 500 == 0:
            print(f"  WavLM train+val: {i}/{len(trainval_paths)}")
        audio = load_audio(p)
        if audio is None:
            print(f"  Warning: Skipping train+val sample {i} (load failed)")
            continue
        wavlm_train.append(wavlm_feature(audio, extractor, wavlm, device))
        wavlm_y_train.append(y)

    wavlm_test, wavlm_y_test = [], []
    for i, (p, y) in enumerate(zip(test_paths, test_labels)):
        if i % 200 == 0:
            print(f"  WavLM test: {i}/{len(test_paths)}")
        audio = load_audio(p)
        if audio is None:
            print(f"  Warning: Skipping test sample {i} (load failed)")
            continue
        wavlm_test.append(wavlm_feature(audio, extractor, wavlm, device))
        wavlm_y_test.append(y)

    wavlm_train = np.array(wavlm_train)
    wavlm_test = np.array(wavlm_test)
    wavlm_y_train = np.array(wavlm_y_train)
    wavlm_y_test = np.array(wavlm_y_test)

    wavlm_scaler = StandardScaler()
    wavlm_train = wavlm_scaler.fit_transform(wavlm_train)
    wavlm_test = wavlm_scaler.transform(wavlm_test)

    print(
        f"  WavLM features extracted: train={len(wavlm_train)}, test={len(wavlm_test)}"
    )

    # Free GPU memory
    del wavlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Evaluate all methods
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING METHODS")
    print("=" * 60)
    results = []

    # MFCC + SVM
    print("\n[1/5] MFCC + SVM")
    results.append(
        evaluate_method(
            "MFCC+SVM",
            SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
            mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test, label_names,
        )
    )
    print(f"  Weighted F1: {results[-1]['f1_weighted']:.4f}  Macro F1: {results[-1]['f1_macro']:.4f}")

    # MFCC + RandomForest
    print("\n[2/5] MFCC + RandomForest")
    results.append(
        evaluate_method(
            "MFCC+RandomForest",
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test, label_names,
        )
    )
    print(f"  Weighted F1: {results[-1]['f1_weighted']:.4f}  Macro F1: {results[-1]['f1_macro']:.4f}")

    # MFCC + XGBoost
    print("\n[3/5] MFCC + XGBoost")
    results.append(
        evaluate_method(
            "MFCC+XGBoost",
            XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                eval_metric="mlogloss",
            ),
            mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test, label_names,
        )
    )
    print(f"  Weighted F1: {results[-1]['f1_weighted']:.4f}  Macro F1: {results[-1]['f1_macro']:.4f}")

    # WavLM + Logistic Regression
    print("\n[4/5] WavLM + LogReg")
    results.append(
        evaluate_method(
            "WavLM+LogReg",
            LogisticRegression(max_iter=1000, random_state=42),
            wavlm_train, wavlm_y_train, wavlm_test, wavlm_y_test, label_names,
        )
    )
    print(f"  Weighted F1: {results[-1]['f1_weighted']:.4f}  Macro F1: {results[-1]['f1_macro']:.4f}")

    # WavLM + SVM
    print("\n[5/5] WavLM + SVM")
    results.append(
        evaluate_method(
            "WavLM+SVM",
            SVC(kernel="rbf", C=5.0, gamma="scale", random_state=42),
            wavlm_train, wavlm_y_train, wavlm_test, wavlm_y_test, label_names,
        )
    )
    print(f"  Weighted F1: {results[-1]['f1_weighted']:.4f}  Macro F1: {results[-1]['f1_macro']:.4f}")

    # ------------------------------------------------------------------
    # 5. Rank and save
    # ------------------------------------------------------------------
    ranked = sorted(results, key=lambda x: x["f1_weighted"], reverse=True)
    summary = {
        "protocol": "Leak-free benchmark on full ViSEC (5280 samples)",
        "split": "80% Train+Val / 10% Test (from split_manifest.json)",
        "samples_total": int(len(df)),
        "trainval_size_nominal": int(len(trainval_idx)),
        "test_size_nominal": int(len(test_idx)),
        "mfcc_trainval_evaluated": int(len(mfcc_train)),
        "mfcc_test_evaluated": int(len(mfcc_test)),
        "wavlm_trainval_evaluated": int(len(wavlm_train)),
        "wavlm_test_evaluated": int(len(wavlm_test)),
        "device": str(device),
        "emotion_labels": label_names,
        "ranked_results": ranked,
        "best_method": ranked[0],
    }

    # Report if any samples were skipped
    if len(mfcc_test) < len(test_idx):
        print(f"\n⚠ MFCC: {len(test_idx) - len(mfcc_test)} test samples skipped (nominal={len(test_idx)}, evaluated={len(mfcc_test)})")
    if len(wavlm_test) < len(test_idx):
        print(f"⚠ WavLM: {len(test_idx) - len(wavlm_test)} test samples skipped (nominal={len(test_idx)}, evaluated={len(wavlm_test)})")


    output_path = Path(__file__).parent / "benchmark_results_gpu.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (ranked by Weighted F1)")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        print(
            f"  {i}. {r['method']:20s}  "
            f"wF1={r['f1_weighted']:.4f}  "
            f"mF1={r['f1_macro']:.4f}  "
            f"Acc={r['accuracy']:.4f}  "
            f"Latency={r['latency_ms_per_sample']:.2f}ms"
        )

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
