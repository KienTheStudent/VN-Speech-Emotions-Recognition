#!/usr/bin/env python3
"""Unified SER benchmark — Leak-Free Protocol, Apples-to-Apples Evaluation.

Evaluates:
  PRIMARY BASELINES (in main thesis body):
    1. MFCC + RandomForest
    2. ECAPA-TDNN           (results read from ECAPA/emotion_model/metadata.json)
    3. DFAT Late Fusion     (results read from DFAT_Hybrid_Fusion/dualstream_model/metadata.json)



Methodology fixes:
  - Deep methods are run ONCE.
  - Stochastic classical methods (RF) are run 5 times with seeds [42, 123, 456, 789, 2026].
  - Apples-to-apples latency measures End-to-End time per sample.
  - Silent sample loss during audio loading is tracked and asserted.
  - Split manifest checksum is verified.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from split_validator import validate_manifest

warnings.filterwarnings("ignore")

MANIFEST_PATH = Path(__file__).parent / "split_manifest.json"
SEEDS = [42, 123, 456, 789, 2026]


# ==================== UTILITIES ====================

def load_manifest():
    """Load the fixed split manifest and validate."""
    checksum = validate_manifest()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest loaded: {manifest['total_samples']} total samples")
    print(f"  Train: {len(manifest['train_indices'])}")
    print(f"  Val:   {len(manifest['val_indices'])}")
    print(f"  Test:  {len(manifest['test_indices'])}")
    return manifest, checksum


def load_audio(path_dict, sr=16000):
    try:
        if isinstance(path_dict, dict) and "bytes" in path_dict:
            audio, _ = librosa.load(io.BytesIO(path_dict["bytes"]), sr=sr)
            return audio, None
        if isinstance(path_dict, dict) and "path" in path_dict:
            audio, _ = librosa.load(path_dict["path"], sr=sr)
            return audio, None
        if isinstance(path_dict, str):
            audio, _ = librosa.load(path_dict, sr=sr)
            return audio, None
    except Exception as e:
        return None, str(e)
    return None, "Unknown path format"


class FeatureExtractor:
    def __init__(self, method):
        self.method = method
        
    def extract(self, audio):
        if self.method == "mfcc":
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            return np.mean(mfcc, axis=1)


def evaluate_single_seed(clf_factory, x_train, y_train, x_test, y_test, label_names, seed):
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
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
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


def evaluate_pipeline(name, clf_factory, feat_type, is_stochastic,
                      train_paths, train_labels, test_paths, test_labels,
                      label_names, rejected_log):
    print(f"\n{'='*50}")
    print(f"  {name} — {'Stochastic (5 seeds)' if is_stochastic else 'Deterministic (1 seed)'}")
    print(f"{'='*50}")

    extractor = FeatureExtractor(feat_type)
    
    # Precompute training features
    print("  Extracting training features...")
    x_train, y_train_clean = [], []
    for p, y in zip(train_paths, train_labels):
        audio, err = load_audio(p)
        if audio is None:
            rejected_log.append({"split": "train", "path": str(p), "error": err})
            continue
        feat = extractor.extract(audio)
        x_train.append(feat)
        y_train_clean.append(y)
        
    # Precompute testing features & measure apples-to-apples latency
    print("  Extracting testing features & measuring latency...")
    x_test, y_test_clean = [], []
    extract_times = []
    
    for p, y in zip(test_paths, test_labels):
        t0 = time.perf_counter()
        audio, err = load_audio(p)
        if audio is None:
            rejected_log.append({"split": "test", "path": str(p), "error": err})
            continue
        feat = extractor.extract(audio)
        extract_times.append(time.perf_counter() - t0)
        x_test.append(feat)
        y_test_clean.append(y)
        
    feat_extract_ms = np.median(extract_times) * 1000

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train_clean = np.array(y_train_clean)
    y_test_clean = np.array(y_test_clean)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Evaluate seeds
    active_seeds = SEEDS if is_stochastic else [42]
    runs = []
    for seed in active_seeds:
        result = evaluate_single_seed(clf_factory, x_train, y_train_clean, x_test, y_test_clean, label_names, seed)
        runs.append(result)
        print(f"  seed={seed}: wF1={result['f1_weighted']:.4f}  "
              f"mF1={result['f1_macro']:.4f}  Acc={result['accuracy']:.4f}")

    wf1s = [r["f1_weighted"] for r in runs]
    mf1s = [r["f1_macro"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    clf_lats = [r["classifier_ms_per_sample"] for r in runs]

    median_idx = int(np.argsort(wf1s)[len(wf1s) // 2])
    representative = runs[median_idx]

    summary = {
        "method": name,
        "n_seeds": len(active_seeds),
        "seeds": active_seeds,
        "f1_weighted_mean": float(np.mean(wf1s)),
        "f1_weighted_std": float(np.std(wf1s)) if is_stochastic else 0.0,
        "f1_macro_mean": float(np.mean(mf1s)),
        "f1_macro_std": float(np.std(mf1s)) if is_stochastic else 0.0,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)) if is_stochastic else 0.0,
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

def make_rf_factory(n_estimators=300):
    def factory(seed):
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    return factory, True


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("BENCHMARK: Full ViSEC — Leak-Free, Apples-to-Apples Eval")
    print("=" * 60)

    # 1. Load manifest & dataset
    manifest, manifest_checksum = load_manifest()
    print("\nLoading ViSEC dataset...")
    ds = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
    df = ds.to_pandas()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    label_names = le.classes_.tolist()

    train_idx = manifest["train_indices"]
    test_idx = manifest["test_indices"]

    train_paths = df["path"].iloc[train_idx].values
    train_labels = df["label"].iloc[train_idx].values
    test_paths = df["path"].iloc[test_idx].values
    test_labels = df["label"].iloc[test_idx].values

    rejected_log = []
    results = []

    # 2. Evaluate Models
    rf_factory, rf_stoch = make_rf_factory(300)
    results.append(evaluate_pipeline(
        "MFCC+RandomForest", rf_factory, "mfcc", rf_stoch,
        train_paths, train_labels, test_paths, test_labels, label_names, rejected_log
    ))

    # 3. Handle rejected samples
    if len(rejected_log) > 0:
        with open("rejected_samples.json", "w") as f:
            json.dump(rejected_log, f, indent=2)
        print(f"\n⚠ WARNING: {len(rejected_log)} samples failed to load. See rejected_samples.json")
    
    # Expected number of processed samples for test set must match exactly
    processed_test_len = len(test_idx) - len([r for r in rejected_log if r['split'] == 'test'])
    assert processed_test_len == len(test_idx), f"Test set sample loss! Expected {len(test_idx)}, got {processed_test_len}"

    # 4. Read Deep Models & Check Hashes
    ecapa_meta_path = Path(__file__).parent / "ECAPA" / "emotion_model" / "metadata.json"
    dfat_meta_path = Path(__file__).parent / "DFAT_Hybrid_Fusion" / "dualstream_model" / "metadata.json"

    def read_deep_model(meta_path, method_name, latency_note):
        if not meta_path.exists():
            print(f"  ⚠ {method_name} metadata not found — skipping")
            return None
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        # Verify hash contract
        if "split_checksum" in meta and meta["split_checksum"] != manifest_checksum:
            raise ValueError(f"Metadata {meta_path} checksum mismatch! The model was trained on an older split. Please retrain.")
            
        entry = {
            "method": method_name,
            "n_seeds": meta.get("n_seeds", 1),
            "seeds": meta.get("seeds", [42]),
            "f1_weighted_mean": meta.get("test_f1_weighted_mean", meta.get("test_f1_weighted")),
            "f1_weighted_std": meta.get("test_f1_weighted_std", 0.0),
            "f1_macro_mean": meta.get("test_f1_macro_mean", meta.get("test_f1_macro")),
            "f1_macro_std": meta.get("test_f1_macro_std", 0.0),
            "accuracy_mean": meta.get("test_accuracy_mean", meta.get("test_accuracy")),
            "accuracy_std": meta.get("test_accuracy_std", 0.0),
            "latency": {
                "feature_extraction_ms_per_sample": None,
                "classifier_ms_per_sample": None,
                "total_e2e_ms_per_sample": None,
                "note": latency_note,
            },
            "representative_run": meta.get("representative_run", {
                "seed": 42,
                "f1_weighted": meta.get("test_f1_weighted"),
                "f1_macro": meta.get("test_f1_macro"),
                "accuracy": meta.get("test_accuracy"),
                "classification_report": meta.get("classification_report"),
                "confusion_matrix": meta.get("confusion_matrix"),
            }),
            "note": f"Results from {meta_path.name}",
        }
        print(f"  {method_name} (from metadata): wF1={entry['f1_weighted_mean']:.4f} ± {entry['f1_weighted_std']:.4f}")
        return entry

    e_res = read_deep_model(ecapa_meta_path, "ECAPA-TDNN (simplified implementation)", "GPU-bound; mel-spec + neural fwd")
    if e_res: results.append(e_res)
    
    d_res = read_deep_model(dfat_meta_path, "DFAT Late Fusion", "GPU-bound; WavLM + Whisper + PhoBERT")
    if d_res: results.append(d_res)

    # 5. Rank and save
    ranked = sorted(results, key=lambda x: x["f1_weighted_mean"], reverse=True)

    primary_models_list = ["MFCC+RandomForest", "ECAPA-TDNN (simplified implementation)", "DFAT Late Fusion"]

    global_best = ranked[0]["method"] if ranked else None
    primary_ranked = [r for r in ranked if r["method"] in primary_models_list]
    primary_best = primary_ranked[0]["method"] if primary_ranked else None

    summary = {
        "protocol": "Leak-free benchmark on full ViSEC, Apples-to-Apples Eval",
        "split": "80% Train / 10% Test (from split_manifest.json)",
        "samples_total": int(len(df)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "seeds": SEEDS,
        "device": "cpu/cuda",
        "emotion_labels": label_names,
        "primary_models": primary_models_list,
        "ranked_results": ranked,
        "global_best_method": global_best,
        "primary_best_method": primary_best,
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
            f"  {i}. {r['method']:40s}  "
            f"wF1={r['f1_weighted_mean']:.4f} {std_str:>12s}  "
            f"mF1={r['f1_macro_mean']:.4f}  "
            f"Acc={r['accuracy_mean']:.4f}"
        )

    print(f"\n✓ Results saved to: {output_path}")

if __name__ == "__main__":
    main()
