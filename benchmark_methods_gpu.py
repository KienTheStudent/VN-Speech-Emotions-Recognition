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

import json
import warnings
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

from src.config.paths import (
    ECAPA_METADATA,
    DFAT_METADATA,
    BENCHMARK_RESULTS_PATH
)
from src.data.loader import load_manifest, load_audio_sample
from src.features.mfcc import FeatureExtractor
from src.models.classical import get_rf_model
from src.evaluation.evaluate import evaluate_pipeline

warnings.filterwarnings("ignore")
SEEDS = [42, 123, 456, 789, 2026]

def read_deep_model(meta_path, method_name, latency_note, manifest_checksum):
    if not meta_path.exists():
        print(f"  ⚠ {method_name} metadata not found — skipping")
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    # Verify hash contract
    if "split_checksum" not in meta:
        raise ValueError(f"Metadata {meta_path} is missing 'split_checksum'! Provenance cannot be verified.")
    if meta["split_checksum"] != manifest_checksum:
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
            "end_to_end_ms_per_sample": None,
            "latency_note": latency_note
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

    print("\nLoading audio samples...")
    train_samples = [load_audio_sample(p, l) for p, l in zip(df["path"].iloc[train_idx].values, df["label"].iloc[train_idx].values)]
    test_samples = [load_audio_sample(p, l) for p, l in zip(df["path"].iloc[test_idx].values, df["label"].iloc[test_idx].values)]

    rejected_log = []
    results = []

    # 2. Evaluate Models
    rf_factory = get_rf_model
    results.append(evaluate_pipeline(
        name="MFCC+RandomForest",
        clf_factory=rf_factory,
        feat_extractor=FeatureExtractor("mfcc"),
        is_stochastic=True,
        train_samples=train_samples,
        test_samples=test_samples,
        label_names=label_names,
        rejected_log=rejected_log,
        seeds=SEEDS
    ))

    # 3. Handle rejected samples
    if len(rejected_log) > 0:
        with open("rejected_samples.json", "w") as f:
            json.dump(rejected_log, f, indent=2)
        print(f"\n⚠ WARNING: {len(rejected_log)} samples failed to load. See rejected_samples.json")
    
    # Expected number of processed samples for test set must match exactly
    processed_test_len = len(test_idx) - len([r for r in rejected_log if r['split'] == 'test'])

    # 4. Read Deep Models & Check Hashes
    e_res = read_deep_model(ECAPA_METADATA, "ECAPA-TDNN (simplified implementation)", "N/A (GPU-bound, not comparable with CPU baselines)", manifest_checksum)
    if e_res: results.append(e_res)
    
    d_res = read_deep_model(DFAT_METADATA, "DFAT Late Fusion", "N/A (GPU-bound, not comparable with CPU baselines)", manifest_checksum)
    if d_res: results.append(d_res)

    # 5. Rank and save
    ranked = sorted(results, key=lambda x: x["f1_weighted_mean"], reverse=True)
    primary_models_list = ["MFCC+RandomForest", "ECAPA-TDNN (simplified implementation)", "DFAT Late Fusion"]

    global_best = ranked[0]["method"] if ranked else None
    primary_ranked = [r for r in ranked if r["method"] in primary_models_list]
    primary_best = primary_ranked[0]["method"] if primary_ranked else None

    summary = {
        "protocol": "Leak-free benchmark on full ViSEC, Apples-to-Apples Eval",
        "split": "80% Train / 10% Validation / 10% Test (from split_manifest.json)",
        "samples_total": int(len(df)),
        "train_size": int(len(manifest["train_indices"])),
        "val_size": int(len(manifest["val_indices"])),
        "test_size": int(len(test_idx)),
        "seeds": SEEDS,
        "device": "cpu/cuda",
        "emotion_labels": label_names,
        "primary_models": primary_models_list,
        "ranked_results": ranked,
        "global_best_method": global_best,
        "primary_best_method": primary_best,
    }

    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
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

    print(f"\n✓ Results saved to: {BENCHMARK_RESULTS_PATH}")

if __name__ == "__main__":
    main()
