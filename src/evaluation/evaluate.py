import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import evaluate_single_seed, summarize_runs

def evaluate_pipeline(name, clf_factory, feat_extractor, is_stochastic,
                      train_samples, test_samples, label_names, rejected_log,
                      seeds=[42, 123, 456, 789, 2026]):
    print(f"\n{'='*50}")
    print(f"  {name} — {'Stochastic (5 seeds)' if is_stochastic else 'Deterministic (1 seed)'}")
    print(f"{'='*50}")

    print("  Extracting training features...")
    x_train, y_train_clean = [], []
    for sample in train_samples:
        if sample.status == "error":
            rejected_log.append({"split": "train", "path": sample.path, "error": sample.error_reason})
            continue
        feat = feat_extractor.extract(sample.audio)
        x_train.append(feat)
        y_train_clean.append(sample.label)
        
    print("  Extracting testing features & measuring latency...")
    x_test, y_test_clean = [], []
    extract_times = []
    
    for sample in test_samples:
        if sample.status == "error":
            rejected_log.append({"split": "test", "path": sample.path, "error": sample.error_reason})
            continue
            
        t0 = time.perf_counter()
        feat = feat_extractor.extract(sample.audio)
        extract_times.append(time.perf_counter() - t0)
        
        x_test.append(feat)
        y_test_clean.append(sample.label)
        
    feat_extract_ms = np.median(extract_times) * 1000

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train_clean = np.array(y_train_clean)
    y_test_clean = np.array(y_test_clean)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    active_seeds = seeds if is_stochastic else [42]
    runs = []
    for seed in active_seeds:
        result = evaluate_single_seed(clf_factory, x_train, y_train_clean, x_test, y_test_clean, label_names, seed)
        runs.append(result)
        print(f"  seed={seed}: wF1={result['f1_weighted']:.4f}  "
              f"mF1={result['f1_macro']:.4f}  Acc={result['accuracy']:.4f}")

    summary = summarize_runs(name, runs, is_stochastic, feat_extract_ms)
    print(f"  → Mean wF1: {summary['f1_weighted_mean']:.4f} ± {summary['f1_weighted_std']:.4f}")
    return summary
