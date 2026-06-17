import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def evaluate_single_seed(clf_factory, x_train, y_train, x_test, y_test, label_names, seed):
    clf = clf_factory(seed)
    clf.fit(x_train, y_train)

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

def summarize_runs(name, runs, is_stochastic, feat_extract_ms):
    wf1s = [r["f1_weighted"] for r in runs]
    mf1s = [r["f1_macro"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    clf_lats = [r["classifier_ms_per_sample"] for r in runs]
    
    # Calculate median index for representative run
    median_idx = int(np.argsort(wf1s)[len(wf1s) // 2])
    representative = runs[median_idx]
    
    # Create latency standard dict
    avg_clf_lat = np.mean(clf_lats)
    latency_dict = {
        "feature_extraction_ms_per_sample": round(feat_extract_ms, 4),
        "classifier_ms_per_sample": round(avg_clf_lat, 4),
        "end_to_end_ms_per_sample": round(feat_extract_ms + avg_clf_lat, 4),
        "latency_note": "End-to-End latency is comparable across baselines. Feature-extraction is CPU-bound."
    }

    return {
        "method": name,
        "is_stochastic": is_stochastic,
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "f1_weighted_mean": float(np.mean(wf1s)),
        "f1_weighted_std": float(np.std(wf1s)) if is_stochastic else 0.0,
        "f1_macro_mean": float(np.mean(mf1s)),
        "f1_macro_std": float(np.std(mf1s)) if is_stochastic else 0.0,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)) if is_stochastic else 0.0,
        "latency": latency_dict,
        "representative_run": representative
    }
