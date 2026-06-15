# ==================== PART 3: MODEL 2 - Random Forest with MFCC ====================
print("🤖 MODEL 2: Random Forest with MFCC Features")

from sklearn.ensemble import RandomForestClassifier

if MODE == "demo":
    print("✨ DEMO MODE: Loading pre-computed results...")
    with open("benchmark_results_gpu.json", "r", encoding="utf-8") as f:
        bench_res = json.load(f)
    
    res = next(r for r in bench_res['ranked_results'] if r['method'] == 'MFCC+RandomForest')
    print(f"✓ F1 Score (Weighted): {res['f1_weighted']:.4f}")
    print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
else:
    print("🚀 RETRAIN MODE: Training model from scratch...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    res = evaluate_method("MFCC+RandomForest", clf, mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test, emotion_labels)
    print(f"✓ F1 Score (Weighted): {res['f1_weighted']:.4f}")
    print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
