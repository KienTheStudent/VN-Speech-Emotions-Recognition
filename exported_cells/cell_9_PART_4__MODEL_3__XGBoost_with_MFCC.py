# ==================== PART 4: MODEL 3 - XGBoost with MFCC ====================
print("🤖 MODEL 3: XGBoost with MFCC Features")

from xgboost import XGBClassifier

if MODE == "demo":
    print("✨ DEMO MODE: Loading pre-computed results...")
    with open("benchmark_results_gpu.json", "r", encoding="utf-8") as f:
        bench_res = json.load(f)
    
    res = next(r for r in bench_res['ranked_results'] if r['method'] == 'MFCC+XGBoost')
    print(f"✓ F1 Score (Weighted): {res['f1_weighted']:.4f}")
    print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
else:
    print("🚀 RETRAIN MODE: Training model from scratch...")
    clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="mlogloss")
    res = evaluate_method("MFCC+XGBoost", clf, mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test, emotion_labels)
    print(f"✓ F1 Score (Weighted): {res['f1_weighted']:.4f}")
    print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
