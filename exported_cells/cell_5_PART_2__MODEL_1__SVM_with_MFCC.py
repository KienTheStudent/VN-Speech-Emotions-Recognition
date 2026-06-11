# ==================== PART 2: MODEL 1 - SVM with MFCC ====================
print("🤖 MODEL 1: SVM with MFCC Features")

from benchmark_methods_gpu import load_audio, mfcc_feature, evaluate_method
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import time

if MODE == "demo":
    print("✨ DEMO MODE: Loading pre-computed results...")
    with open("benchmark_results_gpu.json", "r", encoding="utf-8") as f:
        bench_res = json.load(f)
    
    res = next(r for r in bench_res['ranked_results'] if r['method'] == 'MFCC+SVM')
    print(f"✓ F1 Score (Weighted): {res['f1_weighted']:.4f}")
    print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
    
    cm = np.array(res['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title("SVM Confusion Matrix (Test Set)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
else:
    print("🚀 RETRAIN MODE: Training model from scratch...")
    mfcc_train, mfcc_y_train = [], []
    for p, y in zip(X_trainval, y_trainval):
        audio = load_audio(p)
        if audio is not None:
            mfcc_train.append(mfcc_feature(audio))
            mfcc_y_train.append(y)
            
    mfcc_test, mfcc_y_test = [], []
    for p, y in zip(X_test, y_test):
        audio = load_audio(p)
        if audio is not None:
            mfcc_test.append(mfcc_feature(audio))
            mfcc_y_test.append(y)
            
    scaler = StandardScaler()
    mfcc_train = scaler.fit_transform(mfcc_train)
    mfcc_test = scaler.transform(mfcc_test)
    
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    res = evaluate_method("MFCC+SVM", clf, mfcc_train, mfcc_y_train, mfcc_test, mfcc_y_test, emotion_labels)
    print(f"✓ F1 Score (Weighted): {res['f1_weighted']:.4f}")
    print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
