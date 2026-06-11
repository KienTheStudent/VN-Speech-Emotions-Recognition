# ==================== PART 5: MODEL 4 - WavLM Baseline ====================
print("🤖 MODEL 4: WavLM Feature Extractor + SVM / LogReg")

from transformers import AutoFeatureExtractor, AutoModel
import torch
from benchmark_methods_gpu import wavlm_feature
from sklearn.linear_model import LogisticRegression

if MODE == "demo":
    print("✨ DEMO MODE: Loading pre-computed results...")
    with open("benchmark_results_gpu.json", "r", encoding="utf-8") as f:
        bench_res = json.load(f)
    
    res = next((r for r in bench_res['ranked_results'] if r['method'] == 'WavLM+SVM'), None)
    if res:
        print(f"✓ WavLM+SVM F1 Score (Weighted): {res['f1_weighted']:.4f}")
        print(f"✓ Latency: {res['latency_ms_per_sample']:.2f} ms/sample")
else:
    print("🚀 RETRAIN MODE: Training model from scratch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting WavLM embeddings on {device}...")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    wavlm_train, wavlm_y_train = [], []
    for p, y in zip(X_trainval, y_trainval):
        audio = load_audio(p)
        if audio is not None:
            wavlm_train.append(wavlm_feature(audio, extractor, wavlm, device))
            wavlm_y_train.append(y)
            
    wavlm_test, wavlm_y_test = [], []
    for p, y in zip(X_test, y_test):
        audio = load_audio(p)
        if audio is not None:
            wavlm_test.append(wavlm_feature(audio, extractor, wavlm, device))
            wavlm_y_test.append(y)
            
    scaler = StandardScaler()
    wavlm_train = scaler.fit_transform(wavlm_train)
    wavlm_test = scaler.transform(wavlm_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    res = evaluate_method("WavLM+LogReg", clf, wavlm_train, wavlm_y_train, wavlm_test, wavlm_y_test, emotion_labels)
    print(f"✓ WavLM+LogReg F1 Score (Weighted): {res['f1_weighted']:.4f}")
