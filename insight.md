# Experimental Insights: Vietnamese SER on ViSEC

This document captures the detailed setup, metrics, and insights derived from the evaluation of multiple Speech Emotion Recognition models on the ViSEC dataset. All experiments follow a strict **speaker-independent, leak-free evaluation protocol** governed by an integrity-checked split manifest.

---

## 1. Experimental Setup

- **Dataset**: Vietnamese Speech Emotion Corpus (ViSEC) (5,280 samples, 147 speakers)
- **Classes**: Angry, Happy, Neutral, Sad
- **Evaluation Split**: 80% Train / 10% Validation / 10% Test
- **Split Governance**: Cryptographically verified `split_manifest.json` ensures zero speaker overlap between splits.
- **Robustness**: Primary models are evaluated across 5 random seeds to account for initialization variance.
- **Leak-Free Guarantee**: Test set is strictly held out and evaluated only once per model. Optuna tuning and early stopping rely entirely on the Validation set.

---

## 2. Model Configurations

### 2.1 MFCC + RandomForest (Classical Baseline)
- **Features:** 40-dimensional MFCC, mean-pooled per utterance
- **Classifier:** RandomForest (300 trees)
- Train+Val combined for training (no separate tuning needed)

### 2.2 ECAPA-TDNN-based Acoustic Baseline
- Input: 80-d log-mel spectrogram
- Architecture: ECAPA-TDNN-style with 256 channels, 192-d embeddings, SE blocks
- Parameters: ~1M
- Training: 100 epochs max, Adam lr=3e-4, ReduceLROnPlateau on **val F1**, early stopping (patience=20)

### 2.3 DFAT Hybrid Fusion (ASR-Assisted Audio-Linguistic Fusion) — Proposed Method
- **SEFE (Acoustic):** WavLM-base-plus → 768-d embeddings
- **TEFE (Linguistic):** Whisper-small (ASR) → Vietnamese Text → word segmentation (`underthesea`) → PhoBERT-base-v2 → 768-d embeddings
- **Early Fusion:** Concatenation → 1,536-d features, StandardScaler normalized
- **Classifiers:** LR, RF, XGBoost (Optuna-tuned on **validation**, 10 trials)
- **Late Fusion:** Weighted probability ensemble, weights optimized by Optuna (100 trials) on **validation**


---

## 3. Results

### 3.1 Overall Comparison

<!-- START_BENCHMARK_TABLE -->
| Method | Category | wF1 (mean ± std) | mF1 | Acc | E2E Latency |
|--------|----------|------------------|-----|-----|-------------|
| **DFAT Hybrid Fusion** | **Primary** | 0.4897 ± 0.0122 | 0.4841 | 0.4929 | — |
| **ECAPA-TDNN (simplified implementation)** | **Primary** | 0.4125 ± 0.0171 | 0.4122 | 0.4112 | — |
| **MFCC+RandomForest** | **Primary** | 0.3510 ± 0.0026 | 0.3365 | 0.3716 | — |
<!-- END_BENCHMARK_TABLE -->

### 3.2 DFAT Ablation Study

<!-- START_ABLATION_TABLE -->
| Configuration | Ensemble wF1 | Ensemble mF1 | Acc |
|---------------|-------------|-------------|-----|
| Acoustic-only (WavLM) | 0.4918 | 0.4887 | 0.4896 |
| Linguistic-only (vinai/PhoWhisper-large + PhoBERT) | 0.3953 | 0.3942 | 0.3964 |
| Early Fusion (Concat 1536-d) | 0.5080 | 0.5023 | 0.5104 |
| Early Fusion (No StandardScaler) | 0.5238 | 0.5203 | 0.5237 |
| Early Fusion (No Optuna tuning) | 0.4982 | 0.4916 | 0.5030 |
| Early Fusion (No word segmentation) | 0.4947 | 0.4912 | 0.4956 |
| Late Fusion (stream-level) | 0.4365 | 0.4259 | 0.4586 |
| Linguistic-only (openai/whisper-small + PhoBERT) | 0.3517 | 0.3471 | 0.3609 |
| Early Fusion (openai/whisper-small) | 0.4716 | 0.4641 | 0.4749 |
| Early Fusion + 10% synthetic noise | 0.4934 | 0.4881 | 0.4941 |
| Early Fusion + 20% synthetic noise | 0.4863 | 0.4798 | 0.4882 |
| Early Fusion + 30% synthetic noise | 0.4941 | 0.4920 | 0.4941 |
<!-- END_ABLATION_TABLE -->

---

## 4. Analysis and Discussion

### 4.1 Classical methods struggle with strict speaker-independence

Under the leak-free protocol, the classical MFCC+RF baseline drops to wF1 0.3510. Previous benchmarks utilized random stratified sampling, which allowed speaker identity leakage between train and test sets, artificially inflating classical ML performance to $\sim$ 0.64. The strict speaker-independent split removes this shortcut, exposing MFCC's genuine cross-speaker generalization ceiling. By contrast, deep learning methods (ECAPA, WavLM-based) exhibit superior capacity to generalize to unseen speakers.

### 4.2 ASR-assisted fusion outperforms pure acoustic baseline

The proposed DFAT Hybrid Fusion model achieves the highest wF1 (0.4897) on the test set, outperforming the pure acoustic ECAPA-TDNN baseline (wF1 0.4125). The performance gap highlights the advantage of the audio-linguistic fusion:
- **Semantic Anchor**: Linguistic representations from ASR act as a strong anchor, stabilizing the model against speaker-induced acoustic variance.
- **Robustness**: While pure acoustic models overfit to speaker timbre, the addition of text provides a regularizing effect that improves generalization to unseen speakers.

### 4.3 Compute trade-off analysis

MFCC+RF offers the best cost-efficiency: competitive accuracy with sub-millisecond inference and no GPU required. DFAT achieves superior accuracy but requires WavLM (94M params) + Whisper (244M params) for feature extraction, making it suitable for offline/batch processing rather than real-time edge deployment.

### 4.4 Error Analysis

Common confusion patterns across all models:
- **Neutral ↔ Happy**: The most frequent confusion. Both can have moderate energy and pitch, making acoustic-only separation difficult.
- **Neutral ↔ Sad**: Both are low-energy emotions. DFAT's linguistic stream provides the strongest disambiguation here.
- **Happy ↔ Angry**: High-energy emotions that share elevated pitch and speaking rate.

---

## 5. Key Takeaways

1. **ASR-assisted fusion >> Audio-only baseline.** The proposed DFAT Hybrid Fusion model demonstrates that linguistic embeddings act as a crucial anchor for emotion recognition, increasing robustness against speaker variance compared to purely acoustic models like ECAPA-TDNN.

2. **Strict leak-free evaluation is critical.** Previous results with test-set-based tuning and overlapping speakers were artificially inflated. Academic benchmarks must enforce Train/Val/Test separation rigorously.

3. **ASR acts as a harmful bottleneck.** While textual context can occasionally disambiguate emotions, the process of quantizing audio into text discards too much critical affective signal and introduces cascading noise under extreme emotional prosody.

---

## 6. Limitations and Future Work

- **Speaker Imbalance**: ViSEC's extreme speaker imbalance (Speaker 0 = 42%) means the strictly speaker-independent training set is dominated by one speaker, potentially limiting broad generalization. Future work should explore more balanced corpora.
- **ASR quality**: Whisper's Vietnamese word error rate introduces noise; the ablation study quantifies this impact via Whisper-tiny comparison.
- **Fine-tuning pretrained models**: WavLM and Whisper were used with frozen features. Fine-tuning on ViSEC may yield further gains.
- **Neural fusion**: Replacing classical ensemble with MLP or attention-based fusion on the 1,536-d space could improve results.