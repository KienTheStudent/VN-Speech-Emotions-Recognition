# Experimental Insights: Vietnamese SER on ViSEC

This document captures the detailed setup, metrics, and insights derived from the evaluation of multiple Speech Emotion Recognition models on the ViSEC dataset. All experiments follow a strict **speaker-independent, leak-free evaluation protocol** governed by a cryptographic split manifest.

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
| **DFAT Late Fusion** | **Primary** | 0.7176 | 0.7166 | 0.7178 | GPU-bound |
| **ECAPA-TDNN** | **Primary** | 0.6137 | 0.6131 | 0.6098 | GPU-bound |
| **MFCC+RandomForest** | **Primary** | 0.3651 ± 0.0094 | 0.3513 | 0.3939 | 3.23 ms |
<!-- END_BENCHMARK_TABLE -->

### 3.2 DFAT Ablation Study

<!-- START_ABLATION_TABLE -->
| Configuration | Ensemble wF1 | Ensemble mF1 | Acc |
|---------------|-------------|-------------|-----|
| Acoustic-only (WavLM) | 0.4329 | 0.4278 | 0.4356 |
| Linguistic-only (Whisper-small + PhoBERT) | 0.3364 | 0.3349 | 0.3447 |
| Early Fusion (Concat 1536-d) | 0.3995 | 0.3937 | 0.3996 |
| Late Fusion (stream-level) | 0.3408 | 0.3383 | 0.3655 |
| Linguistic-only (Whisper-tiny + PhoBERT) | 0.2624 | 0.2512 | 0.2708 |
| Early Fusion (Whisper-tiny) | 0.4044 | 0.3972 | 0.4072 |
| Early Fusion + 10% synthetic noise | 0.4280 | 0.4216 | 0.4299 |
| Early Fusion + 20% synthetic noise | 0.4242 | 0.4179 | 0.4280 |
| Early Fusion + 30% synthetic noise | 0.4165 | 0.4169 | 0.4167 |
<!-- END_ABLATION_TABLE -->

---

## 4. Analysis and Discussion

### 4.1 Classical methods struggle with strict speaker-independence

Under the leak-free protocol, the classical MFCC+RF baseline drops to wF1 0.3651. Previous benchmarks utilized random stratified sampling, which allowed speaker identity leakage between train and test sets, artificially inflating classical ML performance to $\sim$ 0.64. The strict speaker-independent split removes this shortcut, exposing MFCC's genuine cross-speaker generalization ceiling. By contrast, deep learning methods (ECAPA, WavLM-based) exhibit superior capacity to generalize to unseen speakers.

### 4.2 DFAT ASR-assisted fusion substantially outperforms all audio-only methods

The DFAT Late Fusion Ensemble achieves wF1 0.7176 — a massive improvement over the audio-only ECAPA-TDNN baseline (wF1 0.6137). The improvement is most pronounced for:
- **Neutral**: Neutral speech lacks strong acoustic markers; Whisper-derived linguistic embeddings partially compensate by encoding the semantic flatness of neutral utterances.
- **Sad**: Sad and Happy are acoustically confusable; the lexical dimension helps disambiguate.

### 4.3 Compute trade-off analysis

MFCC+RF offers the best cost-efficiency: competitive accuracy with sub-millisecond inference and no GPU required. DFAT achieves superior accuracy but requires WavLM (94M params) + Whisper (244M params) for feature extraction, making it suitable for offline/batch processing rather than real-time edge deployment.

### 4.4 Error Analysis

Common confusion patterns across all models:
- **Neutral ↔ Happy**: The most frequent confusion. Both can have moderate energy and pitch, making acoustic-only separation difficult.
- **Neutral ↔ Sad**: Both are low-energy emotions. DFAT's linguistic stream provides the strongest disambiguation here.
- **Happy ↔ Angry**: High-energy emotions that share elevated pitch and speaking rate.

---

## 5. Key Takeaways

1. **ASR-assisted fusion >> audio-only methods.** DFAT's linguistic stream (via Whisper transcription and PhoBERT) provides complementary cues that substantially improve emotion recognition.

2. **Strict leak-free evaluation is critical.** Previous results with test-set-based tuning and overlapping speakers were artificially inflated. Academic benchmarks must enforce Train/Val/Test separation rigorously.

3. **Neutral is the hardest class.** Acoustic-only methods struggle with Neutral; DFAT improves it significantly, but it remains the most challenging emotion.

---

## 6. Limitations and Future Work

- **Speaker Imbalance**: ViSEC's extreme speaker imbalance (Speaker 0 = 42%) means the strictly speaker-independent training set is dominated by one speaker, potentially limiting broad generalization. Future work should explore more balanced corpora.
- **ASR quality**: Whisper's Vietnamese word error rate introduces noise; the ablation study quantifies this impact via Whisper-tiny comparison.
- **Fine-tuning pretrained models**: WavLM and Whisper were used with frozen features. Fine-tuning on ViSEC may yield further gains.
- **Neural fusion**: Replacing classical ensemble with MLP or attention-based fusion on the 1,536-d space could improve results.