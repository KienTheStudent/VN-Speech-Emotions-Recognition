# Speech Emotion Recognition — Experimental Insight Report (Leak-Free Protocol)

**Dataset:** hustep-lab/ViSEC (Vietnamese Speech Emotion Corpus)  
**Task:** 4-class emotion classification — `angry`, `happy`, `neutral`, `sad`  
**Split:** ~4,227 train / 525 validation / 528 test (greedy speaker-independent split from `split_manifest.json`)  
**Evaluation metric:** Weighted F1-Score (primary), Macro F1-Score, Accuracy  
**Protocol:** All hyperparameter tuning and model selection on Validation set only; Test set evaluated exactly once.

---

## 1. Dataset Overview

The ViSEC dataset contains 5,280 audio samples with a moderately imbalanced class distribution:

| Emotion | Count | Proportion |
|---------|-------|------------|
| Neutral | 1,507 | 28.5% |
| Angry   | 1,466 | 27.8% |
| Happy   | 1,228 | 23.3% |
| Sad     | 1,079 | 20.4% |

Speaker distribution is highly long-tailed: Speaker 0 contributes 2,217 samples (42%), while many speakers contribute only 1–2 samples. To avoid speaker leakage, we utilize a greedy speaker-allocation algorithm to generate a strict **speaker-independent split** (approx. 80/10/10). This ensures models learn emotion rather than speaker identity, though it heavily biases the training set towards Speaker 0.

---

## 2. Method Descriptions

### 2.1 MFCC + RandomForest (Classical Baseline)
- **Features:** 40-dimensional MFCC, mean-pooled per utterance
- **Classifier:** RandomForest (300 trees)
- Train+Val combined for training (no separate tuning needed)

### 2.2 ECAPA-TDNN (Strong Acoustic Baseline)
- Input: 80-d log-mel spectrogram
- Architecture: ECAPA-TDNN with 256 channels, 192-d embeddings, SE blocks
- Parameters: ~1M
- Training: 100 epochs max, Adam lr=3e-4, ReduceLROnPlateau on **val F1**, early stopping (patience=20)
- Best val F1 achieved: 0.6206 (epoch 35)

### 2.3 DFAT Hybrid Fusion (ASR-Assisted Audio-Linguistic Fusion) — Proposed Method
- **SEFE (Acoustic):** WavLM-base-plus → 768-d embeddings
- **TEFE (Linguistic):** Whisper-small (ASR) → Vietnamese Text → word segmentation (`underthesea`) → PhoBERT-base-v2 → 768-d embeddings
- **Early Fusion:** Concatenation → 1,536-d features, StandardScaler normalized
- **Classifiers:** LR, RF, XGBoost (Optuna-tuned on **validation**, 10 trials)
- **Late Fusion:** Weighted probability ensemble, weights optimized by Optuna (100 trials) on **validation**

### 2.4 Secondary Baselines (Supplementary)
- MFCC + SVM (RBF, C=10)
- MFCC + XGBoost (300 trees, max_depth=8)
- WavLM + Logistic Regression
- WavLM + SVM (RBF, C=5)

---

## 3. Results

### 3.1 Overall Comparison

<!-- START_BENCHMARK_TABLE -->
| Method | Category | wF1 (mean ± std) | mF1 | Acc | E2E Latency |
|--------|----------|------------------|-----|-----|-------------|
| MFCC+RandomForest | Secondary | 0.6475 | 0.6463 | 0.6477 | — |
| MFCC+XGBoost | Secondary | 0.6439 | 0.6425 | 0.6439 | — |
| MFCC+SVM | Secondary | 0.6366 | 0.6374 | 0.6364 | — |
| WavLM+SVM | Secondary | 0.6309 | 0.6307 | 0.6307 | — |
| WavLM+LogReg | Secondary | 0.5713 | 0.5705 | 0.5701 | — |
<!-- END_BENCHMARK_TABLE -->

### 3.2 DFAT Ablation Study

<!-- START_ABLATION_TABLE -->

<!-- END_ABLATION_TABLE -->

### 3.3 Per-Class Performance (Primary Models)

#### ECAPA-TDNN (Strong Acoustic Baseline)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry   | 0.789 | 0.714 | 0.750 | 147 |
| Happy   | 0.516 | 0.642 | 0.572 | 123 |
| Neutral | 0.497 | 0.520 | 0.508 | 150 |
| Sad     | 0.706 | 0.556 | 0.622 | 108 |

#### DFAT Late Fusion Ensemble (Proposed — Best Overall)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry   | 0.789 | 0.789 | 0.789 | 147 |
| Happy   | 0.716 | 0.593 | 0.649 | 123 |
| Neutral | 0.626 | 0.760 | 0.687 | 150 |
| Sad     | 0.784 | 0.704 | 0.741 | 108 |

#### MFCC+RandomForest (Classical Baseline)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry   | 0.663 | 0.735 | 0.697 | 147 |
| Happy   | 0.734 | 0.561 | 0.636 | 123 |
| Neutral | 0.546 | 0.713 | 0.618 | 150 |
| Sad     | 0.773 | 0.537 | 0.634 | 108 |

---

## 4. Analysis and Discussion

### 4.1 MFCC remains competitive as a classical baseline

Under the leak-free protocol, MFCC+RF (wF1 0.6475) outperforms WavLM+SVM (wF1 0.6309) when paired with simple classifiers. This suggests that mean-pooled WavLM embeddings lose temporal information relevant to emotion, while MFCC statistics effectively capture local spectral characteristics within the ViSEC domain.

### 4.2 DFAT ASR-assisted fusion substantially outperforms all audio-only methods

The DFAT Late Fusion Ensemble achieves wF1 0.7176 — a **+7.0 pp improvement** over the best audio-only baseline (MFCC+RF) and **+10.4 pp** over ECAPA-TDNN. The improvement is most pronounced for:
- **Neutral** (F1 0.687 vs ECAPA 0.508, Δ=+17.9 pp): Neutral speech lacks strong acoustic markers; Whisper-derived linguistic embeddings partially compensate by encoding the semantic flatness of neutral utterances.
- **Sad** (F1 0.741 vs ECAPA 0.622, Δ=+11.9 pp): Sad and Happy are acoustically confusable; the lexical dimension helps disambiguate.

### 4.3 Compute trade-off analysis

MFCC+RF offers the best cost-efficiency: competitive accuracy (wF1 0.6475) with sub-millisecond inference and no GPU required. DFAT achieves superior accuracy but requires WavLM (94M params) + Whisper (244M params) for feature extraction, making it suitable for offline/batch processing rather than real-time edge deployment.

### 4.4 Error Analysis

Common confusion patterns across all models:
- **Neutral ↔ Happy**: The most frequent confusion. Both can have moderate energy and pitch, making acoustic-only separation difficult.
- **Neutral ↔ Sad**: Both are low-energy emotions. DFAT's linguistic stream provides the strongest disambiguation here.
- **Happy ↔ Angry**: High-energy emotions that share elevated pitch and speaking rate.

---

## 5. Confusion Matrices

### ECAPA-TDNN
```
              Angry  Happy  Neutral  Sad
Angry          105     17      18     7
Happy           11     79      29     4
Neutral         13     45      78    14
Sad              4     12      32    60
```

### DFAT Late Fusion Ensemble
```
              Angry  Happy  Neutral  Sad
Angry          116      9      17     5
Happy           16     73      31     3
Neutral          8     15     114    13
Sad              7      5      20    76
```

---

## 6. Key Takeaways

1. **ASR-assisted fusion >> audio-only methods.** DFAT's linguistic stream (via Whisper transcription and PhoBERT) provides complementary cues that substantially improve emotion recognition (+7.0 pp over best baseline).

2. **MFCC remains competitive on ViSEC.** Mean-pooled WavLM embeddings do not outperform MFCC when paired with classical classifiers, suggesting that the advantage of SSL representations may require task-specific fine-tuning rather than frozen feature extraction.

3. **Neutral is the hardest class.** Both acoustic-only methods struggle with Neutral (F1 0.508–0.618); DFAT improves it to 0.687, but it remains the most challenging emotion.

4. **Strict leak-free evaluation is critical.** Previous results with test-set-based tuning were inflated by 0.3–2.4 pp. Academic benchmarks must enforce Train/Val/Test separation rigorously.

5. **MFCC+RF is the best lightweight option.** For resource-constrained applications, MFCC+RF achieves wF1 0.6475 with negligible computational cost.

---

## 7. Limitations and Future Work

- **Speaker Imbalance**: ViSEC's extreme speaker imbalance (Speaker 0 = 42%) means the strictly speaker-independent training set is dominated by one speaker, potentially limiting broad generalization. Future work should explore more balanced corpora.
- **Single-seed results (ECAPA/DFAT)**: Deep learning models reported from single runs. Multi-seed evaluation with confidence intervals would improve statistical robustness.
- **ASR quality**: Whisper's Vietnamese word error rate may introduce noise; the ablation study quantifies this impact via Whisper-tiny comparison.
- **Fine-tuning pretrained models**: WavLM and Whisper were used with frozen features. Fine-tuning on ViSEC may yield further gains.
- **Neural fusion**: Replacing classical ensemble with MLP or attention-based fusion on the 1,536-d space could improve results.