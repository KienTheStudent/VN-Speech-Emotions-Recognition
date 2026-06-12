# Vietnamese Speech Emotion Recognition (SER)

**Comparative Benchmarking and ASR-Assisted Audio-Linguistic Fusion for Vietnamese SER on ViSEC**

This repository implements and evaluates multiple Speech Emotion Recognition approaches on the ViSEC (Vietnamese Speech Emotion Corpus) dataset under a strict **speaker-independent, leak-free evaluation protocol**.

## Research Narrative

> **"Does audio-linguistic fusion outperform classical and strong acoustic baselines for Vietnamese SER under speaker-disjoint evaluation?"**

The thesis follows a three-tier comparison:

1. **Classical Baseline** — MFCC + RandomForest: lightweight, interpretable, no deep learning required.
2. **Strong Acoustic Baseline** — ECAPA-TDNN: end-to-end deep learning with temporal/spectral modeling.
3. **Proposed Method** — DFAT Hybrid Fusion: ASR-assisted audio-linguistic fusion combining WavLM acoustic embeddings with PhoBERT linguistic embeddings (derived via Whisper ASR).

Additional baselines (MFCC+SVM, MFCC+XGBoost, WavLM+LogReg, WavLM+SVM) are included as secondary references in the supplementary benchmark.

## Dataset

**ViSEC** from HuggingFace: `hustep-lab/ViSEC`
- **Total samples**: 5,280 utterances from 147 speakers
- **Emotions**: 4 classes — angry, happy, neutral, sad
- **Language**: Vietnamese
- **Split**: Speaker-independent — ~80% Train / 10% Val / 10% Test (from `split_manifest.json`)

## Evaluation Protocol

- **Split**: Greedy speaker-independent allocation; no speaker appears in more than one partition.
- **Metrics**: Weighted F1 (primary), Macro F1, Accuracy, per-class Precision/Recall/F1.
- **Repeated runs**: Classical ML classifiers evaluated with 5 random seeds (mean ± std).
- **Leak-free**: All hyperparameter tuning on Validation set only; Test set evaluated exactly once.

## Results

### Main Benchmark

<!-- START_BENCHMARK_TABLE -->
| Method | Category | wF1 (mean ± std) | mF1 | Acc | E2E Latency |
|--------|----------|------------------|-----|-----|-------------|
| MFCC+RandomForest | Secondary | 0.6475 | 0.6463 | 0.6477 | — |
| MFCC+XGBoost | Secondary | 0.6439 | 0.6425 | 0.6439 | — |
| MFCC+SVM | Secondary | 0.6366 | 0.6374 | 0.6364 | — |
| WavLM+SVM | Secondary | 0.6309 | 0.6307 | 0.6307 | — |
| WavLM+LogReg | Secondary | 0.5713 | 0.5705 | 0.5701 | — |
<!-- END_BENCHMARK_TABLE -->

### DFAT Ablation Study

<!-- START_ABLATION_TABLE -->

<!-- END_ABLATION_TABLE -->

## Methods

### 1. MFCC + RandomForest (Classical Baseline)

- **Features**: 40-dimensional MFCC, mean-pooled per utterance
- **Classifier**: RandomForest with 300 estimators
- **Rationale**: Demonstrates that handcrafted spectral features remain competitive for Vietnamese SER, establishing a strong cost-effective baseline.

### 2. ECAPA-TDNN (Strong Acoustic Baseline)

- **Input**: 80-channel log-mel spectrogram
- **Architecture**: ECAPA-TDNN with 256 channels, 192-d embeddings, SE blocks (~1M parameters)
- **Training**: Up to 100 epochs with Adam (lr=3e-4), ReduceLROnPlateau on validation F1, early stopping (patience=20)
- **Rationale**: A purpose-built temporal deep learning baseline that learns emotion-discriminative features end-to-end, providing a strong acoustic-only comparison.

### 3. DFAT Hybrid Fusion (Proposed Method)

**ASR-Assisted Audio-Linguistic Fusion**

- **Stream 1 — SEFE (Acoustic)**: WavLM-base-plus → 768-d acoustic embeddings
- **Stream 2 — TEFE (Linguistic)**: Whisper-small ASR → Vietnamese text → `underthesea` word segmentation → PhoBERT-base-v2 → 768-d linguistic embeddings
- **Early Fusion**: Concatenation → 1,536-d features, StandardScaler normalized
- **Classifiers**: LR, RF, XGBoost (Optuna-tuned on validation, 10 trials)
- **Late Fusion**: Weighted probability ensemble, weights optimized by Optuna (100 trials) on validation

> **Note**: The linguistic stream is derived from ASR transcription of the *same audio signal*, not from an independent text source. We therefore characterize this as **ASR-assisted audio-linguistic fusion** rather than true multimodal learning.

```
Audio → [WavLM] → acoustic embedding (768-d)
Audio → [Whisper ASR] → text → [word segmentation] → [PhoBERT] → linguistic embedding (768-d)
→ Concat (1536-d) → [LR + RF + XGB] → Weighted Voting
```

## Repository Structure

```
.
├── benchmark_methods_gpu.py    # Unified benchmark (5-seed repeated eval)
├── sync_results.py             # Auto-generate tables in README/report/insight
├── generate_splits.py          # Speaker-independent split generation
├── split_manifest.json         # Fixed Train/Val/Test manifest
├── benchmark_results_gpu.json  # Single source of truth for results
│
├── ECAPA/
│   ├── train_emotion_model.py  # ECAPA-TDNN training script
│   ├── predict_emotion.py      # ECAPA-TDNN inference
│   └── emotion_model/          # Trained checkpoint + metadata
│
├── DFAT_Hybrid_Fusion/
│   ├── train_dualstream.py     # DFAT training script
│   ├── predict_dualstream.py   # DFAT inference
│   ├── ablation_study.py       # DFAT ablation experiments
│   └── dualstream_model/       # Trained models + metadata
│
├── Report_SER.tex              # LaTeX thesis report
├── insight.md                  # Experimental insight report
├── SER.ipynb                   # Live inference notebook
└── requirements.txt            # Python dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Benchmark (5-seed evaluation)

```bash
python benchmark_methods_gpu.py
```

### Train DFAT

```bash
python DFAT_Hybrid_Fusion/train_dualstream.py
```

### Run Ablation Study

```bash
python DFAT_Hybrid_Fusion/ablation_study.py
```

### Sync Results to Documentation

```bash
python sync_results.py
```

## License

MIT License

## Acknowledgments

- ViSEC dataset creators (HUSTEP Lab)
- HuggingFace Transformers team
- Pre-trained model authors (WavLM, Whisper, PhoBERT, ECAPA-TDNN)

---

**Last updated**: June 2026