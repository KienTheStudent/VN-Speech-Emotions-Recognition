# Vietnamese Speech Emotion Recognition (SER)

**Comparative Benchmarking and ASR-Assisted Audio-Linguistic Fusion for Vietnamese SER on ViSEC**

This repository implements and evaluates multiple Speech Emotion Recognition approaches on the ViSEC (Vietnamese Speech Emotion Corpus) dataset under a strict **speaker-independent, leak-free evaluation protocol**.

## Project Artifacts

To present a unified narrative, this repository is organized into three main artifacts:
1. **Academic Report (`Report_SER.tex`)**: The formal, academic source of truth detailing the motivation, methodology, and exhaustive findings.
2. **Evaluation Pipeline (Scripts)**: The raw Python scripts (`benchmark_methods_gpu.py`, `DFAT_Hybrid_Fusion/`) that enforce the strict, reproducible evaluation protocol.
3. **Live Demo (`internal/SER.ipynb`)**: An interactive Jupyter Notebook serving purely as a demonstration of the inference pipeline and live audio evaluation.

## Research Narrative

> **"Does audio-linguistic fusion outperform classical and strong acoustic baselines for Vietnamese SER under speaker-disjoint evaluation?"**

The thesis follows a three-tier comparison:

1. **Classical Baseline** — MFCC + RandomForest: lightweight, interpretable, no deep learning required.
2. **Strong Acoustic Baseline** — ECAPA-TDNN (simplified implementation): end-to-end deep learning with temporal/spectral modeling.
3. **Proposed Method** — DFAT Hybrid Fusion: ASR-assisted audio-linguistic fusion combining WavLM acoustic embeddings with PhoBERT linguistic embeddings (derived via Whisper ASR).


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
| **DFAT Late Fusion** | **Primary** | 0.7176 | 0.7166 | 0.7178 | N/A (GPU-bound, not comparable with CPU baselines) |
| **ECAPA-TDNN** | **Primary** | 0.6137 | 0.6131 | 0.6098 | N/A (GPU-bound, not comparable with CPU baselines) |
| **MFCC+RandomForest** | **Primary** | 0.3651 ± 0.0094 | 0.3513 | 0.3939 | 3.23 ms |
<!-- END_BENCHMARK_TABLE -->

### DFAT Ablation Study

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

## Methods

### 1. MFCC + RandomForest (Classical Baseline)

- **Features**: 40-dimensional MFCC, mean-pooled per utterance
- **Classifier**: RandomForest with 300 estimators
- **Rationale**: Demonstrates that handcrafted spectral features remain competitive for Vietnamese SER, establishing a strong cost-effective baseline.

### 2. ECAPA-TDNN-based Acoustic Baseline

- **Input**: 80-channel log-mel spectrogram
- **Architecture**: ECAPA-TDNN-style with 256 channels, 192-d embeddings, SE blocks (~1M parameters)
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

```text
.
├── benchmark_methods_gpu.py    # Unified benchmark pipeline (5-seed)
├── benchmark_results_gpu.json  # Single source of truth for results
├── split_manifest.json         # Fixed Train/Val/Test manifest
├── README.md                   # Project overview & instructions
├── Report_SER.tex              # Academic thesis and formal findings
├── insight.md                  # Experimental insight report
│
├── ECAPA/                      # Deep Acoustic Baseline
│   ├── train_emotion_model.py  # ECAPA-TDNN training script
│   ├── predict_emotion.py      # ECAPA-TDNN inference
│   └── emotion_model/          # Trained checkpoint + metadata
│
├── DFAT_Hybrid_Fusion/         # Proposed Audio-Linguistic Fusion
│   ├── train_dualstream.py     # DFAT training script
│   ├── predict_dualstream.py   # DFAT inference
│   ├── ablation_study.py       # DFAT ablation experiments
│   └── dualstream_model/       # Trained models + metadata
│
└── internal/                   # Utilities, generators, and demo
    ├── SER.ipynb               # Live Inference Demo (Notebook)
    ├── sync_results.py         # Auto-generates tables in docs
    └── generate_splits.py      # Split & manifest generator
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
python internal/sync_results.py
```

## License

MIT License

## Acknowledgments

- ViSEC dataset creators (HUSTEP Lab)
- HuggingFace Transformers team
- Pre-trained model authors (WavLM, Whisper, PhoBERT, ECAPA-TDNN)

---

**Last updated**: June 2026