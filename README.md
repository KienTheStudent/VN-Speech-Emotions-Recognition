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
3. **Proposed Method** — DFAT Hybrid Fusion: ASR-assisted audio-linguistic fusion combining WavLM acoustic embeddings with PhoBERT linguistic embeddings via hard invalid-text fallback and a single XGBoost classifier.


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
| Method | Category | wF1 (mean ± std) | mF1 | Acc |
|--------|----------|------------------|-----|-----|
| **DFAT Hybrid Fusion (Proposed)** | **Primary** | 0.4897 ± 0.0122 | 0.4841 | 0.4929 |
| **ECAPA-TDNN (simplified implementation)** | **Primary** | 0.4125 ± 0.0171 | 0.4122 | 0.4112 |
| **MFCC+RandomForest** | **Primary** | 0.3510 ± 0.0026 | 0.3365 | 0.3716 |
<!-- END_BENCHMARK_TABLE -->

### DFAT Ablation Study

<!-- START_ABLATION_TABLE -->
| Configuration | wF1 | mF1 | Acc |
|---------------|-----|-----|-----|
| Phase 1: Raw Concat (No Scaler) | 0.4913 | 0.4852 | 0.4938 |
| Phase 2: Per-Stream Scaler | 0.4897 | 0.4841 | 0.4929 |
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

**ASR-Assisted Audio-Linguistic Fusion (Single-Axis Protocol)**

- **Stream 1 — SEFE (Acoustic)**: WavLM-base-plus → 768-d acoustic embeddings
- **Stream 2 — TEFE (Linguistic)**: PhoWhisper-large ASR → Vietnamese text → `underthesea` word segmentation → PhoBERT-base-v2 → 768-d linguistic embeddings
- **Hard Invalid-Text Fallback**: If the ASR transcript is empty or fails, the linguistic embedding is zeroed out (automatically falling back to acoustic-only).
- **Early Fusion**: Concatenation → 1,536-d features (optional Per-Stream StandardScaler).
- **Classifier**: Single XGBoost classifier, Optuna-tuned (5 core hyperparameters, 30 trials) on a speaker-aware validation split.

> **Note**: The linguistic stream is derived from ASR transcription of the *same audio signal*, not from an independent text source. We therefore characterize this as **ASR-assisted audio-linguistic fusion** rather than true multimodal learning.

```
Audio → [WavLM] → acoustic embedding (768-d)
Audio → [PhoWhisper ASR] → text → [word segmentation] → [PhoBERT] → linguistic embedding (768-d)
→ Concat (1536-d) → [XGBoost] → Emotion Label
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
- Pre-trained model authors (WavLM, PhoWhisper, PhoBERT, ECAPA-TDNN)

---

**Last updated**: June 2026