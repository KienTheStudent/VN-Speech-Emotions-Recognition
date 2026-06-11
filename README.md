# Vietnamese Speech Emotion Recognition (SER)

A comprehensive implementation of multiple Speech Emotion Recognition models on the ViSEC (Vietnamese Speech Emotion Corpus) dataset, including traditional ML approaches, deep learning models, and state-of-the-art transformer-based architectures.

## 📊 Dataset

**ViSEC Dataset** from HuggingFace: `hustep-lab/ViSEC`
- **Total samples**: ~5,400 utterances
- **Emotions**: 4 classes (angry, happy, neutral, sad)
- **Language**: Vietnamese
- **Format**: Audio files with emotion labels

## 🔧 Common Requirements

Install all required dependencies:

```bash
# Core libraries (required for all models)
pip install datasets librosa soundfile transformers torch torchaudio scikit-learn
pip install matplotlib seaborn pandas numpy tqdm

# Additional libraries for specific models
pip install xgboost speechbrain optuna openai-whisper funasr modelscope addict
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## 🤖 Implemented Models

### 1. Traditional Machine Learning Models

**Models**: SVM, Random Forest, XGBoost
- **Features**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Feature dimension**: 40 MFCC coefficients
- **Pros**: Fast inference, interpretable

**Usage**:
```python
# Extract MFCC features
mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
mfcc_mean = np.mean(mfcc, axis=1)

# Train SVM/RF/XGBoost on extracted features
```

---

### 2. Wav2Vec2 Feature Extractor + SVM

**Architecture**: Wav2Vec2 (facebook/wav2vec2-base-960h) + SVM classifier
- **Features**: 768-dim embeddings from Wav2Vec2 layer 9
- **Pre-training**: Trained on English speech, fine-tuned for SER
- **Pros**: Leverages self-supervised learning

**Key code**:
```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# Extract layer 9 features for SER
```

---

### 3. HuBERT (Pre-trained)

**Model**: `superb/hubert-base-superb-er`
- **Pre-trained**: On emotion recognition task
- **Features**: Direct emotion predictions
- **Pros**: Ready-to-use for emotion recognition

**Usage**:
```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model = AutoModelForAudioClassification.from_pretrained("superb/hubert-base-superb-er")
```

---

### 4. WavLM (Microsoft)

**Model**: `microsoft/wavlm-base-plus`
- **Architecture**: Transformer-based speech model
- **Training**: Custom classification head on WavLM features
- **Features**: 1024-dim embeddings
- **Epochs**: 1 (quick training with pre-trained features)
- **Pros**: State-of-the-art speech understanding

**Training**:
```python
from transformers import WavLMForSequenceClassification

model = WavLMForSequenceClassification.from_pretrained(
    "microsoft/wavlm-base-plus",
    num_labels=4
)
```

---

### 5. ECAPA-TDNN

**Architecture**: Emphasized Channel Attention, Propagation and Aggregation in TDNN
- **Features**: Mel-spectrogram (80-channel)
- **Components**: 
  - Time Delay Neural Networks (TDNN)
  - Squeeze-and-Excitation blocks
  - Statistics pooling (mean + std)
- **Training**: From scratch with 100 epochs
- **Pros**: Specialized for speaker/emotion embeddings

**Key features**:
```python
class ECAPA_TDNN(nn.Module):
    # TDNN layers with SE blocks
    # Statistics pooling for robust features
    # Embedding dimension: 192
```

---

### 6. DFAT Hybrid Fusion Pipeline

**Architecture**: Dual-stream Feature Aggregation with Late Fusion

**Feature Extractors**:
1. **SEFE** (Speech Emotion Feature Extractor): WavLM → 1024-dim
2. **TEFE** (Textual Emotion Feature Extractor): Whisper → 1024-dim

**Fusion Strategy**:
- **Early Fusion**: Concatenate features (2048-dim)
- **Late Fusion**: Weighted ensemble of 3 classifiers
  - Logistic Regression
  - Random Forest
  - XGBoost (with Optuna optimization)
  
**Pros**: Multi-modal approach, robust ensemble

**Pipeline**:
```
Audio → [WavLM + Whisper] → Concat → [LR + RF + XGB] → Weighted Voting
```

---

### 7. CNN + GRU Hybrid (Improved)

**Architecture**: 
- **CNN**: 3 convolutional blocks for feature extraction from mel-spectrograms
- **GRU**: Bidirectional GRU (2 layers) for temporal modeling
- **Attention**: Attention mechanism for weighted pooling
- **Features**: 128-channel mel-spectrogram

**Training strategies**:
- Data augmentation (SpecAugment, pitch shift, time stretch)
- Class weights for imbalanced data
- Cosine annealing learning rate
- Early stopping
- Gradient clipping

**Pros**: End-to-end learning, handles temporal dynamics

**Key components**:
```python
class ImprovedCNNGRU(nn.Module):
    # CNN: 64 → 128 → 256 channels
    # BiGRU: 256 hidden units, 2 layers
    # Attention pooling
    # FC layers with dropout
```

---

### 8. SpeechFormer++

**Architecture**: Hierarchical Transformer with Multi-Model Feature Fusion

**Feature Extractors**:
1. **Emotion2Vec+**: `iic/emotion2vec_plus_base` (768-dim)
2. **Wav2Vec2 Large**: `facebook/wav2vec2-large-960h` (512-dim projected)

**Total features**: 1280-dim concatenated

**Transformer**:
- **Hierarchical structure**:
  - Fine-grained layers (detailed emotional nuances)
  - Coarse-grained layers (high-level patterns)
- **Fusion**: Concatenate + FC layer
- **Layers**: 4 transformer blocks (2 fine + 2 coarse)
- **Heads**: 8 attention heads

**Pros**: State-of-the-art multi-scale emotion modeling

**Training**:
```python
model = SpeechFormerPlusPlus(
    input_dim=1280,
    num_labels=4,
    num_layers=4,
    num_heads=8
)
```

---

### 9. Multi-Stage Training Framework

**Based on**: "ishowspeech" team's 2nd place solution at VLSP 2025 SER competition

**Key innovations**:

1. **Feature Extraction**: Wav2Vec2 layer 9/12 (768-dim)
2. **Hybrid Loss**: 
   - Cross-Entropy + Supervised Contrastive Learning
   - Formula: `L = (1-α)L_ce + αL_scl` (α=0.5)
3. **k-NN Interpolation**: 
   - Build k-NN database from training embeddings
   - Interpolate predictions: `p(y|x) = β*p_model + (1-β)*p_knn`
   - β=0.7, k=5
4. **Data Preprocessing**:
   - 128-channel filterbank
   - SpecAugment (F=27, pS=0.05)
   - 25ms window, 10ms stride

**Training stages**:
- Stage 1: Train with hybrid loss
- Stage 2: Build k-NN interpolation database
- Stage 3: Inference with k-NN refinement
**Pros**: State-of-the-art Vietnamese SER, competition-proven


---
### 1. Training Configuration

Most models can be configured with these parameters:

```python
# Common parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50-100
DROPOUT = 0.3

# Model-specific
# WavLM: 1 epoch (pre-trained)
# ECAPA-TDNN: 100 epochs (from scratch)
# CNN+GRU: 50 epochs with early stopping
# SpeechFormer++: 25 epochs
# Multi-Stage: 100 epochs
```

## 📊 Data Preprocessing

All models use consistent preprocessing:

```python
# Audio loading
sample_rate = 16000
min_duration = 0.5  # seconds

# Feature extraction
n_mels = 128  # for spectrograms
n_mfcc = 40   # for MFCC
window_size = 25  # ms
hop_length = 10   # ms

# Augmentation (training only)
spec_augment = True
freq_mask = 27
time_mask_ratio = 0.05
```

## 🎯 Model Selection Guide

**Choose based on your needs**:

1. **Fast inference + Limited resources** → SVM/RF/XGBoost
2. **Good accuracy + Moderate resources** → Wav2Vec2, WavLM
3. **Best accuracy + Research purpose** → Multi-Stage + k-NN
4. **Multi-modal features** → DFAT Fusion
5. **End-to-end learning** → CNN+GRU
6. **State-of-the-art** → SpeechFormer++ or Multi-Stage


## 📄 License

MIT License - See LICENSE file for details


## 🙏 Acknowledgments

- ViSEC dataset creators
- HuggingFace Transformers team
- VLSP 2025 competition organizers
- Pre-trained model authors (Wav2Vec2, WavLM, HuBERT, Emotion2Vec, Whisper)

---

**Last updated**: January 2026