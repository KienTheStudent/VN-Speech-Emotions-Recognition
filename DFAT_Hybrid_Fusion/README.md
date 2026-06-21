# DFAT Hybrid Fusion (Dual-Stream Emotion Recognition)

Speech emotion recognition using WavLM + PhoWhisper + Single XGBoost with Hard Invalid-Text Fallback.

## Installation

```bash
pip install datasets librosa soundfile transformers torch torchaudio scikit-learn xgboost optuna openai-whisper
```

## Structure

```
.
├── train_dualstream.py       # Train model
├── predict_dualstream.py      # Prediction
├── ablation_study.py          # Ablation study runner
└── dualstream_model/          # Trained models
    ├── xgb_model.pkl
    ├── scaler.pkl
    └── metadata.json
```

## Training

```bash
python train_dualstream.py [--asr_model vinai/PhoWhisper-large] [--use_scaler]
```

The script will:

* Parse the strict `split_manifest.json`
* Extract features with WavLM (acoustic) and PhoWhisper/PhoBERT (linguistic)
* Apply a Hard Invalid-Text Fallback (zeroes text stream if transcript fails)
* Tune a single XGBoost classifier via Optuna (30 trials) on the validation set
* Save the best model to the `dualstream_model/` directory

## Prediction

```bash
python predict_dualstream.py input.wav --model_dir dualstream_model
```

Output:

```json
{"angry": 0.7, "happy": 0.1, "neutral": 0.1, "sad": 0.05}
```

## Usage in Python

```python
from predict_dualstream import predict_emotion

result = predict_emotion("audio.wav", "dualstream_model")
print(result)
# {'angry': 0.7, 'happy': 0.1, ...}
```

## Requirements

* Python 3.8+
* CUDA (optional, to accelerate with GPU)
* RAM: 8GB+
* Audio: WAV, 16kHz
```
