# Dual-Stream Emotion Recognition

Speech emotion recognition using WavLM + Whisper + Ensemble.

## Installation

```bash
pip install datasets librosa soundfile transformers torch torchaudio scikit-learn xgboost optuna openai-whisper
```

## Structure

```
.
├── train_dualstream.py       # Train model
├── predict_dualstream.py      # Prediction
└── dualstream_model/          # Trained models
    ├── lr_model.pkl
    ├── rf_model.pkl
    ├── xgb_model.pkl
    ├── scaler.pkl
    └── metadata.json
```

## Training

```bash
python train_dualstream.py
```

The script will:

* Download the ViSEC dataset
* Extract features with WavLM (acoustic) and Whisper (textual)
* Train 3 models: Logistic Regression, Random Forest, XGBoost
* Optimize ensemble weights
* Save to the `dualstream_model/` directory

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
