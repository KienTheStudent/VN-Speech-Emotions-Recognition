# ECAPA-TDNN Emotion Recognition

Speech emotion recognition system for Vietnamese using the ECAPA-TDNN model.

## Installation

```bash
pip install datasets librosa soundfile torch torchaudio scikit-learn huggingface_hub numpy pandas matplotlib seaborn
```

## Directory Structure

```
.
├── predict_emotion.py       # Emotion prediction script
├── train_emotion_model.py   # Model training script
├── emotion_model/           # Directory containing the trained model
│   ├── best_ecapa_model.pth
│   └── metadata.json
└── README.md
```

## 1. Model Training

To train the model from scratch with the ViSEC dataset:

```bash
python train_emotion_model.py
```

The script will:

* Download the ViSEC dataset from HuggingFace
* Extract mel-spectrogram features
* Train the ECAPA-TDNN model
* Save the best model to `emotion_model/best_ecapa_model.pth`
* Save metadata (emotion list, F1 score, etc.) to `emotion_model/metadata.json`

## 2. Emotion Prediction

### Basic Usage

```bash
python predict_emotion.py input_audio.wav --model_dir emotion_model
```

Output (JSON format):

```json
{"angry": 0.7, "happy": 0.1, "neutral": 0.05, "sad": 0.03}
```

### Options

**1. Specify a custom model file:**

```bash
python predict_emotion.py audio.wav --model_dir emotion_model/best_ecapa_model.pth
```

**2. Pretty-print output (with indentation):**

```bash
python predict_emotion.py audio.wav --model_dir emotion_model --output pretty
```

Output:

```json
{
  "angry": 0.7,
  "happy": 0.1,
  "neutral": 0.05,
  "sad": 0.03,
}
```

## 3. Python API

```python
from predict_emotion import predict_emotion

# Predict emotion
result = predict_emotion(
    audio_path="input_audio.wav",
    model_path="emotion_model/best_ecapa_model.pth"
)

print(result)
# {'angry': 0.7, 'happy': 0.1, 'neutral': 0.1, 'sad': 0.1, ...}

# Get the emotion with the highest probability
top_emotion = max(result, key=result.get)
print(f"Predicted emotion: {top_emotion} ({result[top_emotion]:.2%})")
```

## 4. Batch Processing

```bash
#!/bin/bash
# process_batch.sh

for audio in audio_files/*.wav; do
    echo "Processing: $audio"
    python predict_emotion.py "$audio" --model_dir emotion_model
done
```

## 5. API Format

### Input

* **Audio file**: WAV format (16kHz mono recommended)
* **Model directory**: Directory containing the `.pth` file and `metadata.json`

### Output

JSON object with emotion names as keys and probabilities (0-1) as values:

```json
{
  "angry": 0.15,
  "happy": 0.45,
  "neutral": 0.12,
  "sad": 0.10,
}
```

## 6. Supported Emotions

The ViSEC dataset supports 4 emotions:

* `angry`
* `happy`
* `neutral`
* `sad`

## 7. System Requirements

* Python 3.7+
* CUDA (optional, to accelerate with GPU)
* RAM: Minimum 4GB
* Disk: ~500MB for model and dependencies

## 8. Troubleshooting

### Error: "Audio file not found"

Check if the audio file path is correct.

### Error: "No model file found"

Ensure the model directory contains the `best_ecapa_model.pth` or `model.pth` file.

### Error: "Error loading model"

Check if the model was trained with the same number of emotions.

### Poor Audio Quality

* Use uncompressed WAV audio
* Recommended sampling rate: 16kHz
* Mono channel
* Minimum duration: 1 second

## 9. License & Citation

If you use this code, please cite the ViSEC dataset:

```bibtex
@dataset{visec2024,
  title={ViSEC: Vietnamese Speech Emotion Corpus},
  author={HUSTEP Lab},
  year={2024},
  publisher={HuggingFace}
}
```