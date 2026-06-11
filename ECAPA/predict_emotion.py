#!/usr/bin/env python3
"""
Emotion Prediction Script using ECAPA-TDNN
Usage: python predict_emotion.py input_audio.wav --model_dir /path/to/model
"""

import argparse
import json
import sys
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== MODEL DEFINITION ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=False)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        s = s.unsqueeze(2)
        return x * s


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN for speech embeddings"""
    def __init__(self, input_size=80, channels=256, emb_size=192):
        super(ECAPA_TDNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, channels, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.layer1 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, dilation=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=3, dilation=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.se1 = SEBlock(channels)
        self.se2 = SEBlock(channels)
        self.se3 = SEBlock(channels)

        self.conv2 = nn.Conv1d(channels * 3, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.fc1 = nn.Linear(channels * 2, channels)
        self.bn3 = nn.BatchNorm1d(channels)

        self.fc2 = nn.Linear(channels, emb_size)
        self.bn4 = nn.BatchNorm1d(emb_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x1 = self.layer1(x)
        x1 = self.se1(x1)
        
        x2 = self.layer2(x)
        x2 = self.se2(x2)
        
        x3 = self.layer3(x)
        x3 = self.se3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        mean = x.mean(dim=2)
        std = x.std(dim=2, unbiased=False)
        x = torch.cat([mean, std], dim=1)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn4(x)

        return x


class EmotionClassifier(nn.Module):
    """Emotion Classifier with ECAPA-TDNN backbone"""
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.ecapa = ECAPA_TDNN(input_size=80, channels=256, emb_size=192)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        emb = self.ecapa(x)
        emb = self.dropout(emb)
        out = self.classifier(emb)
        return out


# ==================== FEATURE EXTRACTION ====================

def load_audio(audio_path, sr=16000):
    """Load audio file"""
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def extract_features(audio, sr=16000, n_mels=80):
    """Extract mel-spectrogram features"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=512,
        hop_length=160,
        win_length=400
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

    return log_mel_spec


# ==================== PREDICTION ====================

def predict_emotion(audio_path, model_path, emotion_labels=None):
    """
    Predict emotion from audio file
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model (.pth file)
        emotion_labels: List of emotion labels (default: Vietnamese emotions)
    
    Returns:
        Dictionary with emotion probabilities
    """
    # Try to load emotion labels from model metadata if not provided
    if emotion_labels is None:
        model_dir = Path(model_path).parent
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as _f:
                _meta = json.load(_f)
                emotion_labels = _meta.get('emotion_labels', ['angry', 'happy', 'neutral', 'sad'])
        else:
            emotion_labels = ['angry', 'happy', 'neutral', 'sad']
    
    num_classes = len(emotion_labels)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")
    
    model.eval()
    
    # Load and process audio
    audio = load_audio(audio_path, sr=16000)
    features = extract_features(audio, sr=16000, n_mels=80)
    
    # Convert to tensor and add batch dimension
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Create result dictionary
    result = {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}
    
    return result


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Predict emotion from audio using ECAPA-TDNN model'
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to input audio file (WAV format)'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to model directory or .pth file'
    )
    parser.add_argument(
        '--emotions',
        type=str,
        nargs='+',
        default=None,
        help='List of emotion labels (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        choices=['json', 'pretty'],
        default='json',
        help='Output format: json (compact) or pretty (formatted)'
    )
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine model path
    model_dir = Path(args.model_dir)
    if model_dir.is_file() and model_dir.suffix == '.pth':
        model_path = model_dir
    elif model_dir.is_dir():
        # Look for model file in directory
        possible_names = ['best_ecapa_model.pth', 'model.pth', 'checkpoint.pth']
        model_path = None
        for name in possible_names:
            candidate = model_dir / name
            if candidate.exists():
                model_path = candidate
                break
        if model_path is None:
            print(f"Error: No model file found in {args.model_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Invalid model path: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Predict
    try:
        result = predict_emotion(
            str(audio_path),
            str(model_path),
            emotion_labels=args.emotions
        )
        
        # Output result
        if args.output == 'json':
            print(json.dumps(result))
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()