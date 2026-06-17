import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, method="mfcc"):
        self.method = method
        
    def extract(self, audio):
        if self.method == "mfcc":
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            return np.mean(mfcc, axis=1)
        raise ValueError(f"Unknown feature method: {self.method}")
