import json
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from src.data.loader import load_manifest

print("Loading dualstream features...")
# To quickly test this, we would need to run extract features or load them.
# The features are not cached on disk in `train_dualstream.py`, they are extracted on the fly!
