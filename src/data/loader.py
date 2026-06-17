import io
import json
import librosa
from pathlib import Path
from src.config.paths import SPLIT_MANIFEST_PATH
from src.data.schemas import AudioSample
from internal.split_validator import validate_manifest

def load_manifest():
    """Load the fixed split manifest and validate."""
    checksum = validate_manifest()
    with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest loaded: {manifest['total_samples']} total samples")
    print(f"  Train: {len(manifest['train_indices'])}")
    print(f"  Val:   {len(manifest['val_indices'])}")
    print(f"  Test:  {len(manifest['test_indices'])}")
    return manifest, checksum

def load_audio_sample(path_dict, label=None, sr=16000) -> AudioSample:
    """Load audio and return a standard AudioSample object."""
    sample = AudioSample(path=str(path_dict), label=label, sr=sr)
    try:
        if isinstance(path_dict, dict) and "bytes" in path_dict:
            audio, _ = librosa.load(io.BytesIO(path_dict["bytes"]), sr=sr)
            sample.audio = audio
            sample.path = "bytes"
        elif isinstance(path_dict, dict) and "path" in path_dict:
            audio, _ = librosa.load(path_dict["path"], sr=sr)
            sample.audio = audio
            sample.path = path_dict["path"]
        elif isinstance(path_dict, str):
            audio, _ = librosa.load(path_dict, sr=sr)
            sample.audio = audio
        else:
            sample.status = "error"
            sample.error_reason = "Unknown path format"
    except Exception as e:
        sample.status = "error"
        sample.error_reason = str(e)
    
    return sample
