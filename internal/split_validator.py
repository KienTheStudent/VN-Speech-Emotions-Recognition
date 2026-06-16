import json
import hashlib
from pathlib import Path

def validate_manifest():
    manifest_path = Path(__file__).resolve().parent.parent / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing split_manifest.json. Run generate_splits.py first.")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Verify checksum
    expected_checksum = manifest.pop("checksum", None)
    if not expected_checksum:
        raise ValueError("split_manifest.json is missing a checksum. Please regenerate it.")
        
    manifest_string = json.dumps(manifest, sort_keys=True)
    actual_checksum = hashlib.sha256(manifest_string.encode('utf-8')).hexdigest()
    
    if actual_checksum != expected_checksum:
        raise ValueError(f"split_manifest.json checksum mismatch!\nExpected: {expected_checksum}\nActual: {actual_checksum}\nThis indicates the file was manually modified or corrupted.")
        
    # Verify no overlap
    train_set = set(manifest["train_indices"])
    val_set = set(manifest["val_indices"])
    test_set = set(manifest["test_indices"])
    
    if len(train_set & val_set) > 0: raise ValueError("Train/Val overlap detected in manifest!")
    if len(train_set & test_set) > 0: raise ValueError("Train/Test overlap detected in manifest!")
    if len(val_set & test_set) > 0: raise ValueError("Val/Test overlap detected in manifest!")
    
    return expected_checksum
