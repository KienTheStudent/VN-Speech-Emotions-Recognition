#!/usr/bin/env python3
"""
Training script for ECAPA-TDNN Emotion Recognition — Leak-Free Protocol.

Reads the fixed Train/Val/Test split from split_manifest.json.
- Train set: used for gradient updates.
- Val set: used for early stopping, scheduler stepping, and best checkpoint.
- Test set: evaluated exactly ONCE at the end for final reporting.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader

# Removed sys.path logic to keep import topology clean


warnings.filterwarnings('ignore')

# Import model classes from predict_emotion.py
from predict_emotion import ECAPA_TDNN, EmotionClassifier

MANIFEST_PATH = Path(__file__).parent.parent / "split_manifest.json"


# ==================== DATA LOADING ====================

def load_manifest():
    """Load the fixed split manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def load_audio(path_dict, sr=16000):
    """Load audio file from bytes or path"""
    try:
        import io
        if isinstance(path_dict, dict) and 'bytes' in path_dict:
            audio_bytes = path_dict['bytes']
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
            return audio
        elif isinstance(path_dict, str):
            audio, _ = librosa.load(path_dict, sr=sr)
            return audio
        elif isinstance(path_dict, dict) and 'path' in path_dict:
            audio, _ = librosa.load(path_dict['path'], sr=sr)
            return audio
    except Exception as e:
        return None
    return None


def extract_features(audio, sr=16000, n_mels=80):
    """Extract mel-spectrogram features"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=512, hop_length=160, win_length=400
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    return log_mel_spec


def prepare_features(audio_paths, labels, split_name=""):
    """Prepare features for a given split"""
    features_list = []
    labels_list = []

    for i, (path, label) in enumerate(zip(audio_paths, labels)):
        if i % 500 == 0:
            print(f"  [{split_name}] Processed {i}/{len(audio_paths)} samples...")

        audio = load_audio(path, sr=16000)
        if audio is not None and len(audio) > 0:
            features = extract_features(audio)
            features_list.append(features)
            labels_list.append(label)

    return features_list, np.array(labels_list)


# ==================== DATASET ====================

class AudioFeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return feat, label.squeeze()


def collate_fn(batch):
    """Custom collate function to pad sequences"""
    features, labels = zip(*batch)
    max_len = max([f.shape[1] for f in features])

    padded_features = []
    for feat in features:
        if feat.shape[1] < max_len:
            pad_len = max_len - feat.shape[1]
            feat = torch.nn.functional.pad(feat, (0, pad_len))
        padded_features.append(feat)

    features = torch.stack(padded_features)
    labels = torch.stack(list(labels))
    return features, labels


# ==================== TRAINING ====================

def train_epoch(model, loader, criterion, optimizer, device, epoch=0, warmup_epochs=2):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Learning rate warmup — only override during warmup phase
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003 * lr_scale

    for batch_idx, (features, labels) in enumerate(loader):
        features, labels = features.to(device), labels.to(device)

        if torch.isnan(features).any():
            continue

        optimizer.zero_grad()
        outputs = model(features)

        if torch.isnan(outputs).any():
            continue

        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    """Evaluate model — returns predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# ==================== MAIN TRAINING ====================

def main():
    print("=" * 60)
    print("TRAINING ECAPA-TDNN — LEAK-FREE PROTOCOL")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load manifest & dataset
    # ------------------------------------------------------------------
    print("\nLoading split manifest...")
    manifest = load_manifest()
    split_checksum = manifest.get("checksum", "unknown")
    print(f"  Train: {len(manifest['train_indices'])}")
    print(f"  Val:   {len(manifest['val_indices'])}")
    print(f"  Test:  {len(manifest['test_indices'])}")

    print("\nLoading ViSEC dataset...")
    dataset = load_dataset("hustep-lab/ViSEC", trust_remote_code=True)
    df = dataset['train'].to_pandas()
    print(f"Dataset: {len(df)} samples")

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])
    emotion_labels = le.classes_.tolist()
    num_labels = len(emotion_labels)
    print(f"Emotions: {emotion_labels}")

    # ------------------------------------------------------------------
    # 2. Split data using manifest
    # ------------------------------------------------------------------
    train_idx = manifest['train_indices']
    val_idx = manifest['val_indices']
    test_idx = manifest['test_indices']

    X_train = df['path'].iloc[train_idx].values
    y_train = df['label'].iloc[train_idx].values
    X_val = df['path'].iloc[val_idx].values
    y_val = df['label'].iloc[val_idx].values
    X_test = df['path'].iloc[test_idx].values
    y_test = df['label'].iloc[test_idx].values

    print(f"\nSplit sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Extract features
    # ------------------------------------------------------------------
    print("\nExtracting features...")
    X_train_feat, y_train_clean = prepare_features(X_train, y_train, "Train")
    X_val_feat, y_val_clean = prepare_features(X_val, y_val, "Val")
    X_test_feat, y_test_clean = prepare_features(X_test, y_test, "Test")

    print(f"  Features extracted: Train={len(X_train_feat)}, Val={len(X_val_feat)}, Test={len(X_test_feat)}")

    # ------------------------------------------------------------------
    # 4. Create dataloaders
    # ------------------------------------------------------------------
    train_dataset = AudioFeaturesDataset(X_train_feat, y_train_clean)
    val_dataset = AudioFeaturesDataset(X_val_feat, y_val_clean)
    test_dataset = AudioFeaturesDataset(X_test_feat, y_test_clean)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # 5. Initialize model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    seeds = [42, 123, 456, 789, 2026]
    runs = []
    
    output_dir = Path(__file__).parent / "emotion_model"
    output_dir.mkdir(exist_ok=True)

    for seed in seeds:
        print("\n" + "=" * 50)
        print(f"TRAINING SEED {seed}")
        print("=" * 50)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        model = EmotionClassifier(num_labels).to(device)
        class_counts = np.bincount(y_train_clean)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )

        num_epochs = 100
        best_val_f1 = 0
        patience = 20
        patience_counter = 0
        warmup_epochs = 2
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, warmup_epochs
            )
            
            if np.isnan(train_loss):
                print(f"NaN loss at epoch {epoch+1}, stopping")
                break

            val_preds, val_labels = evaluate(model, val_loader, device)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            if epoch >= warmup_epochs:
                scheduler.step(val_f1)

            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), output_dir / f'best_ecapa_model_seed_{seed}.pth')
                patience_counter = 0
            else:
                if epoch >= warmup_epochs:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation on TEST set for this seed
        model.load_state_dict(torch.load(output_dir / f'best_ecapa_model_seed_{seed}.pth', weights_only=True))
        test_preds, test_labels = evaluate(model, test_loader, device)

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        test_f1_macro = f1_score(test_labels, test_preds, average='macro')
        
        report_dict = classification_report(test_labels, test_preds, target_names=emotion_labels, output_dict=True)
        cm = confusion_matrix(test_labels, test_preds).tolist()
        
        runs.append({
            'seed': seed,
            'f1_weighted': float(test_f1_weighted),
            'f1_macro': float(test_f1_macro),
            'accuracy': float(test_acc),
            'best_val_f1': float(best_val_f1),
            'classification_report': report_dict,
            'confusion_matrix': cm
        })
        
        print(f"  -> Seed {seed} Test wF1: {test_f1_weighted:.4f}")

    # Calculate multi-seed stats
    wf1s = [r['f1_weighted'] for r in runs]
    mf1s = [r['f1_macro'] for r in runs]
    accs = [r['accuracy'] for r in runs]
    
    median_idx = int(np.argsort(wf1s)[len(wf1s) // 2])
    representative = runs[median_idx]
    
    # Keep only the representative model
    import shutil
    shutil.copyfile(output_dir / f'best_ecapa_model_seed_{representative["seed"]}.pth', output_dir / 'best_ecapa_model.pth')
    
    # Cleanup seed models
    for seed in seeds:
        (output_dir / f'best_ecapa_model_seed_{seed}.pth').unlink(missing_ok=True)
    
    metadata = {
        'protocol': 'Leak-free: Val for early stopping/scheduler, Test evaluated once, 5-seed repeated',
        'split_source': 'split_manifest.json',
        'split_checksum': split_checksum,
        'emotion_labels': emotion_labels,
        'num_classes': num_labels,
        'n_seeds': 5,
        'seeds': seeds,
        'test_accuracy_mean': float(np.mean(accs)),
        'test_accuracy_std': float(np.std(accs)),
        'test_f1_weighted_mean': float(np.mean(wf1s)),
        'test_f1_weighted_std': float(np.std(wf1s)),
        'test_f1_macro_mean': float(np.mean(mf1s)),
        'test_f1_macro_std': float(np.std(mf1s)),
        'representative_run': representative,
        'model_config': {
            'input_size': 80,
            'channels': 256,
            'emb_size': 192
        }
    }

    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Models saved to: {output_dir}/")
    print(f"✓ Mean test F1 (weighted): {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")

if __name__ == "__main__":
    main()