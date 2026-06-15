# ==================== PART 6: ECAPA-TDNN ====================
print("🤖 CREATING AND TRAINING ECAPA-TDNN MODEL")

import sys
sys.path.append(os.path.abspath("./ECAPA"))
from train_emotion_model import prepare_features, AudioFeaturesDataset, collate_fn, train_epoch, evaluate
from predict_emotion import ECAPA_TDNN, EmotionClassifier
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(num_labels).to(device)

if MODE == "demo":
    checkpoint_path = "./ECAPA/best_ecapa_model.pth" if os.path.exists("./ECAPA/best_ecapa_model.pth") else "./ECAPA/emotion_model/best_ecapa_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"✨ DEMO MODE: Found checkpoint {checkpoint_path}, loading model...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        print("Extracting features for Test set...")
        X_test_feat, y_test_clean = prepare_features(X_test, y_test, "Test")
        test_dataset = AudioFeaturesDataset(X_test_feat, y_test_clean)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        test_preds, test_labels = evaluate(model, test_loader, device)
        from sklearn.metrics import f1_score, accuracy_score
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        print(f"✓ Final test F1 (weighted): {test_f1_weighted:.4f}")
        
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title("ECAPA-TDNN Confusion Matrix")
        plt.show()
    else:
        print("❌ ECAPA-TDNN checkpoint not found. Please switch to MODE = 'retrain'.")
else:
    print("🚀 RETRAIN MODE: Training model from scratch...")
    # Standard training loop from codebase
    print("Extracting features...")
    X_train_feat, y_train_clean = prepare_features(X_train, y_train, "Train")
    X_val_feat, y_val_clean = prepare_features(X_val, y_val, "Val")
    X_test_feat, y_test_clean = prepare_features(X_test, y_test, "Test")

    train_dataset = AudioFeaturesDataset(X_train_feat, y_train_clean)
    val_dataset = AudioFeaturesDataset(X_val_feat, y_val_clean)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    num_epochs = 20 # Shortened for notebook
    best_val_f1 = 0
    from sklearn.metrics import f1_score
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        if epoch >= 2: scheduler.step(val_f1)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
