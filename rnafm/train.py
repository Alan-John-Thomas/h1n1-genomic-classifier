import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from model import RNAFMClassifier
from dataset import create_dataloader

def train():
    # 1. Device configuration (use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths to data and checkpoints
    train_csv = os.path.join("..", "data", "processed", "train.csv")
    val_csv = os.path.join("..", "data", "processed", "val.csv")
    checkpoint_dir = os.path.join("..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "rnafm_best.pt")

    # 3. Initialize model
    classifier = RNAFMClassifier(num_classes=3).to(device)
    alphabet = classifier.alphabet

    # 4. Load data
    print("Loading data...")
    train_loader = create_dataloader(train_csv, alphabet, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_csv, alphabet, batch_size=4, shuffle=False)

    # 5. Loss function and optimizer (only train the classification head)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs = 5
    best_val_f1 = 0.0

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        classifier.train()
        total_loss = 0.0

        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = classifier(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation Phase ---
        classifier.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                logits = classifier(tokens)
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        # Save best model checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(classifier.state_dict(), best_model_path)
            print(f"  --> Saved new best model to {best_model_path}")

    print(f"\nTraining Complete! Best Validation Macro-F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train()