import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from model import RNAFMClassifier
from dataset import create_dataloader

try:
    from tqdm import tqdm
except ImportError:
    # Fallback simple progress bar if tqdm is not installed
    def tqdm(iterable, desc=""):
        total = len(iterable)
        for i, item in enumerate(iterable):
            if i % 10 == 0 or i == total - 1:
                print(f"{desc}: {i+1}/{total} batches", end="\r")
            yield item
        print()

def extract_embeddings(model, dataloader, device, desc="Extracting"):
    """Extracts 640-dim sequence representations once through frozen RNA-FM."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc=desc):
            tokens = tokens.to(device)
            results = model.model(tokens, repr_layers=[12])
            token_embeddings = results["representations"][12]
            
            # Mean pooling across sequence tokens -> [batch, 640]
            seq_embeddings = token_embeddings.mean(dim=1)
            
            all_embeddings.append(seq_embeddings.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_csv = os.path.join("..", "data", "processed", "train.csv")
    val_csv = os.path.join("..", "data", "processed", "val.csv")
    checkpoint_dir = os.path.join("..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "rnafm_best.pt")

    # 1. Initialize foundation model
    print("Loading RNA-FM model...")
    rnafm_full = RNAFMClassifier(num_classes=3).to(device)
    alphabet = rnafm_full.alphabet

    # 2. Data loaders (batch_size 8 for faster feature extraction on CPU)
    print("Preparing data loaders...")
    train_loader = create_dataloader(train_csv, alphabet, batch_size=8, shuffle=False)
    val_loader = create_dataloader(val_csv, alphabet, batch_size=8, shuffle=False)

    # 3. Extract & cache embeddings once
    cache_dir = os.path.join(".", "cached_embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_emb.pt")
    val_cache = os.path.join(cache_dir, "val_emb.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached embeddings from disk...")
        train_x, train_y = torch.load(train_cache)
        val_x, val_y = torch.load(val_cache)
    else:
        print("\nExtracting feature embeddings through frozen RNA-FM (one-time pass)...")
        train_x, train_y = extract_embeddings(rnafm_full, train_loader, device, desc="Train Embeddings")
        val_x, val_y = extract_embeddings(rnafm_full, val_loader, device, desc="Val Embeddings")
        torch.save((train_x, train_y), train_cache)
        torch.save((val_x, val_y), val_cache)
        print("Embeddings cached successfully!")

    # 4. Train the lightweight classifier head (lightning fast!)
    classifier = rnafm_full.classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_emb_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    epochs = 15
    best_val_f1 = 0.0

    print("\nTraining classification head...")
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0

        for x, y in train_emb_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_x.to(device))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_targets = val_y.numpy()

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {total_loss/len(train_emb_loader):.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(rnafm_full.state_dict(), best_model_path)

    print(f"\nTraining Complete! Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")

if __name__ == "__main__":
    train()