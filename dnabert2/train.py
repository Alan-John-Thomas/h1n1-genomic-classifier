import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from model import DNABERT2Classifier
from dataset import create_dataloader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        total = len(iterable)
        for i, item in enumerate(iterable):
            if i % 10 == 0 or i == total - 1:
                print(f"{desc}: {i+1}/{total} batches", end="\r")
            yield item
        print()


def extract_embeddings(model, dataloader, device, desc="Extracting"):
    """Extracts 768-dim sequence representations once through frozen DNABERT-2."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc=desc):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Masked mean pooling across sequence tokens -> [batch, 768]
            seq_embeddings = model.extract_features(input_ids, attention_mask)

            all_embeddings.append(seq_embeddings.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_csv = os.path.join(project_root, "data", "processed", "train.csv")
    val_csv = os.path.join(project_root, "data", "processed", "val.csv")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "dnabert2_best.pt")

    # 1. Initialize foundation model
    print("Loading DNABERT-2 foundation model...")
    dnabert2_full = DNABERT2Classifier(num_classes=3).to(device)
    tokenizer = dnabert2_full.tokenizer

    # 2. Data loaders
    print("Preparing data loaders...")
    train_loader = create_dataloader(train_csv, tokenizer, batch_size=8, shuffle=False)
    val_loader = create_dataloader(val_csv, tokenizer, batch_size=8, shuffle=False)

    # 3. Extract & cache embeddings once
    cache_dir = os.path.join(current_dir, "cached_embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_emb.pt")
    val_cache = os.path.join(cache_dir, "val_emb.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached embeddings from disk...")
        train_x, train_y = torch.load(train_cache)
        val_x, val_y = torch.load(val_cache)
    else:
        print("\nExtracting feature embeddings through frozen DNABERT-2 (one-time pass)...")
        train_x, train_y = extract_embeddings(dnabert2_full, train_loader, device, desc="Train Embeddings")
        val_x, val_y = extract_embeddings(dnabert2_full, val_loader, device, desc="Val Embeddings")
        torch.save((train_x, train_y), train_cache)
        torch.save((val_x, val_y), val_cache)
        print("Embeddings cached successfully!")

    # 4. Train the lightweight classifier head (lightning fast!)
    classifier = dnabert2_full.classifier.to(device)
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
            torch.save(dnabert2_full.state_dict(), best_model_path)

    print(f"\nTraining Complete! Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    train()
