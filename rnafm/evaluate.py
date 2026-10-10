import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from model import RNAFMClassifier
from dataset import create_dataloader

def evaluate():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths
    test_csv = os.path.join("..", "data", "processed", "test.csv")
    model_path = os.path.join("..", "checkpoints", "rnafm_best.pt")
    logs_dir = os.path.join("..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    cm_plot_path = os.path.join(logs_dir, "confusion_matrix_rnafm.png")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Please run train.py first!")
        return

    # 3. Load model and checkpoint
    classifier = RNAFMClassifier(num_classes=3).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()
    alphabet = classifier.alphabet

    # 4. Load test data
    print("Loading test data...")
    test_loader = create_dataloader(test_csv, alphabet, batch_size=4, shuffle=False)

    # 5. Inference loop
    test_preds, test_targets = [], []
    print("Running evaluation on test set...")

    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = classifier(tokens)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # 6. Compute metrics
    acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average="macro")
    class_names = ["Human", "Swine", "Avian"]

    print("\n" + "="*50)
    print("          RNA-FM FINAL TEST RESULTS")
    print("="*50)
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    print(f"  Macro-F1 Score: {macro_f1:.4f}")
    print("="*50)
    print("\nClassification Report:\n")
    print(classification_report(test_targets, test_preds, target_names=class_names))

    # 7. Generate and save Confusion Matrix plot
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("RNA-FM: Host Tropism Confusion Matrix")
    plt.xlabel("Predicted Host")
    plt.ylabel("Actual Host")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    print(f"Confusion matrix plot saved to: {cm_plot_path}")

if __name__ == "__main__":
    evaluate()