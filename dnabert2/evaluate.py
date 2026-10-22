import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        for item in iterable:
            yield item

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import DNABERT2Classifier
from dataset import create_dataloader


def evaluate():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_csv = os.path.join(project_root, "data", "processed", "test.csv")
    model_path = os.path.join(project_root, "checkpoints", "dnabert2_best.pt")
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    cm_plot_path = os.path.join(logs_dir, "confusion_matrix_dnabert2.png")

    # 3. Check whether the trained model exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Please run train.py first!")
        return

    # 4. Load the trained model
    print("Loading trained DNABERT-2 classifier...")
    classifier = DNABERT2Classifier(num_classes=3).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()
    tokenizer = classifier.tokenizer

    # 5. Load test data
    print("Loading test data...")
    test_loader = create_dataloader(test_csv, tokenizer, batch_size=8, shuffle=False)
    print(f"Total test sequences: {len(test_loader.dataset)}")

    # 6. Evaluate the test dataset
    test_preds = []
    test_targets = []

    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Evaluating test set"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = classifier(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    # 7. Compute evaluation metrics
    acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average="macro")
    class_names = ["Human", "Swine", "Avian"]

    print("\n" + "=" * 50)
    print("        DNABERT-2 FINAL TEST RESULTS")
    print("=" * 50)
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Macro-F1 Score: {macro_f1:.4f}")
    print("=" * 50)

    print("\nClassification Report:\n")
    print(
        classification_report(
            test_targets,
            test_preds,
            labels=[0, 1, 2],
            target_names=class_names,
            zero_division=0
        )
    )

    # 8. Generate and save the confusion matrix
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("DNABERT-2: Host Tropism Confusion Matrix")
    plt.xlabel("Predicted Host")
    plt.ylabel("Actual Host")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    print(f"Confusion matrix saved to: {cm_plot_path}")


if __name__ == "__main__":
    evaluate()
