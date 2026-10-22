
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import RNAFMClassifier
from dataset import create_dataloader


def evaluate():
    # 1. Device configuration
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Paths
    test_csv = os.path.join(
        "..", "data", "processed", "test.csv"
    )
    model_path = os.path.join(
        "..", "checkpoints", "rnafm_best.pt"
    )
    logs_dir = os.path.join("..", "logs")

    os.makedirs(logs_dir, exist_ok=True)

    cm_plot_path = os.path.join(
        logs_dir, "confusion_matrix_rnafm.png"
    )

    # 3. Check whether the trained model exists
    if not os.path.exists(model_path):
        print(
            f"Error: Model checkpoint not found at {model_path}. "
            "Please run train.py first!"
        )
        return

    # 4. Load the trained model
    print("Loading trained RNA-FM classifier...")

    classifier = RNAFMClassifier(num_classes=3).to(device)

    classifier.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    classifier.eval()
    alphabet = classifier.alphabet

    # 5. Load test data
    print("Loading test data...")

    test_loader = create_dataloader(
        test_csv,
        alphabet,
        batch_size=4,
        shuffle=False
    )

    print(f"Total test sequences: {len(test_loader.dataset)}")
    print(f"Total batches: {len(test_loader)}")

    # 6. Evaluate the test dataset with a progress bar
    test_preds = []
    test_targets = []

    print("\nRunning evaluation on test set...")

    with torch.no_grad():
        for tokens, labels in tqdm(
            test_loader,
            total=len(test_loader),
            desc="Evaluating test set",
            unit="batch",
            dynamic_ncols=True
        ):
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = classifier(tokens)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # 7. Compute evaluation metrics
    acc = accuracy_score(test_targets, test_preds)

    macro_f1 = f1_score(
        test_targets,
        test_preds,
        average="macro"
    )

    class_names = ["Human", "Swine", "Avian"]

    print("\n" + "=" * 50)
    print("          RNA-FM FINAL TEST RESULTS")
    print("=" * 50)
    print(f"  Test Accuracy: {acc * 100:.2f}%")
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
    cm = confusion_matrix(
        test_targets,
        test_preds,
        labels=[0, 1, 2]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("RNA-FM: Host Tropism Confusion Matrix")
    plt.xlabel("Predicted Host")
    plt.ylabel("Actual Host")
    plt.tight_layout()

    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    print(f"\nConfusion matrix saved to: {cm_plot_path}")
    print("Test evaluation completed successfully!")


if __name__ == "__main__":
    evaluate()