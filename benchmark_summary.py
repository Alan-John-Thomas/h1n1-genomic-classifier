"""
Unified Benchmark Summary: DNABERT-2 vs. RNA-FM
Evaluates both trained foundation models on the identical 337-sequence test set
and produces the official comparative evaluation table for the MCA project report.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def evaluate_dnabert2(test_csv, checkpoint_path, device):
    from dnabert2.model import DNABERT2Classifier
    from dnabert2.dataset import create_dataloader

    print("Evaluating DNABERT-2...")
    model = DNABERT2Classifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loader = create_dataloader(test_csv, model.tokenizer, batch_size=8, shuffle=False)
    preds, targets = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(p)
            targets.extend(labels.numpy())

    return np.array(targets), np.array(preds)


def evaluate_rnafm(test_csv, checkpoint_path, device):
    # Add rnafm directory to path
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "rnafm"))
    from rnafm.model import RNAFMClassifier
    from rnafm.dataset import create_dataloader

    print("Evaluating RNA-FM...")
    model = RNAFMClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loader = create_dataloader(test_csv, model.alphabet, batch_size=8, shuffle=False)
    preds, targets = [], []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            logits = model(tokens)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(p)
            targets.extend(labels.numpy())

    return np.array(targets), np.array(preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Device: {device}")

    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    dnabert2_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "dnabert2_best.pt")
    rnafm_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "rnafm_best.pt")
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    results = []

    # 1. Evaluate DNABERT-2
    if os.path.exists(dnabert2_ckpt):
        d_targets, d_preds = evaluate_dnabert2(test_csv, dnabert2_ckpt, device)
        d_acc = accuracy_score(d_targets, d_preds)
        d_f1 = f1_score(d_targets, d_preds, average="macro")
        d_p = precision_score(d_targets, d_preds, average="macro")
        d_r = recall_score(d_targets, d_preds, average="macro")
        results.append({
            "Model": "DNABERT-2",
            "Pre-training Modality": "Double-stranded DNA (BPE)",
            "Parameters": "117M",
            "Hidden Dim": 768,
            "Test Accuracy": f"{d_acc*100:.2f}%",
            "Macro-F1": f"{d_f1:.4f}",
            "Macro-Precision": f"{d_p:.4f}",
            "Macro-Recall": f"{d_r:.4f}",
        })
    else:
        print(f"Note: DNABERT-2 checkpoint not found at {dnabert2_ckpt}. Run dnabert2/train.py first.")

    # 2. Evaluate RNA-FM
    if os.path.exists(rnafm_ckpt):
        r_targets, r_preds = evaluate_rnafm(test_csv, rnafm_ckpt, device)
        r_acc = accuracy_score(r_targets, r_preds)
        r_f1 = f1_score(r_targets, r_preds, average="macro")
        r_p = precision_score(r_targets, r_preds, average="macro")
        r_r = recall_score(r_targets, r_preds, average="macro")
        results.append({
            "Model": "RNA-FM",
            "Pre-training Modality": "Single-stranded RNA (Char)",
            "Parameters": "100M",
            "Hidden Dim": 640,
            "Test Accuracy": f"{r_acc*100:.2f}%",
            "Macro-F1": f"{r_f1:.4f}",
            "Macro-Precision": f"{r_p:.4f}",
            "Macro-Recall": f"{r_r:.4f}",
        })
    else:
        print(f"Note: RNA-FM checkpoint not found at {rnafm_ckpt}. Run rnafm/train.py first.")

    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("          OFFICIAL H1N1 BENCHMARK SUMMARY TABLE")
        print("=" * 80)
        print(df_res.to_string(index=False))
        print("=" * 80)

        # Save table to CSV for report
        summary_csv = os.path.join(logs_dir, "benchmark_summary.csv")
        df_res.to_csv(summary_csv, index=False)
        print(f"\nSummary table saved to: {summary_csv}")


if __name__ == "__main__":
    main()
