"""
Unified Benchmark Summary: DNABERT-2 vs. RNA-FM
Evaluates both trained foundation models on the identical 338-sequence test set,
produces the official comparative evaluation table, and automatically generates
publication-quality comparison bar charts in logs/.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def evaluate_dnabert2(test_csv, checkpoint_path, device):
    from dnabert2.model import DNABERT2Classifier
    from dnabert2.dataset import create_dataloader

    print("Evaluating DNABERT-2 on test set...")
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
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "rnafm"))
    from rnafm.model import RNAFMClassifier
    from rnafm.dataset import create_dataloader

    print("Evaluating RNA-FM on test set...")
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


def plot_overall_comparison(dna_scores, rna_scores, save_path):
    sns.set_theme(style="whitegrid", font="sans-serif")
    metrics = ["Accuracy (%)", "Macro-F1", "Macro-Precision", "Macro-Recall"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, dna_scores, width, label="DNABERT-2 (dsDNA BPE)", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, rna_scores, width, label="RNA-FM (ssRNA Char)", color="#e07a5f", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (%)", fontweight="bold", fontsize=12)
    ax.set_title("Overall Foundation Model Benchmark Comparison\n(H1N1 Segment 4 Host Tropism)", fontweight="bold", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(80, 95)
    ax.legend(frameon=True, loc="upper right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved overall comparison chart to: {save_path}")


def plot_per_class_comparison(dna_per_class, rna_per_class, save_path):
    sns.set_theme(style="whitegrid", font="sans-serif")
    hosts = ["Human (Class 0)", "Swine (Class 1)", "Avian (Class 2)"]

    x = np.arange(len(hosts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, dna_per_class, width, label="DNABERT-2", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, rna_per_class, width, label="RNA-FM", color="#e07a5f", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("F1-Score (%)", fontweight="bold", fontsize=12)
    ax.set_title("Per-Class Host Tropism F1-Score Breakdown\n(Human vs. Swine vs. Avian)", fontweight="bold", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(hosts, fontsize=11)
    ax.set_ylim(75, 100)
    ax.legend(frameon=True, loc="lower right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        star = " (Best)" if i == 2 else ""
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}%{star}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved per-class comparison chart to: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Device: {device}")

    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    dnabert2_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "dnabert2_best.pt")
    rnafm_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "rnafm_best.pt")
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    results = []
    d_overall = None
    r_overall = None
    d_class_f1 = None
    r_class_f1 = None

    # 1. Evaluate DNABERT-2
    if os.path.exists(dnabert2_ckpt):
        d_targets, d_preds = evaluate_dnabert2(test_csv, dnabert2_ckpt, device)
        d_acc = accuracy_score(d_targets, d_preds)
        d_f1 = f1_score(d_targets, d_preds, average="macro")
        d_p = precision_score(d_targets, d_preds, average="macro")
        d_r = recall_score(d_targets, d_preds, average="macro")
        d_class_f1 = f1_score(d_targets, d_preds, average=None) * 100.0
        d_overall = [d_acc * 100.0, d_f1 * 100.0, d_p * 100.0, d_r * 100.0]

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
        r_class_f1 = f1_score(r_targets, r_preds, average=None) * 100.0
        r_overall = [r_acc * 100.0, r_f1 * 100.0, r_p * 100.0, r_r * 100.0]

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

        # Save table to CSV
        summary_csv = os.path.join(logs_dir, "benchmark_summary.csv")
        df_res.to_csv(summary_csv, index=False)
        print(f"\n[Table] Summary table saved to: {summary_csv}")

        # Automatically generate publication-quality comparison plots
        if d_overall is not None and r_overall is not None:
            p1 = os.path.join(logs_dir, "benchmark_comparison.png")
            p2 = os.path.join(logs_dir, "per_class_comparison.png")
            plot_overall_comparison(d_overall, r_overall, p1)
            plot_per_class_comparison(d_class_f1, r_class_f1, p2)
            print("\n[Done] All benchmark comparison plots generated successfully in logs/!")


if __name__ == "__main__":
    main()
