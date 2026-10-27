"""
Benchmark Visualization Generator: DNABERT-2 vs. RNA-FM
Generates publication-quality figures comparing foundation model performance
for the MCA Semester 3 Mini Project presentation and final report.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Official Benchmark Metrics from README
OVERALL_METRICS = {
    "Metric": ["Accuracy (%)", "Macro-F1", "Macro-Precision", "Macro-Recall"],
    "DNABERT-2": [88.46, 0.8832 * 100, 0.8895 * 100, 0.8847 * 100],
    "RNA-FM":    [86.39, 0.8635 * 100, 0.8659 * 100, 0.8639 * 100],
}

PER_CLASS_F1 = {
    "Host": ["Human (Class 0)", "Swine (Class 1)", "Avian (Class 2)"],
    "DNABERT-2": [0.8738 * 100, 0.8413 * 100, 0.9421 * 100],
    "RNA-FM":    [0.8372 * 100, 0.7967 * 100, 0.9608 * 100],
}


def plot_overall_comparison(save_path):
    metrics = OVERALL_METRICS["Metric"]
    dna_scores = OVERALL_METRICS["DNABERT-2"]
    rna_scores = OVERALL_METRICS["RNA-FM"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, dna_scores, width, label="DNABERT-2 (dsDNA BPE)", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, rna_scores, width, label="RNA-FM (ssRNA Char)", color="#e07a5f", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Overall Foundation Model Benchmark Comparison\n(H1N1 Segment 4 Host Tropism)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(80, 95)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Value labels on bars
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


def plot_per_class_comparison(save_path):
    hosts = PER_CLASS_F1["Host"]
    dna_scores = PER_CLASS_F1["DNABERT-2"]
    rna_scores = PER_CLASS_F1["RNA-FM"]

    x = np.arange(len(hosts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, dna_scores, width, label="DNABERT-2", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, rna_scores, width, label="RNA-FM", color="#e07a5f", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("F1-Score (%)", fontweight="bold")
    ax.set_title("Per-Class Host Tropism F1-Score Breakdown\n(Human vs. Swine vs. Avian)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(hosts)
    ax.set_ylim(75, 100)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Highlight RNA-FM advantage on Avian
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
    print("=" * 65)
    print("Generating Benchmark Visualizations for Project Report...")
    print("=" * 65)

    p1 = os.path.join(LOGS_DIR, "benchmark_comparison.png")
    p2 = os.path.join(LOGS_DIR, "per_class_comparison.png")

    plot_overall_comparison(p1)
    plot_per_class_comparison(p2)

    print("\nVisualizations successfully generated in 'logs/' folder!")
    print("=" * 65)


if __name__ == "__main__":
    main()
