# H1N1 Genomic Classifier: Benchmarking Genomic Foundation Models (DNABERT-2 vs. RNA-FM)

## Project Overview

This repository provides an empirical benchmarking pipeline comparing DNA and RNA genomic foundation models on host tropism classification (Human vs. Swine vs. Avian) using Influenza A virus (H1N1) Segment 4 (Hemagglutinin / HA) nucleotide sequences.

### Objectives
1. Comparative evaluation of foundation model pre-training modalities:
   * DNABERT-2 (zhihan1996/DNABERT-2-117M): 117M parameters, pre-trained on multi-species double-stranded DNA genomes using Byte-Pair Encoding (BPE).
   * RNA-FM (cuhkaih/rnafm): 100M parameters, pre-trained on 23+ million non-coding single-stranded RNA sequences using single-character tokenization.
2. Target Genomic Region:
   * Segment 4 (Hemagglutinin / HA, full-length coding sequence ~1,700 to 1,778 bp): Primary surface glycoprotein responsible for host-cell attachment and viral entry via sialic acid receptor binding.
3. Quality-Controlled Balanced Dataset:
   * 2,250 non-redundant, full-length sequences (750 Human, 750 Swine, 750 Avian) sourced from the NCBI Virus repository, with ambiguous reads filtered.

## Benchmark Results

Evaluated on 338 held-out test sequences (106 Human, 120 Swine, 124 Avian) under frozen-backbone feature extraction:

| Model | Pre-training Modality | Parameters | Hidden Dim | Test Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DNABERT-2 | Double-stranded DNA (BPE) | 117M | 768 | 88.46% | 0.8832 | 0.8895 | 0.8847 |
| RNA-FM | Single-stranded RNA (Char) | 100M | 640 | 86.39% | 0.8635 | 0.8659 | 0.8639 |

### Per-Class Test F1-Scores

| Model | Human (Class 0) | Swine (Class 1) | Avian (Class 2) |
| :--- | :--- | :--- | :--- |
| DNABERT-2 | 0.8738 | 0.8413 | 0.9421 |
| RNA-FM | 0.8372 | 0.7967 | 0.9608 |

## Repository Structure

```text
h1n1-genomic-classifier/
├── data/
│   ├── raw/                  # Raw FASTA sequences (NCBI Virus, local)
│   └── processed/            # Stratified train, val, test splits (train.csv, val.csv, test.csv)
├── dnabert2/                 # Track A: DNABERT-2 foundation model pipeline
│   ├── dataset.py            # Tokenizer wrapper and PyTorch Dataset
│   ├── model.py              # Sequence classifier with native attention fallback
│   ├── train.py              # Training loop with embedding caching
│   └── evaluate.py           # Test set evaluation and confusion matrix generator
├── rnafm/                    # Track B: RNA-FM foundation model pipeline
│   ├── dataset.py            # RNA sequence converter (T->U) and PyTorch Dataset
│   ├── model.py              # Sequence classifier using RNA-FM 640-dim representations
│   ├── train.py              # Training loop with embedding caching
│   └── evaluate.py           # Test set evaluation and confusion matrix generator
├── src/
│   ├── __init__.py
│   └── dataset.py            # BioPython FASTA parser and curation pipeline
├── checkpoints/              # Best trained classification head weights (local)
├── logs/                     # Confusion matrices, benchmark tables, and comparison charts
├── benchmark_summary.py      # Unified evaluation script across both models
├── environment.yml           # Conda environment definition
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Alan-John-Thomas/h1n1-genomic-classifier.git
cd h1n1-genomic-classifier
```

### 2. Environment Setup
```bash
conda env create -f environment.yml
conda activate h1n1-env
```

Install pinned PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
pip install -r requirements.txt
```

### 3. Verify CUDA Availability
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## Pipeline Execution

### Step 1: Preprocess Dataset
Curate raw FASTA sequences into balanced splits:
```bash
python src/dataset.py
```

### Step 2: Train Foundation Model Heads
Train frozen-backbone classification heads on GPU:
```bash
# Track A: DNABERT-2
python dnabert2/train.py

# Track B: RNA-FM
python rnafm/train.py
```

### Step 3: Individual Model Evaluation
```bash
# Evaluate DNABERT-2 on test set
python dnabert2/evaluate.py

# Evaluate RNA-FM on test set
python rnafm/evaluate.py
```

### Step 4: Run Unified Comparative Benchmark
Evaluate both models on the identical test split and output the comparison table and plot:
```bash
python benchmark_summary.py
```
