# H1N1 Genomic Classifier: Benchmarking Genomic Foundation Models (DNABERT-2 vs. RNA-FM)

## Project Overview

This repository implements a computational biology pipeline to classify host tropism phenotypes (Human vs. Swine vs. Avian) from Influenza A (H1N1) viral nucleotide sequences.

### Objectives
1. Benchmark single-stranded vs. double-stranded genomic foundation models:
   - DNABERT-2 (zhihan1996/DNABERT-2-117M): Pre-trained on multi-species double-stranded DNA genomes using BPE tokenization.
   - RNA-FM (qibinc/RNA-FM): Pre-trained on 23+ million non-coding single-stranded RNA sequences.
2. Target Genomic Region:
   - Segment 4 (Hemagglutinin / HA, ~1,700 to 1,778 bp): Primary glycoprotein determining viral entry and host-range restriction via sialic acid linkage specificity.
3. Curated NCBI Virus Dataset:
   - Stratified dataset sampled from NCBI Virus repository, filtered for length completeness and zero ambiguous nucleotides.

## Repository Structure

```text
h1n1-genomic-classifier/
├── data/
│   ├── raw/                  # Raw FASTA sequences from NCBI Virus (local)
│   └── processed/            # Stratified train, val, and test splits (CSV)
├── src/
│   ├── __init__.py
│   ├── dataset.py            # BioPython FASTA parsing, QC, and PyTorch Dataset
│   ├── models.py             # Model factory for DNABERT-2 and RNA-FM classifiers
│   └── utils.py              # Metric calculation and plotting utilities
├── checkpoints/              # Saved model checkpoints (local)
├── logs/                     # Experiment logs and training metrics (local)
├── train.py                  # Training pipeline with frozen-backbone feature extraction
├── evaluate.py               # Evaluation script across accuracy, macro-F1, and confusion matrix
├── environment.yml           # Conda environment definition
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <REPO-URL>
cd h1n1-genomic-classifier
```

### 2. Environment Setup
Create the environment using the provided specification file:
```bash
conda env create -f environment.yml
conda activate h1n1-env
```

If running on a machine with an NVIDIA GPU, verify PyTorch CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 3. Verify Setup
Run a basic check to verify CUDA detection and library imports:
```bash
python -c "import torch, transformers, Bio; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

## Pipeline Execution

### Step 1: Preprocess Dataset
Parse the raw FASTA file, remove duplicates and ambiguous reads, and generate balanced train/val/test splits:
```bash
python src/dataset.py
```

### Step 2: Model Training
Train a sequence classification head on top of the frozen foundation model:
```bash
# Train with DNABERT-2 backbone
python train.py --model dnabert2 --batch_size 8 --epochs 10

# Train with RNA-FM backbone
python train.py --model rnafm --batch_size 8 --epochs 10
```

### Step 3: Evaluation
Compute validation and test metrics (Accuracy, Macro-F1, Confusion Matrix):
```bash
python evaluate.py --model dnabert2
python evaluate.py --model rnafm
```
