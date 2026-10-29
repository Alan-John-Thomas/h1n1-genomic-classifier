"""
Live Host Tropism Prediction CLI for H1N1 Segment 4 (HA) Sequences.
Supports single raw nucleotide sequences, FASTA files, or test set samples
using either DNABERT-2 (117M) or RNA-FM (100M) foundation model backbones.
"""

import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

CLASS_MAP = {0: "Human", 1: "Swine", 2: "Avian"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict H1N1 Host Tropism Phenotype (Human vs. Swine vs. Avian)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rnafm",
        choices=["rnafm", "dnabert2"],
        help="Foundation model backbone to use for inference (default: rnafm)",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="Raw nucleotide sequence string (A, C, G, T/U)",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Randomly sample a sequence from data/processed/test.csv",
    )
    return parser.parse_args()


def load_sample_from_test():
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv}")
    df = pd.read_csv(test_csv)
    row = df.sample(n=1, random_state=random.randint(1, 100000)).iloc[0]
    return str(row["sequence"]), int(row["label"]), str(row.get("accession", "Unknown Accession"))


def load_fasta(fasta_path: str):
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found at: {fasta_path}")
    lines = open(fasta_path, "r", encoding="utf-8", errors="replace").read().splitlines()
    seq_lines = [l.strip() for l in lines if not l.startswith(">") and l.strip()]
    if not seq_lines:
        raise ValueError(f"No valid nucleotide sequence found in: {fasta_path}")
    header = lines[0][1:].strip() if lines and lines[0].startswith(">") else os.path.basename(fasta_path)
    return "".join(seq_lines), header


def predict_rnafm(seq: str, device):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "rnafm"))
    from rnafm.model import RNAFMClassifier

    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "rnafm_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "rnafm_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Trained RNA-FM checkpoint not found at '{ckpt_path}'. "
            "Please run 'python rnafm/train.py' first."
        )

    model = RNAFMClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded trained checkpoint: {ckpt_path}")
    model.eval()

    # Preprocess sequence: T -> U and max 1020 tokens
    rna_seq = seq.strip().upper().replace("T", "U")[:1020]
    batch_converter = model.alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("input_seq", rna_seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        logits = model(tokens)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_class = int(probs.argmax())
    return pred_class, probs


def predict_dnabert2(seq: str, device):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dnabert2"))
    from dnabert2.model import DNABERT2Classifier

    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "dnabert2_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "dnabert2_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Trained DNABERT-2 checkpoint not found at '{ckpt_path}'. "
            "Please run 'python dnabert2/train.py' first."
        )

    model = DNABERT2Classifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded trained checkpoint: {ckpt_path}")
    model.eval()

    # DNA format: ensure U -> T, upper
    dna_seq = seq.strip().upper().replace("U", "T")
    encoding = model.tokenizer(
        dna_seq,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_class = int(probs.argmax())
    return pred_class, probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_label = None
    accession = None

    if args.sample:
        raw_seq, true_label, accession = load_sample_from_test()
        print(f"[Sampling] Selected sample from test.csv: Accession {accession}")
    elif args.fasta:
        raw_seq, accession = load_fasta(args.fasta)
    elif args.seq:
        raw_seq = args.seq.strip()
    else:
        print("No input provided. Defaulting to randomly sampling a sequence from test.csv...")
        raw_seq, true_label, accession = load_sample_from_test()

    # Route prediction to appropriate foundation model
    if args.model == "dnabert2":
        pred_class, probs = predict_dnabert2(raw_seq, device)
        model_display = "DNABERT-2 (117M dsDNA BPE)"
    else:
        pred_class, probs = predict_rnafm(raw_seq, device)
        model_display = "RNA-FM (100M ssRNA Char)"

    pred_host = CLASS_MAP[pred_class]

    print("\n" + "=" * 65)
    print("        H1N1 VIRUS HOST TROPISM PREDICTION RESULT")
    print("=" * 65)
    if accession:
        print(f"  Accession ID    : {accession}")
    print(f"  Sequence Length : {len(raw_seq)} bp")
    print(f"  Model Used      : {model_display}")
    print(f"  Device          : {device}")
    print("-" * 65)
    print(f"  PREDICTED HOST  : {pred_host.upper()} (Confidence: {probs[pred_class]*100:.2f}%)")
    if true_label is not None:
        true_host = CLASS_MAP[true_label]
        match_str = "CORRECT" if pred_host == true_host else "MISMATCH"
        print(f"  ACTUAL HOST     : {true_host.upper()} [{match_str}]")
    print("-" * 65)
    print("  Host Probability Breakdown:")
    for cid, cname in CLASS_MAP.items():
        bar_len = int(probs[cid] * 30)
        bar_viz = "#" * bar_len + "-" * (30 - bar_len)
        print(f"    - {cname:<6}: [{bar_viz}] {probs[cid]*100:6.2f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
