"""
Live Host Tropism Prediction CLI for H1N1 Segment 4 (HA) Sequences.
Allows testing single raw nucleotide sequences, FASTA files, or test samples.
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
    parser = argparse.ArgumentParser(description="Predict H1N1 Host Tropism Phenotype (Human vs. Swine vs. Avian)")
    parser.add_argument("--seq", type=str, default=None, help="Raw nucleotide sequence (A, C, G, T)")
    parser.add_argument("--fasta", type=str, default=None, help="Path to input FASTA file")
    parser.add_argument("--sample", action="store_true", help="Randomly sample a sequence from test.csv")
    parser.add_argument("--model", type=str, default="rnafm", choices=["rnafm", "dnabert2"], help="Model backbone to use")
    return parser.parse_args()

def load_sample_from_test():
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv}")
    df = pd.read_csv(test_csv)
    row = df.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]
    return row["sequence"], int(row["label"]), row.get("accession", "Unknown Accession")

def predict_rnafm(seq: str, device):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "rnafm"))
    from rnafm.model import RNAFMClassifier

    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "rnafm_best.pt")
    if not os.path.exists(ckpt_path):
        # Also check .pth extension
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "rnafm_best.pth")

    model = RNAFMClassifier(num_classes=3).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded trained checkpoint: {ckpt_path}")
    else:
        print("[Notice] Using foundation model representations (training checkpoint not found locally)")

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

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_label = None
    accession = None

    if args.sample:
        raw_seq, true_label, accession = load_sample_from_test()
        print(f"[Sampling] Selected sample from test.csv: Accession {accession}")
    elif args.fasta:
        if not os.path.exists(args.fasta):
            print(f"Error: FASTA file not found at {args.fasta}")
            return
        lines = open(args.fasta).read().splitlines()
        raw_seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    elif args.seq:
        raw_seq = args.seq.strip()
    else:
        print("No input provided. Defaulting to randomly sampling a sequence from test.csv...")
        raw_seq, true_label, accession = load_sample_from_test()

    pred_class, probs = predict_rnafm(raw_seq, device)
    pred_host = CLASS_MAP[pred_class]

    print("\n" + "=" * 65)
    print("        H1N1 VIRUS HOST TROPISM PREDICTION RESULT")
    print("=" * 65)
    if accession:
        print(f"  Accession ID    : {accession}")
    print(f"  Sequence Length : {len(raw_seq)} bp")
    print(f"  Model Used      : {args.model.upper()}")
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
    print("=" * 65)

if __name__ == "__main__":
    main()
