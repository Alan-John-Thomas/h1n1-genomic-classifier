"""
H1N1 Genomic Dataset Processing and PyTorch Dataset Module.
Parses raw NCBI Virus FASTA sequences, performs quality filtering,
generates balanced stratified splits (Human vs. Swine vs. Avian),
and provides PyTorch Dataset classes for both DNABERT-2 and RNA-FM.
"""

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Label mappings
LABEL_TO_ID = {"Human": 0, "Swine": 1, "Avian": 2}
ID_TO_LABEL = {0: "Human", 1: "Swine", 2: "Avian"}


def parse_and_clean_fasta(
    fasta_path: str,
    output_dir: str,
    samples_per_class: int = 750,
    seed: int = 42,
    min_length: int = 1600,
    max_length: int = 1850,
    max_n_ratio: float = 0.005,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parses raw NCBI FASTA, filters out ambiguous reads and duplicates,
    samples a balanced set across classes, and writes train/val/test CSVs.
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found at: {fasta_path}")

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print(f"Reading raw FASTA from: {fasta_path}")
    raw_records = []
    current_header = ""
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header and current_seq:
                    raw_records.append((current_header, "".join(current_seq).upper()))
                    current_seq = []
                current_header = line[1:]
            else:
                current_seq.append(line)
        if current_header and current_seq:
            raw_records.append((current_header, "".join(current_seq).upper()))

    print(f"Total raw records parsed: {len(raw_records)}")

    # Classify host from NCBI pipe-delimited header
    # Header format: >Accession |Definition|Host|Country
    human_hosts = {"Homo sapiens"}
    swine_keywords = ["sus scrofa", "swine", "pig"]

    # Step 1: Filter by length, quality, and track host annotations
    seq_to_metadata = defaultdict(list)
    seq_to_hosts = defaultdict(set)

    for header, seq in raw_records:
        seq_len = len(seq)
        if seq_len < min_length or seq_len > max_length:
            continue
        if seq.count("N") / seq_len > max_n_ratio:
            continue

        parts = [p.strip() for p in header.split("|")]
        accession = parts[0].split()[0] if len(parts) > 0 else "Unknown"
        definition = parts[1] if len(parts) > 1 else ""
        raw_host = parts[2] if len(parts) > 2 else "Unknown"
        country = parts[3] if len(parts) > 3 else "Unknown"

        # Determine label
        raw_host_lower = raw_host.lower()
        if raw_host in human_hosts:
            host_class = "Human"
        elif any(k in raw_host_lower for k in swine_keywords):
            host_class = "Swine"
        else:
            host_class = "Avian"

        seq_to_hosts[seq].add(host_class)
        seq_to_metadata[seq].append({
            "accession": accession,
            "host_class": host_class,
            "raw_host": raw_host,
            "country": country,
            "length": seq_len,
            "definition": definition,
        })

    # Step 2: Remove cross-species overlap (e.g., direct zoonotic transmission)
    clean_by_class = defaultdict(list)
    overlap_count = 0

    for seq, hosts in seq_to_hosts.items():
        if len(hosts) > 1:
            overlap_count += 1
            continue  # Exclude ambiguous multi-host sequences

        host_class = list(hosts)[0]
        meta = seq_to_metadata[seq][0]
        clean_by_class[host_class].append({
            "accession": meta["accession"],
            "host_class": host_class,
            "label": LABEL_TO_ID[host_class],
            "sequence": seq,
            "length": meta["length"],
            "raw_host": meta["raw_host"],
            "country": meta["country"],
        })

    print(f"Filtered out {overlap_count} sequences with cross-species host overlap.")
    for h in ["Human", "Swine", "Avian"]:
        print(f"  Clean unique sequences in {h}: {len(clean_by_class[h])}")

    # Step 3: Balanced sampling
    sampled_records = []
    for h in ["Human", "Swine", "Avian"]:
        available = clean_by_class[h]
        n_samples = min(samples_per_class, len(available))
        if n_samples < samples_per_class:
            print(f"Warning: Only {len(available)} available for {h} (requested {samples_per_class}). Using all.")
        sampled = random.sample(available, n_samples)
        sampled_records.extend(sampled)

    df_full = pd.DataFrame(sampled_records)
    print(f"\nTotal sampled balanced dataset: {len(df_full)} sequences")
    print(df_full["host_class"].value_counts().to_string())

    # Step 4: Stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        df_full,
        test_size=0.30,
        random_state=seed,
        stratify=df_full["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    # Save to disk
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    summary_path = os.path.join(output_dir, "dataset_summary.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total_samples": len(df_full),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "classes": LABEL_TO_ID,
        "mean_length": float(df_full["length"].mean()),
        "min_length": int(df_full["length"].min()),
        "max_length": int(df_full["length"].max()),
        "seed": seed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSplits successfully generated in {output_dir}:")
    print(f"  Train : {len(train_df)} ({train_path})")
    print(f"  Val   : {len(val_df)} ({val_path})")
    print(f"  Test  : {len(test_df)} ({test_path})")

    return train_df, val_df, test_df


class H1N1Dataset(Dataset):
    """
    PyTorch Dataset for H1N1 Segment 4 classification.
    Supports both DNABERT-2 (DNA alphabet) and RNA-FM (RNA alphabet).
    """
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        is_rna: bool = False,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset split not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_rna = is_rna

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).strip().upper()
        label = int(row["label"])

        # If evaluating RNA-FM, convert DNA Thymine (T) to RNA Uracil (U)
        if self.is_rna:
            seq = seq.replace("T", "U")

        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "accession": row["accession"],
        }
        return item


def get_dataloader(
    csv_path: str,
    tokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    max_length: int = 512,
    is_rna: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience helper to instantiate a DataLoader."""
    dataset = H1N1Dataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_rna=is_rna,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # When executed directly, process raw FASTA into train/val/test CSVs
    raw_fasta = os.path.join("data", "raw", "sequences.fasta")
    processed_dir = os.path.join("data", "processed")
    parse_and_clean_fasta(raw_fasta, processed_dir, samples_per_class=750)
