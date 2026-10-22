import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MAX_DNA_LEN = 512  # BPE compresses the ~1,724 bp HA gene into ~400-450 tokens

class H1N1DNADataset(Dataset):
    """
    PyTorch Dataset for H1N1 Segment 4 (HA) nucleotide sequences.
    Passes raw DNA directly to the DNABERT-2 BPE tokenizer.
    """
    def __init__(self, csv_file: str):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Dataset CSV not found at: {csv_file}")

        self.df = pd.read_csv(csv_file)
        if 'sequence' not in self.df.columns:
            raise ValueError(f"'sequence' column not found in {csv_file}")

        label_col = 'label' if 'label' in self.df.columns else 'host_label'
        self.sequences = self.df['sequence'].tolist()
        self.labels = self.df[label_col].tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = str(self.sequences[idx]).strip().upper()
        label = int(self.labels[idx])
        return seq, label


def create_dataloader(csv_file: str, tokenizer, batch_size: int = 8, shuffle: bool = True, max_len: int = MAX_DNA_LEN) -> DataLoader:
    """
    Creates a PyTorch DataLoader that tokenizes sequences using DNABERT-2's BPE tokenizer.
    Returns: input_ids, attention_mask, labels
    """
    dataset = H1N1DNADataset(csv_file)

    def collate_fn(batch):
        sequences = [seq for seq, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)

        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return encoded["input_ids"], encoded["attention_mask"], labels

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
