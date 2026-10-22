import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# RNA-FM maximum position limit is 1024 tokens (1020 sequence characters + special tokens)
MAX_RNA_LEN = 1020

class H1N1RNADataset(Dataset):
    """
    PyTorch Dataset for H1N1 Segment 4 (HA) sequences.
    Converts DNA (T) to RNA (U) and ensures length fits within RNA-FM's 1024-token ceiling.
    """
    def __init__(self, csv_file, max_len=MAX_RNA_LEN):
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len
        
        if 'sequence' not in self.df.columns:
            raise ValueError(f"'sequence' column not found in {csv_file}")
            
        label_col = 'label' if 'label' in self.df.columns else 'host_label'
        
        self.sequences = self.df['sequence'].tolist()
        self.labels = self.df[label_col].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx]).strip().upper()
        
        # 1. Convert DNA (T) to RNA (U)
        rna_seq = seq.replace("T", "U")
        
        # 2. Slice to the first 1020 nt (covers the vital HA1 receptor binding domain)
        rna_seq = rna_seq[:self.max_len]
        
        label = int(self.labels[idx])
        return rna_seq, label


def create_dataloader(csv_file, alphabet, batch_size=4, shuffle=True):
    """
    Creates a PyTorch DataLoader that tokenizes sequences using RNA-FM's alphabet.
    """
    dataset = H1N1RNADataset(csv_file)
    batch_converter = alphabet.get_batch_converter()

    def collate_fn(batch):
        formatted_data = [(f"seq_{i}", seq) for i, (seq, _) in enumerate(batch)]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        
        _, _, tokens = batch_converter(formatted_data)
        return tokens, labels

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)