import torch
import torch.nn as nn
import fm

class RNAFMClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        print("Loading pre-trained RNA-FM foundation model...")
        # Load directly using the installed fm package
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        
        # Freeze the backbone (we only train the classification head)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Classification head: 640 embedding dimensions -> 3 host classes
        # 0: Human, 1: Swine, 2: Avian
        self.classifier = nn.Linear(640, num_classes)
        
    def forward(self, tokens):
        # Extract features through frozen RNA-FM backbone
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[12])
            token_embeddings = results["representations"][12] # [batch, seq_len, 640]
            
        # Mean Pooling: Average across all nucleotides
        sequence_embedding = token_embeddings.mean(dim=1)   # [batch, 640]
        
        # Pass through classification head
        logits = self.classifier(sequence_embedding)        # [batch, 3]
        return logits