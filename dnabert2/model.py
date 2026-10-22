import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_NAME = "zhihan1996/DNABERT-2-117M"

class DNABERT2Classifier(nn.Module):
    """
    DNABERT-2 Sequence Classifier for H1N1 Segment 4 (HA) Host Tropism.
    Uses a frozen foundation model backbone and trains a lightweight
    linear classification head on mean-pooled token representations.
    """
    def __init__(self, num_classes: int = 3, model_name: str = MODEL_NAME):
        super().__init__()
        print(f"Loading pre-trained DNABERT-2 model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load configuration and model class dynamically to ensure perfect class alignment
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_cls = get_class_from_dynamic_module("bert_layers.BertModel", model_name)
        model_cls.config_class = config.__class__
        self.model = model_cls.from_pretrained(model_name, config=config)

        # Force native PyTorch attention fallback (robust, CUDA-accelerated, zero Triton dependency)
        for m in list(sys.modules.keys()):
            if "bert_layers" in m:
                sys.modules[m].flash_attn_qkvpacked_func = None

        # Freeze the foundation model backbone (we only train the classification head)
        for param in self.model.parameters():
            param.requires_grad = False

        # Classification head: 768 embedding dimensions -> num_classes (0: Human, 1: Swine, 2: Avian)
        self.classifier = nn.Linear(768, num_classes)

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Extracts 768-dim masked mean-pooled sequence embeddings from frozen DNABERT-2."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs[0]  # [batch, seq_len, 768]

        # Masked mean pooling across sequence tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sequence_embedding = sum_embeddings / sum_mask
        else:
            sequence_embedding = token_embeddings.mean(dim=1)

        return sequence_embedding  # [batch, 768]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        sequence_embedding = self.extract_features(input_ids, attention_mask)
        logits = self.classifier(sequence_embedding)  # [batch, num_classes]
        return logits
