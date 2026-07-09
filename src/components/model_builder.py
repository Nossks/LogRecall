import torch.nn as nn
from transformers import AutoModel
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-small")
    
    def forward(self,io):
        outputs = self.model(**io)
        token_embeddings = outputs.last_hidden_state
        attention_mask = io['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        return nn.functional.normalize(pooled_output, p=2, dim=1)
