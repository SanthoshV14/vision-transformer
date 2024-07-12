import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches: int, d_model: int, dropout: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn((1, 1, d_model)))
        self.postion_embedding = nn.Parameter(torch.randn((1, n_patches, d_model)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        cls_token = self.cls_token.expand((x.shape[0], -1, -1))
        x = torch.cat([cls_token, x], dim=1)
        x = self.postion_embedding + x
        return self.dropout(x)