import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: int) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm((seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual_x = x
        x = self.layernorm(x)
        x = self.mlp(x)
        x = residual_x + self.gelu(x)

        return self.dropout(x)