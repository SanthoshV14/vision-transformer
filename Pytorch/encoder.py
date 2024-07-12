import torch.nn as nn
from multihead_attention import MultiHeadAttention
from mlp_layer import MLPLayer

class EncoderBlock(nn.Module):
    def __init__(self, mha: MultiHeadAttention, mlp: MLPLayer, dropout: int) -> None:
        super().__init__()
        self.mha = mha
        self.mlp = mlp
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mha(x)
        x = self.mlp(x)

        return self.dropout(x)
    
class Encoder(nn.Module):
    def __init__(self, n_x: int, seq_len: int, d_model: int, n_heads: int, dropout: int) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        for i in range(n_x):
            mha = MultiHeadAttention(seq_len, d_model, n_heads, dropout)
            mlp = MLPLayer(seq_len, d_model, dropout)
            encoder = EncoderBlock(mha, mlp, dropout)
            self.encoder_blocks.append(encoder)

    def forward(self, x):
        for encoder in self.encoder_blocks:
            x = encoder(x)
        
        return x