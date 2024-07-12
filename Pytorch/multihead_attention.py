import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_patches: int, d_model: int, n_heads: int, dropout: int) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model//n_heads
        self.layernorm = nn.LayerNorm((n_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual_x = x
        x = self.layernorm(x)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.reshape((q.shape[0], q.shape[1], self.n_heads, self.d_k)).transpose(1, 2)
        k = k.reshape((k.shape[0], k.shape[1], self.n_heads, self.d_k)).transpose(1, 2)
        v = v.reshape((v.shape[0], v.shape[1], self.n_heads, self.d_k)).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.d_k)
        attention_scores = self.softmax(attention_scores)
        x = torch.matmul(attention_scores, v)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], self.n_patches, self.d_model)
        x = self.w_o(x)
        x = residual_x + self.gelu(x)

        return self.dropout(x)
