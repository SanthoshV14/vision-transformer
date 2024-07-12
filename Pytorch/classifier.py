import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, num_class: int, d_model: int, dropout: int) -> None:
        super().__init__()
        self.mlp1 = nn.Linear(d_model, 4*d_model)
        self.mlp2 = nn.Linear(4*d_model, num_class)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x[:, 0, :]
        x = self.mlp1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x