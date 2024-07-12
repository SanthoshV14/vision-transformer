import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self, n_patches: int, patch_size: int, in_features: int, out_features: int, dropout: int) -> None:
        super().__init__()
        self.n_patches = n_patches - 1
        self.patch_size = patch_size
        self.linear_projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def patch_image(self, x):
        x = x.flatten(start_dim=1, end_dim=2)
        x = x.reshape((x.shape[0], self.n_patches, int(self.patch_size**2), 3))
        x = x.flatten(start_dim=-2, end_dim=-1)
        return x

    def forward(self, x):
        x = self.patch_image(x)
        x = self.linear_projection(x)
        return self.dropout(x)