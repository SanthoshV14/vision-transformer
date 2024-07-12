import torch.nn as nn
from linear_projection import LinearProjection
from positional_embedding import PositionalEmbedding
from encoder import Encoder
from classifier import ClassificationHead

class VisionTransformer(nn.Module):
    def __init__(self, patch_size: int, n_patches: int, linear_in_features:int, num_class: int, n_heads: int, d_model=1024, n_x=6, dropout=0.2) -> None:
        super().__init__()
        self.linear_projection = LinearProjection(n_patches, patch_size, linear_in_features, d_model, dropout)
        self.position_embedding = PositionalEmbedding(n_patches, d_model, dropout)
        self.encoder = Encoder(n_x, n_patches, d_model, n_heads, dropout)
        self.classifier = ClassificationHead(num_class, d_model, dropout)

    def forward(self, x):
        x = self.linear_projection(x)
        x = self.position_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)

        return x