import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
from mlp import MultiLayerPerceptron



class Decoder(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4, ff_dim: int = 128, num_layers: int = 4, dropout: float = 0.1, max_seq_len: int = 100, apply_mask: bool = True, input_dim: int = 512):
        super().__init__()
        self.apply_mask = apply_mask
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.projection = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim))
        self.layers = nn.Sequential(*[DecoderLayer(embedding_dim, num_heads, ff_dim, dropout, apply_mask) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.layers(x)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, apply_mask: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.layers = nn.Sequential(*[
            Attention(embedding_dim, num_heads, dropout, apply_mask=apply_mask),
            MultiLayerPerceptron(embedding_dim, ff_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x