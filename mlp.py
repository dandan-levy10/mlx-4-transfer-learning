import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model=64, dff=256, dropout=0.1):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.linear(x)
        return out  # No residual/norm here