import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    def __init__(self, patched_time_step: int, embed_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim).double()
        self.gelu = nn.GELU()
        self.layer = nn.LayerNorm(embed_dim).double()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(embed_dim, embed_dim).double()
        self.dropout2 = nn.Dropout(p=dropout)
        self.proj = nn.Linear(embed_dim, patched_time_step).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = x
        x_mlp = self.dropout2(self.fc2(self.dropout1(self.gelu(self.fc1(x)))))
        out = self.proj(self.layer(x_mlp + res))
        return out.transpose(-1, -2)


if __name__ == '__main__':
    x = torch.randn(32, 4, 128)
    mlp = FeatureMLP(3, 128, 0.5)
    y = mlp(x)
    print(y.shape)
