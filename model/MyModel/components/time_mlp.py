import torch
import torch.nn as nn


class TimeMLP(nn.Module):
    def __init__(self, feature_num: int, embed_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim).double()
        self.gelu = nn.GELU()
        self.layer = nn.LayerNorm(embed_dim).double()
        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, feature_num).double()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = x
        x_mlp = self.dropout1(self.gelu(self.fc1(x)))

        return self.proj(self.layer(x_mlp + res))

if __name__ == '__main__':
    x = torch.randn(32, 3, 128)
    mlp = TimeMLP(4, 128, 0.5)
    y = mlp(x)
    print(y.shape)
