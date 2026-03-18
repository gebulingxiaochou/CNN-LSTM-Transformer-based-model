import torch
import torch.nn as nn
from .rope_position import *

class TimeAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.device = torch.device("cuda")
        self.embed_dim: int = embed_dim
        self.layer1 = nn.LayerNorm(embed_dim).double()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        ).double()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU()
        )
        self.layer2 = nn.LayerNorm(embed_dim).double()

    def _project_to(self, x: torch.Tensor) -> torch.Tensor:
        self.F = x.shape[-1]
        W: torch.Tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(x.shape[-1], self.embed_dim))
        ).double().to(self.device)
        b: torch.Tensor = nn.Parameter(
            nn.init.zeros_(torch.zeros(self.embed_dim))
        ).double().to(self.device)
        return torch.matmul(x, W) + b

    # def _project_back(self, x: torch.Tensor) -> torch.Tensor:
    #     W: torch.Tensor = nn.Parameter(
    #         nn.init.xavier_uniform_(torch.empty(self.embed_dim, self.F))
    #     )
    #     b: torch.Tensor = nn.Parameter(
    #         nn.init.zeros_(torch.zeros(self.F))
    #     )
    #     return torch.matmul(x, W) + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rope = RotaryPositionEmbedding(dim=self.embed_dim)
        x_proj: torch.Tensor = rope(self._project_to(x))
        res1: torch.Tensor = x_proj
        x_attention: torch.Tensor = self.attention(x_proj, x_proj, x_proj)[0]
        output: torch.Tensor = self.layer1(x_attention + res1)
        res2: torch.Tensor = output
        return self.layer2(self.feed_forward(output) + res2)


if __name__ == '__main__':
    x = torch.randn(32, 3, 4)
    model = TimeAttention(embed_dim=128, num_heads=8)
    output = model(x)
    print(output.size())
