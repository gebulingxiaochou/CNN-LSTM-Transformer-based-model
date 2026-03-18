import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.device = torch.device("cuda")
        self.embed_dim: int = embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        ).double()
        self.layer_norm1 = nn.LayerNorm(embed_dim).double()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU()
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim).double()

    def _project_to(self, x: torch.Tensor) -> torch.Tensor:
        # self.T: int = x.shape[1]
        x_feature_map: torch.Tensor = x.transpose(-1, -2)
        W: torch.Tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(x_feature_map.shape[-1], self.embed_dim))
        ).double().to(self.device)
        b: torch.Tensor = nn.Parameter(
            nn.init.zeros_(torch.zeros(self.embed_dim))
        ).double().to(self.device)
        return torch.matmul(x_feature_map, W) + b

    # def _project_back(self, x: torch.Tensor) -> torch.Tensor:
    #     W: torch.Tensor = nn.Parameter(
    #         nn.init.xavier_uniform_(torch.empty(self.embed_dim, self.T))
    #     )
    #     b: torch.Tensor = nn.Parameter(
    #         nn.init.zeros_(torch.zeros(self.T))
    #     )
    #     return (torch.matmul(x, W) + b).transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_project_to: torch.Tensor = self._project_to(x)
        res1: torch.Tensor = x_project_to
        x_attention: torch.Tensor = self.attention(
            x_project_to, x_project_to, x_project_to
        )[0]
        output: torch.Tensor = self.layer_norm1(x_attention + res1)
        res2: torch.Tensor = output
        return self.layer_norm2(self.feed_forward(output) + res2)


if __name__ == '__main__':
    x = torch.randn(32, 3, 4)
    model = FeatureAttention(embed_dim=128, num_heads=8)
    y = model(x)
    print(y.shape)
