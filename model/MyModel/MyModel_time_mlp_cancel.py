import torch
import torch.nn as nn
from .components.data_patch import *
from .components.feature_attention import *
from .components.feature_mlp import *
from .components.time_attention import *
# from .components.time_mlp import *


class FeatureTimeEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, patch_time_step: int,
                 feature_dim: int, dropout: float):
        super().__init__()
        self.feature_attention = FeatureAttention(embed_dim, num_heads)
        self.feature_mlp = FeatureMLP(patch_time_step, embed_dim, dropout)
        self.time_attention = TimeAttention(embed_dim, num_heads)
        # self.time_mlp = TimeMLP(feature_dim, embed_dim, dropout)
        self.proj = nn.Linear(embed_dim, feature_dim).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1: torch.Tensor = x
        feature_out: torch.Tensor = self.feature_mlp(self.feature_attention(x))
        res2: torch.Tensor = feature_out + res1
        time_out: torch.Tensor = self.proj(self.time_attention(feature_out + res1))
        return time_out + res2


class MyModelTimeMLPCancel(nn.Module):
    def __init__(self, patch_size: int, num_block: int, time_step: int, embed_dim: int,
                 num_heads: int, patch_time_step: int, feature_dim: int, dropout: float):
        super().__init__()
        self.patch = DataPatch(patch_size)
        self.blocks = nn.ModuleList([
            FeatureTimeEncoder(embed_dim, num_heads, patch_time_step,
                               feature_dim, dropout) for _ in range(num_block)
        ])
        self.predictor_feature = nn.Sequential(
            nn.Linear(feature_dim, 1).double(),
            nn.ReLU()
        )
        self.predictor_time = nn.Sequential(
            nn.Linear(patch_time_step, 8).double(),
            nn.ReLU()
        )
        self.autoregressor = nn.Linear(time_step, 8).double()
        self.pred = nn.Sequential(
            nn.Linear(16, 64).double(),
            nn.ReLU(),
            nn.Linear(64, 128).double(),
            nn.ReLU(),
            nn.Linear(128, 128).double(),
            nn.ReLU(),
            nn.Linear(128, 64).double(),
            nn.ReLU(),
            nn.Linear(64, 16).double(),
            nn.ReLU(),
            nn.Linear(16, 8).double(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_patched: torch.Tensor = self.patch(x)
        block_out: torch.Tensor = torch.zeros_like(x_patched)
        for block in self.blocks:
            res: torch.Tensor = x_patched
            x_patched = block(x_patched) + res
            block_out += x_patched
        non_linear_out: torch.Tensor = self.predictor_time(
            self.predictor_feature(block_out).squeeze(-1)
        )

        linear_out: torch.Tensor = self.autoregressor(x[..., -1])

        x_mixed: torch.Tensor = torch.cat((linear_out, non_linear_out), dim=-1)

        return self.pred(x_mixed)


if __name__ == '__main__':
    x = torch.randn(32, 16, 4)
    model = MyModel(2, 4, 16,
                    128, 8, 15,
                    4, 0.5)
    y = model(x)
    print(y.shape)
