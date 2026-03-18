import torch
import torch.nn as nn
from .components.data_patch import *
from .components.feature_attention import *
from .components.feature_mlp import *
from .components.time_attention import *
from .components.time_mlp import *


class FeatureTimeEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, patch_time_step: int,
                 feature_dim: int, dropout: float):
        super().__init__()
        self.feature_attention = FeatureAttention(embed_dim, num_heads)
        self.feature_mlp = FeatureMLP(patch_time_step, embed_dim, dropout)
        self.time_attention = TimeAttention(embed_dim, num_heads)
        self.time_mlp = TimeMLP(feature_dim, embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1: torch.Tensor = x
        feature_out: torch.Tensor = self.feature_mlp(self.feature_attention(x))
        res2: torch.Tensor = feature_out + res1
        time_out: torch.Tensor = self.time_mlp(self.time_attention(feature_out + res1))
        return time_out + res2


class MyModel(nn.Module):
    def __init__(self, patch_size: int, num_block: int, time_step: int, embed_dim: int,
                 num_heads: int, patch_time_step: int, feature_dim: int, dropout: float):
        super().__init__()
        self.patch = DataPatch(patch_size)
        self.blocks = nn.ModuleList([
            FeatureTimeEncoder(embed_dim, num_heads, patch_time_step,
                               feature_dim, dropout) for _ in range(num_block)
        ])
        self.lstm = nn.LSTM(feature_dim, embed_dim, batch_first=True).double()
        self.autoregressor = nn.Linear(time_step, 8).double()
        self.pred = nn.Linear(embed_dim, 8).double()
        self.gelu = nn.GELU()
        self.pred_out = nn.Linear(16, 8).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_patched: torch.Tensor = self.patch(x)
        block_out: torch.Tensor = torch.zeros_like(x_patched)
        for block in self.blocks:
            res: torch.Tensor = x_patched
            x_patched = block(x_patched) + res
            block_out += x_patched
        out, (_, _) = self.lstm(block_out)
        non_linear_out = self.pred(out[:, -1, :])
        linear_out: torch.Tensor = self.autoregressor(x[..., -1])

        return self.pred_out(self.gelu(torch.concat([non_linear_out, linear_out], dim=-1)))


if __name__ == '__main__':
    x = torch.randn(32, 16, 4)
    model = MyModel(2, 4, 16,
                    128, 8, 15,
                    4, 0.5)
    y = model(x)
    print(y.shape)
