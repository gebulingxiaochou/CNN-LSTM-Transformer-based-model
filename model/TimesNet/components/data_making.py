import torch.nn as nn
import torch
import math


class TokenEmbedding(nn.Module):
    def __init__(self, channel, d_model):
        super().__init__()
        self.token_maker = nn.Conv1d(
            in_channels=channel,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False
        ).double()
        nn.init.kaiming_normal_(
            self.token_maker.weight,
            mode="fan_in",
            nonlinearity="leaky_relu"
        )

    def forward(self, x):
        x = self.token_maker(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, channel, d_model):
        super().__init__()
        self.token_maker = TokenEmbedding(channel, d_model)
        self.position_embedding = PositionEmbedding(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        pos_enc = self.position_embedding(x)
        token_enc = self.token_maker(x)
        x = token_enc + pos_enc.expand_as(token_enc)
        return self.dropout(x)
