import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    # 经典Transformer的余弦编码
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pos_enc = PositionalEncoding(hidden_size, dropout=0.1)
        self.multi_atten = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True)

    def forward(self, x):
        x = self.pos_enc(x)
        output, _ = self.multi_atten(x, x, x)
        output = self.linear(output)
        output = self.layer_norm1(output)
        output += x
        copy = output.clone()
        output = self.dropout(self.relu(self.feedforward(output)))
        output = self.layer_norm2(output)
        output += copy
        output, _ = self.lstm(output)
        return self.layer_norm3(output) + x


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_features, num_layers):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_size)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                output = x[:, -1, :]
            else:
                output += x[:, -1, :]
        return output
