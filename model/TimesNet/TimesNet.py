import torch
import torch.nn as nn
from .components.data_making import *
from .components.times_block import *


class TimesNet(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, d_model=64, top_k=5, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding = DataEmbedding(c_in, d_model)
        self.layers = nn.ModuleList([
            TimesBlock(seq_len, d_model, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, c_in).double()
        self.projection = nn.Linear(seq_len, pred_len).double()
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(d_model).double() for _ in range(num_layers)
        ])

    def forward(self, x):
        enc_out = self.embedding(x)
        for idx, layer in enumerate(self.layers):
            enc_out = layer(enc_out)
            enc_out = self.layer_norm[idx](enc_out)
        output = self.linear(enc_out)
        output = output.permute(0, 2, 1)
        output = self.projection(output)
        return output.squeeze_()
