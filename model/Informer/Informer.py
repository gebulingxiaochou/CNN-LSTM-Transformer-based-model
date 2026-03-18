import torch
import torch.nn as nn
from .components.data_embedding import *
from .components.encoder_decoder import *


class Informer(nn.Module):
    def __init__(self, c_in, d_model, n_head, seq_len, device):
        super().__init__()
        label_len = int(seq_len / 2)
        forcast_len = label_len
        self.data_embedding = DataEmbedding(c_in, d_model)
        self.encoder = Encoder(n_head, d_model, seq_len)
        self.decoder = Decoder(n_head, d_model, seq_len, label_len, forcast_len, device)
        self.cross_attention = nn.MultiheadAttention(d_model, n_head, batch_first=True).double()
        self.predictor = nn.Linear(int(seq_len / 2 * d_model), forcast_len).double()

    def forward(self, x):
        x = self.data_embedding(x)
        enc = self.encoder(x)
        dec = self.decoder(x)
        out, _ = self.cross_attention(dec, enc, enc)
        output = self.predictor(out.flatten(1))
        return output
