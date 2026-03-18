import torch.nn as nn
import torch
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        ).double()
        nn.init.kaiming_normal_(
            self.tokenConv.weight,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = TokenEmbedding(c_in, d_model)
        self.positionEmbedding = PositionEmbedding(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.tokenConv(x) + self.positionEmbedding(x))

if __name__ == '__main__':
    model = DataEmbedding(1, 128)
    data = torch.rand(32, 14, 1)
    output = model(data)
    print(output.size())