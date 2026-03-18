import torch
import torch.nn as nn
from .inception_block import *
import torch.nn.functional as F


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, pred_len=0, top_k=5, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_model * 2, num_kernels),
            nn.GELU(),
            InceptionBlock(d_model * 2, d_model, num_kernels)
        )

    def FFT_for_Period(self, x, k):
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = torch.mean(torch.abs(xf), dim=-1)
        frequency_list = torch.mean(frequency_list, dim=0)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.size(1) // top_list
        period_weight = torch.mean(torch.abs(xf), dim=-1)
        period_weight = period_weight[:, top_list]
        return period, period_weight

    def forward(self, x):
        B, T, N = x.shape
        period_list, period_weight = self.FFT_for_Period(x, self.top_k)
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            total_length = self.seq_len + self.pred_len
            if total_length % period != 0:
                length = ((total_length // period) + 1) * period
                padding_length = length - total_length
                last_values = x[:, -1:, :].repeat(1, padding_length, 1)
                out = torch.cat([x, last_values], dim=1)
            else:
                length = total_length
                out = x
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)
        period_weight = period_weight.repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, dim=-1)
        res = res + x
        return res
