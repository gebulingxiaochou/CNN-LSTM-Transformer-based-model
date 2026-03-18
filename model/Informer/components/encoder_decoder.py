import torch
import torch.nn as nn
import math
import numpy as np


class MultiProbSparseAttention(nn.Module):
    def __init__(self, n_head, d_model, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = n_head
        self.d_model = d_model
        self.proj_q = nn.Linear(d_model, d_model).double()
        self.proj_k = nn.Linear(d_model, d_model).double()
        self.proj_v = nn.Linear(d_model, d_model).double()

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        Q = self.proj_q(q).reshape(q.shape[0], q.shape[1], self.num_heads, -1)
        K = self.proj_k(k).reshape(k.shape[0], k.shape[1], -1, self.num_heads)
        V = self.proj_v(v).reshape(v.shape[0], v.shape[1], self.num_heads, -1)

        select_num_k = self.seq_len // 3
        select_idx_k = torch.randint(0, self.seq_len, (select_num_k,))
        select_k = K[:, select_idx_k, :, :]

        score_matrix = torch.einsum('bijk,bkjt->bijt', Q, select_k.permute(0, 2, 3, 1))
        score_matrix = score_matrix.transpose(-1, -2).mean(-1)
        max_score, _ = torch.max(score_matrix, dim=-1)
        max_score = max_score / math.sqrt(self.d_model)
        mean_score = score_matrix.mean(-1) / math.sqrt(self.d_model)
        point_matrix = max_score - mean_score
        _, select_idx_q = torch.topk(point_matrix, select_num_k, dim=-1)
        select_Q_list = []
        select_V_list = []
        for batch_idx in range(batch_size):
            batch_select_q = Q[batch_idx, select_idx_q[batch_idx, :], :, :]
            batch_select_v = V[batch_idx, select_idx_q[batch_idx, :], :, :]
            select_Q_list.append(batch_select_q)
            select_V_list.append(batch_select_v)
        select_q = torch.stack(select_Q_list, dim=0)
        select_v = torch.stack(select_V_list, dim=0)

        attention_weight = torch.matmul(select_q, select_k) / math.sqrt(self.d_model)
        attention = torch.matmul(torch.softmax(attention_weight, dim=-1), select_v)
        attention = attention.reshape(batch_size, select_num_k, -1)

        mean_v = attention.mean(dim=1)
        v_new = V.reshape(batch_size, self.seq_len, -1)
        for batch_idx in range(batch_size):
            v_new[batch_idx, select_idx_q[batch_idx, :], :] = attention[batch_idx, :, :]
            mask = np.ones(self.seq_len, dtype=bool)
            choose = select_idx_q[batch_idx, :].detach().cpu().numpy().astype(int)
            mask[choose] = False
            mean_matrix = v_new[batch_idx, mask, :]
            v_new[batch_idx, mask, :] = mean_v[batch_idx, :].squeeze(0).expand_as(mean_matrix)
        return v_new


class Encoder(nn.Module):
    def __init__(self, n_head, d_model, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.distill_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3),
            nn.GELU(),
            nn.MaxPool1d(4, 2, 2)
        ).double()
        self.distill = nn.ModuleList([
            self.distill_block for _ in range(3)
        ])
        self.attention = nn.ModuleList([
            MultiProbSparseAttention(n_head, d_model, seq_len),
            MultiProbSparseAttention(n_head, d_model, int(seq_len / 2)),
            MultiProbSparseAttention(n_head, d_model, int(seq_len / 2)),
        ])

    def forward(self, x):
        output1 = self.distill[0](self.attention[0](x, x, x).transpose(-1, -2)).transpose(-1, -2)
        output1 = self.distill[1](self.attention[1](output1, output1, output1).transpose(-1, -2)).transpose(-1, -2)
        x_new = x[:, int(self.seq_len / 2):, :]
        output2 = self.distill[2](self.attention[2](x_new, x_new, x_new).transpose(-1, -2)).transpose(-1, -2)
        output = torch.cat([output1, output2], dim=1)
        return output


class Decoder(nn.Module):
    def __init__(self, n_head, d_model, seq_len, label_len, forcast_len, device):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.d_model = d_model
        self.label_len = label_len
        self.forcast_len = forcast_len
        self.attention = MultiProbSparseAttention(n_head, d_model, label_len + forcast_len)
        self.distill_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3).double(),
            nn.GELU(),
            nn.MaxPool1d(4, 2, 2)
        )

    def forward(self, x):
        forcast = torch.zeros(x.shape[0], self.forcast_len, self.d_model)
        label = x[:, int(self.seq_len - self.label_len):, :]
        forcast, label = forcast.to(self.device), label.to(self.device)
        x = torch.cat([forcast, label], dim=1)
        output = self.distill_block(self.attention(x, x, x).transpose(-1, -2)).transpose(-1, -2)
        return output
