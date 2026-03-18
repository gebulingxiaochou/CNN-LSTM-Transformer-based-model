import torch.nn as nn


class Autoregressive(nn.Module):
    def __init__(self, time_step):
        super().__init__()
        self.linear = nn.Linear(time_step, 1).double()

    def forward(self, x):
        x = self.linear(x)
        return x
