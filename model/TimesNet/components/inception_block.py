import torch.nn as nn
import torch


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        kernels = []
        for i in range(num_kernels):
            kernel_size = 2 * i + 1
            padding = (kernel_size - 1) // 2
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding).double())
        self.kernels = nn.ModuleList(kernels)
        self.activation = nn.GELU()

    def forward(self, x):
        res_list = []
        for kernel in self.kernels:
            res_list.append(kernel(x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return self.activation(res)
