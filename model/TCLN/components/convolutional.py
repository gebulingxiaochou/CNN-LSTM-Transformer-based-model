import torch
import torch.nn as nn


class Multi_Kernel_CNN(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        # W = (H_in + 2×padding - kernel) / stride + 1
        W = int((num_feature + 2 - 1) / 1 + 1)
        self.conv_path1 = nn.Sequential(
            nn.Conv2d(1, 1, 1, 1, 1),
            nn.Conv2d(1, 1, (2, W), 1, 0)
        )
        self.conv_path2 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 2),
            nn.Conv2d(1, 1, (2, W), 1, 0)
        )
        self.conv_path3 = nn.Sequential(
            nn.Conv2d(1, 1, 5, 1, 3),
            nn.Conv2d(1, 1, (2, W), 1, 0)
        )

    def forward(self, x):
        x1 = self.conv_path1(x)
        x2 = self.conv_path2(x)
        x3 = self.conv_path3(x)
        return x1 + x2 + x3


class ConvolutionalComponent(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.multi_kernel = Multi_Kernel_CNN(num_feature)

    def forward(self, x):
        x.unsqueeze_(2)
        time_length = x.shape[1]
        time_step = []
        for t in range(time_length):
            map = x[:, t, :, :, :]
            map = self.multi_kernel(map)
            time_step.append(map)
        output = torch.stack(time_step, dim=1)
        output.squeeze_()
        return output
