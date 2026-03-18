import torch
import torch.nn as nn
from .components.convolutional import *
from .components.data_preprocess import *
from .components.encoder import *
from .components.autoregressive import *


class TCLN(nn.Module):
    def __init__(self, c_in, hidden_size, num_layers, time_step, device):
        super().__init__()
        self.device = device
        self.preprocess = DataPreprocess()
        self.conv = ConvolutionalComponent(c_in)
        num_features = int((c_in + 2 - 1) / 1)
        self.encoder = Encoder(hidden_size, num_features, num_layers)
        self.auto = Autoregressive(time_step)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 8).double()

    def forward(self, x):
        x1 = x[:, :, -1]
        x2 = self.preprocess(x)
        x2 = x2.to(self.device)
        output1 = self.auto(x1)
        output2 = self.encoder(self.conv(x2))
        output = self.fc(self.relu(output1.expand_as(output2) + output2))
        return output
