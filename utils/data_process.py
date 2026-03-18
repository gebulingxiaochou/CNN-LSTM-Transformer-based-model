import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    def __init__(self, data, win_size, target_size):
        self.target_size = target_size
        self.data = data
        self.winsize = win_size

    def __len__(self):
        return self.data.shape[0] - self.winsize + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.winsize]
        feature = torch.tensor(window[:int(self.winsize - self.target_size)], dtype=torch.float64)
        target = torch.tensor(window[int(self.winsize - self.target_size):, -1], dtype=torch.float64)
        return feature, target


def train_loader(win_size, batch_size, data_path, start_idx, num_feature, target_size):
    train_data = pd.read_csv(data_path).values
    train_data = train_data[:, start_idx:int(num_feature + start_idx)].astype(np.float64)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_dataset = MyDataset(train_data, win_size=win_size, target_size=target_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader

if __name__ == '__main__':
    train = train_loader(
        16, 32, "D:\Code_Working\Python\GitRepository\MixModel\data\HSB_train.csv",
        1, 4, 8
    )

    for feature, target in train:
        print(feature.shape, target.shape)