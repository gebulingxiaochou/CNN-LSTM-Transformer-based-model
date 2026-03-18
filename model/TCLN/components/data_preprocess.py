import torch
import torch.nn as nn


class DataPreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.preprocess(x)

    def preprocess(self, feature):
        batch_size, time_length, num_feature = feature.shape[0], feature.shape[1], feature.shape[2]
        batch_store = []
        for idx_batch in range(batch_size):
            time_store = []
            for idx_time in range(time_length):
                map_data = feature[idx_batch, idx_time, :]
                feature_map = torch.zeros(num_feature, num_feature)
                for row in range(num_feature):
                    feature_map[row, :(num_feature - row)] = map_data[row:]
                time_store.append(feature_map)
            times_map = torch.stack(time_store, dim=0)
            batch_store.append(times_map)
        features = torch.stack(batch_store, dim=0)
        return features
