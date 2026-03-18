import torch
import torch.nn as nn


class DataPatch(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size: int = patch_size
        self.mixer = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(patch_size, 1),
            stride=1,
            padding=0,
        ).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B: int = x.shape[0]
        T: int = x.shape[1]
        batch_list: list[torch.Tensor] = []

        for batch_idx in range(B):
            time_list: list[torch.Tensor] = []
            for time_idx in range(T - self.patch_size + 1):
                tiny_patch_map: torch.Tensor = x[batch_idx, time_idx:time_idx + self.patch_size, :]
                tiny_patch_vector: torch.Tensor = self.mixer(tiny_patch_map.unsqueeze(0).unsqueeze(0))
                time_list.append(tiny_patch_vector.squeeze_())
            huge_patch_map: torch.Tensor = torch.stack(time_list, dim=0)
            batch_list.append(huge_patch_map)
        x_new: torch.Tensor = torch.stack(batch_list, dim=0)

        return x_new


if __name__ == '__main__':
    x = torch.randn(32, 8, 4)
    y = DataPatch(3)(x)
    print(y.shape)
