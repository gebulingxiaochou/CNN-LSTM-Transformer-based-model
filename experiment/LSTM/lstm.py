import torch
import numpy
import torch.nn as nn
from tqdm import tqdm
import json
import os
import argparse
import sys
import matplotlib.pyplot as plt
from itertools import product
from collections.abc import Iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_direct", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--num_feature", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--target_size", default=8, type=int)
    parser.add_argument("--picture_path", type=str)
    parser.add_argument("--loss_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    return parser.parse_args()


ARGS = parse_args()
project_root = ARGS.work_direct
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(ARGS.work_direct)

from utils.data_process import *


class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int):
        super().__init__()
        self.lstm: nn.Module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        ).double()
        self.fc: nn.Module = nn.Linear(
            hidden_size,
            output_size
        ).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lstm(x)[0]
        return self.fc(out[:, -1, :])


def train() -> tuple[str, str]:
    device: torch.device = torch.device(ARGS.device)
    data: torch.Tensor = train_loader(
        win_size=24,
        batch_size=32,
        data_path=ARGS.data_path,
        start_idx=ARGS.start_idx,
        num_feature=ARGS.num_feature,
        target_size=ARGS.target_size
    )
    hidden_size_list: list[int] = [256]
    num_layers_list: list[int] = [4]
    search: Iterable = product(hidden_size_list, num_layers_list)
    search_loss: float = 0.0
    hyper_path: str = ARGS.save_path + f"{ARGS.data_name}_hyper_param.json"
    for index, (i, j) in enumerate(search):
        params: dict[str, any] = {
            "input_size": ARGS.num_feature,
            "hidden_size": i,
            "num_layers": j,
            "output_size": ARGS.target_size
        }
        lstm_search: nn.Module = MyLSTM(**params).to(device)
        loss: nn.Module = nn.MSELoss(reduction="mean")
        optimizer: torch.optim.Optimizer = torch.optim.Adam(lstm_search.parameters(), lr=0.001)
        # if index == 0:
        #     with open(hyper_path, "w") as f:
        #         json.dump(params, f)

        for epoch in range(1):
            for idx, (x, y) in enumerate(tqdm(data)):
                x: torch.Tensor = x.to(device)
                y: torch.Tensor = y.to(device)
                output: torch.Tensor = lstm_search(x)
                mse_loss: torch.Tensor = loss(output, y)
                if index == 0 and epoch == 0 and idx == 0:
                    search_loss = mse_loss.item()
                else:
                    if search_loss >= mse_loss.item():
                        search_loss = mse_loss.item()
                        with open(hyper_path, "w") as f:
                            json.dump(params, f)
                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()

    with open(hyper_path, "r") as f:
        params = json.load(f)
    model_path: str = ARGS.save_path + f"{ARGS.data_name}_train_param.pth"
    lstm_train: nn.Module = MyLSTM(**params).to(device)
    loss: nn.Module = nn.MSELoss(reduction="mean")
    optimizer: torch.optim.Optimizer = torch.optim.Adam(lstm_train.parameters(), lr=0.001)
    scheduler: torch.optim.lr_scheduler.ExponentialLR = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )
    epoch_loss: float = 0.0
    for epoch in range(100):
        print(f'Epoch {epoch}')
        for idx, (x, y) in enumerate(tqdm(data)):
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)
            output: torch.Tensor = lstm_train(x)
            mse_loss: torch.Tensor = loss(output, y)
            if epoch == 0 and idx == 0:
                epoch_loss = mse_loss.item()
            else:
                if epoch_loss >= mse_loss.item():
                    epoch_loss = mse_loss.item()
                    torch.save(lstm_train.state_dict(), model_path)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
        scheduler.step()

    return hyper_path, model_path


def test():
    hyper_path: str
    model_path: str
    hyper_path, model_path= train()
    # hyper_path = "/root/autodl-tmp/LSTM/params/HSB_hyper_param.json"
    # model_path = "/root/autodl-tmp/LSTM/params/HSB_train_param.pth"
    with open(hyper_path, "r") as f:
        hyper_params: dict[str, any] = json.load(f)
    model = MyLSTM(**hyper_params)
    model.load_state_dict(torch.load(model_path))
    model.to(ARGS.device)
    loss_mse: torch.nn.MSELoss = torch.nn.MSELoss(reduction="mean")
    loss_mae: torch.nn.L1Loss = torch.nn.L1Loss(reduction="mean")
    test_data_loader: DataLoader = train_loader(win_size=24,
                                                batch_size=32,
                                                data_path=ARGS.test_data_path,
                                                start_idx=ARGS.start_idx,
                                                num_feature=ARGS.num_feature,
                                                target_size=8
                                                )
    predict_length: list[int] = [1, 4, 8]
    model.eval()
    with torch.no_grad():
        for length in predict_length:
            mse_list: list[float] = []
            mae_list: list[float] = []
            drow_list: list[numpy.ndarray] = []
            for i, (x, y) in enumerate(tqdm(test_data_loader)):
                data: torch.Tensor = x.to(ARGS.device)
                label: torch.Tensor = y.to(ARGS.device)
                pred_x: torch.Tensor = model(data)[:, :length]
                label_y: torch.Tensor = label[:, :length]
                mse: float = loss_mse(pred_x, label_y).item()
                mae: float = loss_mae(pred_x, label_y).item()
                if i == 0:
                    drow_list = [data[0, :, -1].detach().cpu().numpy(),
                                 pred_x[0].detach().cpu().numpy(),
                                 label_y[0].detach().cpu().numpy()]
                mse_list.append(mse)
                mae_list.append(mae)
            mse_dataset: float = sum(mse_list) / len(mse_list)
            mae_dataset: float = sum(mae_list) / len(mae_list)

            with open(ARGS.loss_path + r"loss.txt", "a", encoding="utf-8") as f:
                f.write(f"LSTM\nforcast_length: {length}\ndata_name: {ARGS.data_name}\n"
                        f"mse: {mse_dataset}\nmae: {mae_dataset}\n")

            pred_line: numpy.ndarray = numpy.hstack([drow_list[0], drow_list[1]])
            label_line: numpy.ndarray = numpy.hstack([drow_list[0], drow_list[2]])
            plt.figure(figsize=(9, 5))
            plt.ylim(-2, 2)
            plt.xlim(0, 24)
            plt.plot(pred_line, label="pred_line", color='#1f77b4', linewidth=2, marker='o')
            plt.plot(label_line, label="true_line", color='#ff7f0e', linewidth=2, marker='s')
            plt.xlabel("time_step")
            plt.ylabel("value")
            plt.title(f"LSTM&step{length}&{ARGS.data_name}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(
                ARGS.picture_path + f"{length}_{ARGS.data_name}_LSTM.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=1,
                facecolor="white"
            )


if __name__ == '__main__':
    test()
