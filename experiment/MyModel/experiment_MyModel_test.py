import os
import sys
import argparse
import json
import numpy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("--work_direct", type=str)
parser.add_argument("--hyper_params_path", type=str)
parser.add_argument("--model_params_path", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--test_data_path", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--start_idx", type=int)
parser.add_argument("--num_feature", type=int)
parser.add_argument("--save_path", type=str)
parser.add_argument("--picture_path", type=str)
parser.add_argument("--model_name", type=str)
args: argparse.Namespace = parser.parse_args()

project_dir: str = args.work_direct
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.chdir(args.work_direct)

from model.MyModel.MyModel import *
from utils.data_process import *

with open(args.hyper_params_path, "r", encoding="utf-8") as f:
    hyper_params: dict[str, any] = json.load(f)
my_model: MyModel = MyModel(
    **hyper_params,
    patch_time_step=(hyper_params["time_step"] - hyper_params["patch_size"] + 1)
)
my_model.load_state_dict(torch.load(args.model_params_path))
my_model.to(args.device)
loss_mse: torch.nn.MSELoss = torch.nn.MSELoss(reduction="mean")
loss_mae: torch.nn.L1Loss = torch.nn.L1Loss(reduction="mean")
test_data_loader: DataLoader = train_loader(win_size=hyper_params["time_step"] + 8,
                                            batch_size=32,
                                            data_path=args.test_data_path,
                                            start_idx=args.start_idx,
                                            num_feature=args.num_feature,
                                            target_size=8
                                            )
predict_length: list[int] = [1, 4, 8]
my_model.eval()
with torch.no_grad():
    for length in predict_length:
        mse_list: list[float] = []
        mae_list: list[float] = []
        drow_list: list[numpy.ndarray] = []
        for i, (x, y) in enumerate(tqdm(test_data_loader)):
            data: torch.Tensor = x.to(args.device)
            label: torch.Tensor = y.to(args.device)
            pred_x: torch.Tensor = my_model(data)[:, :length]
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

        with open(args.save_path + r"loss.txt", "a", encoding="utf-8") as f:
            f.write(f"{args.model_name}\nforcast_length: {length}\ndata_name: {args.data_name}\n"
                    f"mse: {mse_dataset}\nmae: {mae_dataset}\n")

        pred_line: numpy.ndarray = numpy.hstack([drow_list[0], drow_list[1]])
        label_line: numpy.ndarray = numpy.hstack([drow_list[0], drow_list[2]])
        plt.figure(figsize=(9, 5))
        plt.ylim(-2, 2)
        plt.xlim(0, 24)
        plt.plot(range(len(pred_line)), pred_line, label="pred_line", color='#1f77b4', linewidth=2, marker='o')
        plt.plot(range(len(label_line)), label_line, label="true_line", color='#ff7f0e', linewidth=2, marker='s')
        plt.xlabel("time_step")
        plt.ylabel("value")
        plt.title(f"{args.model_name}&step{length}&{args.data_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(
            args.picture_path + f"{length}_{args.data_name}_{args.model_name}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=1,
            facecolor="white"
        )
