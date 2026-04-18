import os
import sys
import argparse
import json
import numpy
import torch
from torch.nn.modules import loss
from tqdm import tqdm

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
args: argparse.Namespace = parser.parse_args()

project_dir: str = args.work_direct
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.chdir(args.work_direct)

from model.MyModel.MyModel import *
from model.MyModel.MyModel_patch_cancel import *
from model.MyModel.MyModel_feature_attention_cancel import *
from model.MyModel.MyModel_feature_mlp_cancel import *
from model.MyModel.MyModel_time_attention_cancel import *
from model.MyModel.MyModel_time_mlp_cancel import *
from utils.data_process import *

model_list: list[torch.nn.Module] = []
with open(args.hyper_params_path, "r", encoding="utf-8") as f:
    hyper_params: dict[str, any] = json.load(f)

my_model1: MyModelPatchCancel = MyModelPatchCancel(**hyper_params).to(args.device)
model_list.append(my_model1)

my_model2: MyModelFeaAttenCancel = MyModelFeaAttenCancel(
    **hyper_params,
    patch_time_step=(hyper_params["time_step"] - hyper_params["patch_size"] + 1)
).to(args.device)
model_list.append(my_model2)

my_model3: MyModelFeaMLPCancel = MyModelFeaMLPCancel(
    **hyper_params,
    patch_time_step=(hyper_params["time_step"] - hyper_params["patch_size"] + 1)
).to(args.device)
model_list.append(my_model3)

my_model4: MyModelTimeAttenCancel = MyModelTimeAttenCancel(
    **hyper_params,
    patch_time_step=(hyper_params["time_step"] - hyper_params["patch_size"] + 1)
).to(args.device)
model_list.append(my_model4)

my_model5: MyModelTimeMLPCancel = MyModelTimeMLPCancel(
    **hyper_params,
    patch_time_step=(hyper_params["time_step"] - hyper_params["patch_size"] + 1)
).to(args.device)
model_list.append(my_model5)

my_model6: MyModel = MyModel(
    **hyper_params,
    patch_time_step=(hyper_params["time_step"] - hyper_params["patch_size"] + 1)
).to(args.device)
model_list.append(my_model6)

test_data_loader: DataLoader = train_loader(win_size=hyper_params["time_step"] + 8,
                                            batch_size=32,
                                            data_path=args.test_data_path,
                                            start_idx=args.start_idx,
                                            num_feature=args.num_feature,
                                            target_size=8
                                            )

loss_list: list[float] = []
for model in model_list:
    loss_mse: torch.nn.MSELoss = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model_loss: float = 0.0
    for epoch in range(10):
        for i, (x, y) in enumerate(tqdm(test_data_loader)):
            data: torch.Tensor = x.to(args.device)
            label: torch.Tensor = y.to(args.device)
            pred_x: torch.Tensor = model(data)
            mse: float = loss_mse(pred_x, label)
            if epoch == 0 and i ==0:
                model_loss = mse
            else:
                if mse <= model_loss:
                    model_loss = mse
            optimizer.zero_grad()
            mse.backward()
            optimizer.step()
        scheduler.step()
    loss_list.append(model_loss)

with open(args.save_path + "ablation_loss.txt", "a", encoding="utf-8") as f:
        f.write(f"{args.data_name}&Patch_cancel: {loss_list[0]}\n")
        f.write(f"{args.data_name}&Feature_attention_cancel: {loss_list[1]}\n")
        f.write(f"{args.data_name}&Feature_mlp_cancel: {loss_list[2]}\n")
        f.write(f"{args.data_name}&Time_attention_cancel: {loss_list[3]}\n")
        f.write(f"{args.data_name}&Time_mlp_cancel: {loss_list[4]}\n")
        f.write(f"{args.data_name}&MixModel: {loss_list[5]}\n")