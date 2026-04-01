import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import json
import os
import argparse
import sys
import matplotlib.pyplot as plt
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_direct", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--num_feature", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--picture_path", type=str)
    return parser.parse_args()


args = parse_args()
project_root = args.work_direct
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(args.work_direct)

from model.MyModel.MyModel import *
from utils.data_process import *

config = {
    "patch_size": [2, 4, 8],
    "num_block": [2, 4, 8],
    "win_size": 24,
    "batch_size": 32,
    "random_epoch": 1,
    "epoch": 100,
    "num_heads": [2, 4, 8],
    "hidden_size": [32, 64, 128, 256, 512],
    "learning_rate": 0.001,
    "g": 0.98,
    "t": 1
}


# def random_train(data_name, data_path, start_idx, num_features, save_path):
#     path = save_path + f"{data_name}_hyper_param.json"
#     search = product(
#         config["patch_size"],
#         config["num_block"],
#         config["num_heads"],
#         config["hidden_size"]
#     )
#     loss_rem1 = 0
#     for idx, (i, j, k, z) in enumerate(search):
#         para = {
#             "patch_size": i,
#             "num_block": j,
#             "time_step": int(config["win_size"] - 8),
#             "embed_dim": z,
#             "num_heads": k,
#             "feature_dim": args.num_feature,
#             "dropout": 0.5
#         }
#         if idx == 0:
#             with open(path, "w", encoding="utf-8") as f:
#                 json.dump(para, f, indent=4, ensure_ascii=False)
#
#         device = torch.device(args.device)
#         my_model = MyModel(
#             **para, patch_time_step=(para["time_step"] - para["patch_size"] + 1)
#         ).to(device)
#         epochs = config["random_epoch"]
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(my_model.parameters(), lr=config["learning_rate"])
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["g"])
#         train_data = train_loader(int(para["time_step"] + 8), config["batch_size"],
#                                   data_path=data_path,
#                                   start_idx=start_idx, num_feature=num_features, target_size=8)
#         loss_rem2 = 0
#         # torch.autograd.set_detect_anomaly(True)
#         my_model.train()
#         stop_step = 0
#         if_break_all = False
#         for epoch in range(epochs):
#             # print(f"Random_Search: {i + 1}, Epoch: {epoch + 1}")
#             if if_break_all:
#                 break
#             for idx, (data, label) in enumerate(tqdm(train_data)):
#                 data, label = data.to(device), label.to(device)
#                 output = my_model(data)
#                 # print(output.size(), label.size())
#                 loss = criterion(output, label)
#                 print(f"Loss: {loss.item()}")
#                 loss_rem2 = loss if epoch == 0 and idx == 0 else loss_rem2
#                 if loss <= loss_rem2:
#                     loss_rem2 = loss
#                 else:
#                     stop_step += 1
#                     if stop_step == 10:
#                         if_break_all = True
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             scheduler.step()
#         if idx == 0:
#             loss_rem1 = loss_rem2
#         if loss_rem2 <= loss_rem1:
#             loss_rem1 = loss_rem2
#             with open(path, "w", encoding='utf-8') as f:
#                 json.dump(para, f, indent=4, ensure_ascii=False)
#
#     return path


# path = random_train(args.data_name, args.data_path, args.start_idx, args.num_feature, args.save_path)
path = r"D:\Code_Working\Python\GitRepository\MixModel\experiment\MyModel\params\SHZXS_hyper_param.json"


def train_model(data_name, data_path, start_idx, num_features, save_path, para_path):
    with open(para_path, "r", encoding="utf-8") as f:
        para = json.load(f)

    device = torch.device(args.device)
    my_model = MyModel(
        **para, patch_time_step=(para["time_step"] - para["patch_size"] + 1)
    ).to(device)
    epochs = config["epoch"]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["g"])
    train_data = train_loader(int(para["time_step"] + 8), config["batch_size"],
                              data_path=data_path,
                              start_idx=start_idx, num_feature=num_features, target_size=8)
    loss_rem2 = 0.0
    my_model.train()
    loss_list = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        epoch_loss = 0.0
        for idx, (data, label) in enumerate(tqdm(train_data)):
            data, label = data.to(device), label.to(device)
            output = my_model(data)
            loss = criterion(output, label)
            print(f"Loss: {loss.item()}")
            loss_rem2 = loss.item() if epoch == 0 and idx == 0 else loss_rem2
            if loss <= loss_rem2:
                loss_rem2 = loss.item()
                epoch_loss = loss_rem2
                torch.save(my_model.state_dict(), save_path + f"{data_name}_train_param.pth")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(epoch_loss)
        scheduler.step()

    # plt.figure(figsize=(9, 5))
    # plt.plot(loss_list, color='#1f77b4', linewidth=2, marker='o')
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("MixModel&Epoch&Loss")
    # plt.grid(alpha=0.3)
    # plt.savefig(
    #     args.picture_path + "/MixModel&Epoch&Loss.png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     pad_inches=1,
    #     facecolor="white"
    # )

train_model(args.data_name, args.data_path, args.start_idx, args.num_feature, args.save_path, path)
