import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import json
import os
import argparse
import sys
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
    parser.add_argument("--target_size", default=8 ,type=int)
    return parser.parse_args()


args = parse_args()
project_root = args.work_direct
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(args.work_direct)

from model.Informer.Informer import *
from utils.data_process import *

config = {
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


def random_train(data_name, data_path, start_idx, num_features, save_path):
    path = save_path + f"{data_name}_hyper_param.json"
    search = product(config["num_heads"], config["hidden_size"])
    loss_rem1 = 0
    for idx, (i, j) in enumerate(search):
        para = {
            "c_in": num_features,
            "d_model": j,
            "n_head": i,
            "seq_len": int(config["win_size"] - args.target_size),
        }
        if idx == 0:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(para, f, indent=4, ensure_ascii=False)

        device = torch.device(args.device)
        inFormer = Informer(**para, device=args.device).to(device)
        epochs = config["random_epoch"]
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(inFormer.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["g"])
        train_data = train_loader(int(para["seq_len"] + args.target_size), config["batch_size"],
                                  data_path=data_path,
                                  start_idx=start_idx, num_feature=num_features, target_size=args.target_size)
        loss_rem2 = 0
        inFormer.train()
        for epoch in range(epochs):
            for idx, (data, label) in enumerate(tqdm(train_data)):
                data, label = data.to(device), label.to(device)
                output = inFormer(data)
                loss = criterion(output, label)
                print(f"Loss: {loss.item()}")
                loss_rem2 = loss if epoch == 0 and idx == 0 else loss_rem2
                if loss <= loss_rem2:
                    loss_rem2 = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        if idx == 0:
            loss_rem1 = loss_rem2
        if loss_rem2 <= loss_rem1:
            loss_rem1 = loss_rem2
            with open(path, "w", encoding='utf-8') as f:
                json.dump(para, f, indent=4, ensure_ascii=False)

    return path


path = random_train(args.data_name, args.data_path, args.start_idx, args.num_feature, args.save_path)


def train_model(data_name, data_path, start_idx, num_features, save_path, para_path):
    with open(para_path, "r", encoding="utf-8") as f:
        para = json.load(f)

    device = torch.device(args.device)
    inFormer = Informer(**para, device=args.device).to(device)
    epochs = config["epoch"]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(inFormer.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["g"])
    train_data = train_loader(int(para["seq_len"] + args.target_size), config["batch_size"],
                              data_path=data_path,
                              start_idx=start_idx, num_feature=num_features, target_size=args.target_size)

    loss_rem2 = 0
    inFormer.train()
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        for idx, (data, label) in enumerate(tqdm(train_data)):
            data, label = data.to(device), label.to(device)
            output = inFormer(data)
            loss = criterion(output, label)
            print(f"Loss: {loss.item()}")
            loss_rem2 = loss.item() if epoch == 0 and idx == 0 else loss_rem2
            if loss <= loss_rem2:
                loss_rem2 = loss
                torch.save(inFormer.state_dict(), save_path + f"{data_name}_train_param.pth")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


train_model(args.data_name, args.data_path, args.start_idx, args.num_feature, args.save_path, path)
