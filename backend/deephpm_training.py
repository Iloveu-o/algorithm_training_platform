# -*- coding: utf-8 -*-
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT = _project_root()
sys.path.append(ROOT)

import functions as func  # noqa: E402


# ==========================================
#  用户配置区域 (后端开发人员修改此处参数即可)
# ==========================================
USER_CONFIG = {
    # 1. 数据与路径配置
    "data_path": os.path.join(ROOT, "SeversonBattery.mat"), # 数据集路径
    "save_path": os.path.join(ROOT, "results", "SoH_CaseA_DeepHPM_Sum.pth"), # 结果保存路径
    "settings_path": os.path.join(ROOT, "Settings", "settings_SoH_CaseA.pth"), # 默认设置文件路径

    # 2. 数据集划分
    "train_cells": "91,124",  # 训练集电池编号
    "test_cells": "100",      # 测试集电池编号
    "perc_val": 0.2,          # 验证集比例

    # 3. 网络结构参数
    "layers": "32,32",        # 隐藏层结构
    "inputs_dynamical": "U",  # 动态模型输入变量 (例如 "U" 或 "U,t")
    "inputs_dim_dynamical": 1, # 动态模型输入维度 (对应 inputs_dynamical 的总维度)

    # 4. 训练超参数
    "epochs": 100,            # 训练轮数
    "batch_size": 128,        # 批大小
    "lr": 1e-3,               # 初始学习率
    "step_size": 50,          # 学习率衰减步长
    "gamma": 0.1,             # 学习率衰减率

    # 5. 优化器
    "optimizer": "Adam",      # 优化器选择："Adam", "SGD"

    # 6. 其他
    "device": "auto",         # 设备："auto", "cpu", "cuda"
    "seed": 1234,             # 随机种子
}
# ==========================================


def _parse_int_list(value: str) -> list[int]:
    s = str(value).strip()
    if not s:
        return []
    out: list[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = [p.strip() for p in token.split("-") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Invalid range token: {token}")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                raise ValueError(f"Invalid range token (start > end): {token}")
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(token))
    return out


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float) -> optim.Optimizer:
    key = name.strip().lower()
    if key == "adam":
        return optim.Adam(params, lr=lr)
    if key == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


@dataclass
class TrainConfig:
    data_path: str
    save_path: str
    train_cells: list[int]
    test_cells: list[int]
    perc_val: float
    layers: list[int]
    inputs_dynamical: str
    inputs_dim_dynamical: int
    epochs: int
    batch_size: int
    lr: float
    step_size: int
    gamma: float
    optimizer: str
    device: str
    seed: int


def _ensure_results_dir(save_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _state_dict_to_cpu(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def train_deephpm(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    dev = _device(cfg.device)
    
    # 1. Load Data
    # 序列长度固定为1，参考 SoH_CaseA_DeepHPM_Sum.py
    seq_len = 1
    data = func.SeversonBattery(cfg.data_path, seq_len=seq_len)
    
    inputs_dict, targets_dict = func.create_chosen_cells(
        data,
        idx_cells_train=cfg.train_cells,
        idx_cells_test=cfg.test_cells,
        perc_val=cfg.perc_val,
    )

    inputs_train = inputs_dict["train"].to(dev)
    inputs_val = inputs_dict["val"].to(dev)
    inputs_test = inputs_dict["test"].to(dev)
    
    # 目标：容量衰减 (第一列)
    targets_train = targets_dict["train"][:, :, 0:1].to(dev)
    targets_val = targets_dict["val"][:, :, 0:1].to(dev)
    targets_test = targets_dict["test"][:, :, 0:1].to(dev)

    inputs_dim = inputs_train.shape[2]
    outputs_dim = 1

    # Standardize
    _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
    _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

    # Data Loader
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    # 2. Build Model
    # 禁用 cuDNN 以避免某些兼容性问题 (参考原代码)
    torch.backends.cudnn.enabled = False
    
    model = func.DeepHPMNN(
        seq_len=seq_len,
        inputs_dim=inputs_dim,
        outputs_dim=outputs_dim,
        layers=cfg.layers,
        scaler_inputs=(mean_inputs_train, std_inputs_train),
        scaler_targets=(mean_targets_train, std_targets_train),
        inputs_dynamical=cfg.inputs_dynamical,
        inputs_dim_dynamical=str(cfg.inputs_dim_dynamical) # DeepHPMNN expects string for eval() or direct int usage?
        # func.DeepHPMNN L958: self.inputs_dim_dynamical = eval(inputs_dim_dynamical)
        # So we should pass a string representation of the int, or modify usage.
        # However, passing str(1) -> eval("1") -> 1 works.
    ).to(dev)

    # 3. Setup Optimizer & Loss
    log_sigma_u = torch.zeros(())
    log_sigma_f = torch.zeros(())
    log_sigma_f_t = torch.zeros(())
    
    criterion = func.My_loss(mode='Sum')
    
    optimizer = _build_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # 4. Training Loop (Using func.train)
    model, results_epoch = func.train(
        num_epoch=cfg.epochs,
        batch_size=cfg.batch_size,
        train_loader=train_loader,
        num_slices_train=inputs_train.shape[0],
        inputs_val=inputs_val,
        targets_val=targets_val,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        log_sigma_u=log_sigma_u,
        log_sigma_f=log_sigma_f,
        log_sigma_f_t=log_sigma_f_t
    )

    # 5. Evaluation
    model.eval()
    
    # Test Evaluation
    U_pred_test, _, _ = model(inputs=inputs_test)
    
    # Convert to SoH (1 - PCL)
    U_pred_test_soh = 1.0 - U_pred_test
    targets_test_soh = 1.0 - targets_test
    
    # Calculate Metrics
    mse_test = torch.mean((U_pred_test_soh - targets_test_soh) ** 2).item()
    rmspe_test = torch.sqrt(torch.mean(((U_pred_test_soh - targets_test_soh) / (targets_test_soh + 1e-8)) ** 2)).item()
    
    ss_res = torch.sum((targets_test_soh - U_pred_test_soh) ** 2)
    ss_tot = torch.sum((targets_test_soh - torch.mean(targets_test_soh)) ** 2)
    r2_test = (1 - ss_res / (ss_tot + 1e-8)).item()

    print(f"Test RMSPE: {rmspe_test:.6f}")
    print(f"Test MSE: {mse_test:.6f}")
    print(f"Test R2: {r2_test:.6f}")

    # 6. Save Results
    _ensure_results_dir(cfg.save_path)
    results = {
        "U_true": targets_test_soh.detach().cpu().numpy().squeeze(),
        "U_pred": U_pred_test_soh.detach().cpu().numpy().squeeze(),
        "Cycles": inputs_test[:, :, -1:].detach().cpu().numpy().squeeze(),
        "metric": {"RMSPE": rmspe_test, "MSE": mse_test, "R2": r2_test},
        "config": str(cfg),
        "history": results_epoch, # func.train returns this dict
        "artifact_version": 2,
        "model_type": "DeepHPM",
        "train_config": cfg.__dict__,
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "scaler_inputs": {"mean": mean_inputs_train.detach().cpu(), "std": std_inputs_train.detach().cpu()},
        "scaler_targets": {"mean": mean_targets_train.detach().cpu(), "std": std_targets_train.detach().cpu()},
    }
    torch.save(results, cfg.save_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    
    # Defaults from USER_CONFIG
    parser.add_argument("--data_path", type=str, default=USER_CONFIG["data_path"])
    parser.add_argument("--save_path", type=str, default=USER_CONFIG["save_path"])
    parser.add_argument("--train_cells", type=str, default=USER_CONFIG["train_cells"])
    parser.add_argument("--test_cells", type=str, default=USER_CONFIG["test_cells"])
    parser.add_argument("--perc_val", type=float, default=USER_CONFIG["perc_val"])
    
    parser.add_argument("--layers", type=str, default=USER_CONFIG["layers"])
    parser.add_argument("--inputs_dynamical", type=str, default=USER_CONFIG["inputs_dynamical"])
    parser.add_argument("--inputs_dim_dynamical", type=int, default=USER_CONFIG["inputs_dim_dynamical"])
    
    parser.add_argument("--epochs", type=int, default=USER_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=USER_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=USER_CONFIG["lr"])
    parser.add_argument("--step_size", type=int, default=USER_CONFIG["step_size"])
    parser.add_argument("--gamma", type=float, default=USER_CONFIG["gamma"])
    parser.add_argument("--optimizer", type=str, default=USER_CONFIG["optimizer"], choices=["Adam", "SGD"])
    
    parser.add_argument("--device", type=str, default=USER_CONFIG["device"])
    parser.add_argument("--seed", type=int, default=USER_CONFIG["seed"])

    args = parser.parse_args(argv)

    cfg = TrainConfig(
        data_path=args.data_path,
        save_path=args.save_path,
        train_cells=_parse_int_list(args.train_cells),
        test_cells=_parse_int_list(args.test_cells),
        perc_val=args.perc_val,
        layers=_parse_int_list(args.layers),
        inputs_dynamical=args.inputs_dynamical,
        inputs_dim_dynamical=args.inputs_dim_dynamical,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        optimizer=args.optimizer,
        device=args.device,
        seed=args.seed,
    )

    train_deephpm(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
