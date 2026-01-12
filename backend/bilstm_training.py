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
    "save_path": os.path.join(ROOT, "results", "SoH_CaseA_BiLSTM.pth"), # 结果保存路径
    "settings_path": os.path.join(ROOT, "Settings", "settings_SoH_CaseA.pth"), # 默认设置文件路径

    # 2. 数据集划分
    "train_cells": "91,100",  # 训练集电池编号
    "test_cells": "124",      # 测试集电池编号
    "perc_val": 0.2,          # 验证集比例

    # 3. 网络结构参数
    "hidden_dim": 32,         # LSTM隐藏层节点数
    "num_layers": 2,          # LSTM层数
    "dropout": 0.0,           # Dropout概率 (仅当num_layers > 1时有效)

    # 4. 训练超参数
    "epochs": 100,            # 训练轮数
    "batch_size": 128,        # 批大小
    "lr": 1e-3,               # 初始学习率
    "weight_decay": 0.0,      # 权重衰减 (L2正则化)

    # 5. 优化器与学习率调度
    "optimizer": "Adam",      # 优化器选择：支持 "Adam", "SGD"
    "step_size": 50,          # 学习率衰减步长 (每多少个epoch衰减一次)
    "gamma": 0.1,             # 学习率衰减率

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


def _build_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float) -> optim.Optimizer:
    key = name.strip().lower()
    if key == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


class BiLSTMModel(nn.Module):
    """
    双向LSTM模型用于电池健康状态（SoH）预测。
    参考: SOH_CaseA_BiLSTM.py
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        # bidirectional=True表示使用双向LSTM，输出维度会是hidden_dim*2
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # 创建全连接层，将LSTM输出映射到最终预测值
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        
        # 增加一个维度以匹配目标张量的形状 (batch_size, 1) -> (batch_size, 1, 1) ? 
        # 原代码: return out.unsqueeze(1)
        # 目标 shape 通常是 (batch_size, 1) 或 (batch_size, seq_len, 1)
        # 在 functions.py 的 train loop 中，targets 是 (batch, seq, dim)
        # 这里 BiLSTM 是 sequence-to-one (last step), so output is (batch, 1).
        # 但为了兼容 functions.train loop 的 MSELoss 计算，可能需要调整 shape.
        # SOH_CaseA_BiLSTM.py 中的 targets_train 是 [:, :, 0:1] -> (N, seq_len, 1)
        # 如果 seq_len=1, 则是 (N, 1, 1).
        # out 是 (N, 1). out.unsqueeze(1) -> (N, 1, 1).
        return out.unsqueeze(1)


def _eval_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    eps = 1e-8
    # 确保 shape 匹配
    if pred.shape != true.shape:
        pred = pred.view_as(true)
    
    pred_val = pred.detach()
    true_val = true.detach()
    
    mae = torch.mean(torch.abs(pred_val - true_val)).item()
    mse = torch.mean((pred_val - true_val) ** 2).item()
    rmspe = torch.sqrt(torch.mean(((pred_val - true_val) / (true_val + eps)) ** 2)).item()
    
    ss_res = torch.sum((true_val - pred_val) ** 2)
    ss_tot = torch.sum((true_val - torch.mean(true_val)) ** 2)
    r2 = (1 - ss_res / (ss_tot + eps)).item()
    
    return {"MAE": mae, "MSE": mse, "RMSPE": rmspe, "R2": r2}


@dataclass
class TrainConfig:
    data_path: str
    settings_path: str
    seq_len: int
    perc_val: float
    train_cells: list[int]
    test_cells: list[int]
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    step_size: int
    gamma: float
    hidden_dim: int
    num_layers: int
    dropout: float
    device: str
    seed: int
    save_path: str


def _load_settings(path: str) -> dict:
    if os.path.exists(path):
        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)
    return {}


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


def train_bilstm(cfg: TrainConfig) -> dict[str, dict[str, float]]:
    _set_seed(cfg.seed)
    dev = _device(cfg.device)

    # 1. Load Data
    data = func.SeversonBattery(cfg.data_path, seq_len=cfg.seq_len)
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

    # 2. Build Model
    input_dim = inputs_train.shape[2]
    output_dim = 1
    
    model = BiLSTMModel(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(dev)

    # 3. Setup Optimizer
    optimizer = _build_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    criterion = nn.MSELoss()

    # 4. Data Loader
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # 5. Training Loop
    # 参考 SOH_CaseA_BiLSTM.py 的训练逻辑，但为了统一接口，我们可以尝试复用 functions.train 
    # 或者重写一个简单的 loop 保持与原 BiLSTM 脚本一致。
    # 原脚本很简单，直接 epoch -> batch -> step.
    # 这里我们重写一遍以完全控制。

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        count = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            count += x_batch.size(0)
            
        scheduler.step()
        avg_train_loss = total_loss / count if count > 0 else 0.0
        history["train_loss"].append(avg_train_loss)

        # Validation (optional for logging)
        model.eval()
        with torch.no_grad():
            val_out = model(inputs_val)
            val_loss = criterion(val_out, targets_val).item()
            history["val_loss"].append(val_loss)

    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        pred_train = model(inputs_train)
        pred_val = model(inputs_val)
        pred_test = model(inputs_test)

    # 转换回 SOH (1 - PCL)
    # targets 是 PCL (容量衰减), 所以 SOH = 1 - PCL
    # 预测值也是 PCL
    pred_train_soh = 1.0 - pred_train
    pred_val_soh = 1.0 - pred_val
    pred_test_soh = 1.0 - pred_test
    
    true_train_soh = 1.0 - targets_train
    true_val_soh = 1.0 - targets_val
    true_test_soh = 1.0 - targets_test

    report = {
        "train": _eval_metrics(pred_train_soh, true_train_soh),
        "val": _eval_metrics(pred_val_soh, true_val_soh),
        "test": _eval_metrics(pred_test_soh, true_test_soh),
    }

    # 7. Save Results
    _ensure_results_dir(cfg.save_path)
    results = {
        "U_true": true_test_soh.detach().cpu().numpy().squeeze(),
        "U_pred": pred_test_soh.detach().cpu().numpy().squeeze(),
        "Cycles": inputs_test[:, :, -1:].detach().cpu().numpy().squeeze(),
        "metric": report,
        "history": history,
        "config": str(cfg),
        "artifact_version": 2,
        "model_type": "BiLSTM",
        "train_config": cfg.__dict__,
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "scaler_inputs": {"mean": mean_inputs_train.detach().cpu(), "std": std_inputs_train.detach().cpu()},
        "scaler_targets": {"mean": mean_targets_train.detach().cpu(), "std": std_targets_train.detach().cpu()},
    }
    torch.save(results, cfg.save_path)

    return report


def _print_report(report: dict[str, dict[str, float]]) -> None:
    for split in ["train", "val", "test"]:
        m = report[split]
        print(f"{split.upper()}  RMSPE={m['RMSPE']:.6f}  MSE={m['MSE']:.6f}  R2={m['R2']:.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    
    # Defaults from USER_CONFIG
    parser.add_argument("--data_path", type=str, default=USER_CONFIG["data_path"])
    parser.add_argument("--settings_path", type=str, default=USER_CONFIG["settings_path"])
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--perc_val", type=float, default=USER_CONFIG["perc_val"])
    parser.add_argument("--train_cells", type=str, default=USER_CONFIG["train_cells"])
    parser.add_argument("--test_cells", type=str, default=USER_CONFIG["test_cells"])

    parser.add_argument("--epochs", type=int, default=USER_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=USER_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=USER_CONFIG["lr"])
    parser.add_argument("--optimizer", type=str, default=USER_CONFIG["optimizer"], choices=["Adam", "SGD"])
    parser.add_argument("--weight_decay", type=float, default=USER_CONFIG["weight_decay"])
    parser.add_argument("--step_size", type=int, default=USER_CONFIG["step_size"])
    parser.add_argument("--gamma", type=float, default=USER_CONFIG["gamma"])
    
    parser.add_argument("--hidden_dim", type=int, default=USER_CONFIG["hidden_dim"])
    parser.add_argument("--num_layers", type=int, default=USER_CONFIG["num_layers"])
    parser.add_argument("--dropout", type=float, default=USER_CONFIG["dropout"])

    parser.add_argument("--device", type=str, default=USER_CONFIG["device"])
    parser.add_argument("--seed", type=int, default=USER_CONFIG["seed"])
    parser.add_argument("--save_path", type=str, default=USER_CONFIG["save_path"])

    args = parser.parse_args(argv)

    cfg = TrainConfig(
        data_path=args.data_path,
        settings_path=args.settings_path,
        seq_len=args.seq_len,
        perc_val=args.perc_val,
        train_cells=_parse_int_list(args.train_cells),
        test_cells=_parse_int_list(args.test_cells),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        save_path=args.save_path,
    )

    report = train_bilstm(cfg)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
