# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

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


class TrainingCancelled(Exception):
    pass


# ==========================================
#  用户配置区域 (后端开发人员修改此处参数即可)
# ==========================================
USER_CONFIG = {
    # 0. 模型选择
    "model_type": "DeepHPM", # 算法模型选择："Baseline", "BiLSTM", "DeepHPM"

    # 1. 数据与路径配置
    "data_path": os.path.join(ROOT, "SeversonBattery.mat"), # 数据集路径
    "save_path": "", # 结果保存路径 (留空则自动生成：模型名_年月日时分秒.pth)
    "settings_path": os.path.join(ROOT, "Settings", "settings_SoH_CaseA.pth"), # 默认设置文件路径

    # 2. 数据集划分
    "train_cells": "91-100",  # 训练集电池编号 (DeepHPM建议更多数据，如 "91,124")
    "test_cells": "124",      # 测试集电池编号
    "perc_val": 0.2,          # 验证集比例

    # 3. 通用网络结构参数 (Baseline & DeepHPM)
    "layers": "32,8",        # 隐藏层结构 (用于 Baseline 和 DeepHPM)
    "activation": "Tanh",     # 激活函数 (用于 Baseline): "Tanh", "ReLU", "Sigmoid", "LeakyReLU", "Sin"
    
    # 4. BiLSTM 专属结构参数
    "hidden_dim": 32,         # LSTM隐藏层节点数
    "num_layers": 2,          # LSTM层数
    "dropout": 0.0,           # Dropout概率

    # 5. DeepHPM 专属结构参数
    "inputs_dynamical": "U",  # 动态模型输入变量
    "inputs_dim_dynamical": 1, # 动态模型输入维度

    # 6. 训练超参数
    "epochs": 100,            # 训练轮数
    "batch_size": 32,        # 批大小
    "lr": 1e-4,               # 初始学习率
    "weight_decay": 0.0,      # 权重衰减
    "step_size": 200,          # 学习率衰减步长
    "gamma": 0.5,             # 学习率衰减率

    # 7. 优化器
    "optimizer": "SGD",      # 优化器选择："Adam", "SGD"

    # 8. 其他
    "device": "cuda",         # 设备："auto", "cpu", "cuda"
    "seed": 1234,             # 随机种子
}
# ==========================================


def _parse_int_list(value: str) -> List[int]:
    s = str(value).strip()
    if not s:
        return []
    out: List[int] = []
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


def _get_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "leakyrelu":
        return nn.LeakyReLU()
    if key == "sin":
        return func.Sin()
    raise ValueError(f"Unsupported activation: {name}")


class CustomNeuralNet(nn.Module):
    """
    支持多种激活函数的自定义神经网络。
    替代 functions.Neural_Net 以提供更好的灵活性。
    """
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation_name='Tanh'):
        super(CustomNeuralNet, self).__init__()
        self.seq_len = seq_len
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        
        modules = []
        
        # Input layer
        modules.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
        nn.init.xavier_normal_(modules[-1].weight)
        modules.append(_get_activation(activation_name))
        modules.append(nn.Dropout(p=0.2))
        
        # Hidden layers
        for l in range(len(layers) - 1):
            modules.append(nn.Linear(in_features=layers[l], out_features=layers[l + 1]))
            nn.init.xavier_normal_(modules[-1].weight)
            modules.append(_get_activation(activation_name))
            modules.append(nn.Dropout(p=0.2))
        
        # Output layer
        last_dim = layers[-1] if len(layers) > 0 else inputs_dim
        modules.append(nn.Linear(in_features=last_dim, out_features=outputs_dim))
        nn.init.xavier_normal_(modules[-1].weight)
        
        self.NN = nn.Sequential(*modules)

    def forward(self, x):
        # 保持与 functions.Neural_Net 一致的输入输出处理
        # x shape: (batch, seq_len * input_dim) ? or (batch, input_dim)?
        # functions.Neural_Net forward:
        # x_2D = x.contiguous().view((-1, self.inputs_dim))
        # NN_out_2D = self.NN(x_2D)
        # u_pred = NN_out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))
        
        x_2D = x.contiguous().view((-1, self.inputs_dim))
        NN_out_2D = self.NN(x_2D)
        u_pred = NN_out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))
        return u_pred


class BiLSTMModel(nn.Module):
    """
    双向LSTM模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last step
        out = self.fc(out)
        return out.unsqueeze(1) # (batch, 1, 1) to match targets


class BiLSTMWrapper(nn.Module):
    """
    包装 BiLSTM 模型以适配 functions.train 的接口 (返回 U, F, F_t)
    """
    def __init__(self, model):
        super(BiLSTMWrapper, self).__init__()
        self.model = model

    def forward(self, inputs):
        U = self.model(inputs)
        # BiLSTM 是纯数据驱动，没有物理约束，F 和 F_t 为 0
        F = torch.zeros_like(U)
        F_t = torch.zeros_like(U)
        return U, F, F_t
    
    # 为了兼容 func.train 记录参数，添加虚拟属性
    @property
    def p_r(self): return torch.tensor(0.0)
    @property
    def p_K(self): return torch.tensor(0.0)
    @property
    def p_C(self): return torch.tensor(0.0)


@dataclass
class TrainConfig:
    model_type: str
    data_path: str
    save_path: str
    train_cells: List[int]
    test_cells: List[int]
    perc_val: float
    layers: List[int]
    activation: str
    hidden_dim: int
    num_layers: int
    dropout: float
    inputs_dynamical: str
    inputs_dim_dynamical: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    step_size: int
    gamma: float
    optimizer: str
    device: str
    seed: int


def _eval_metrics(pred_soh: torch.Tensor, true_soh: torch.Tensor) -> dict:
    eps = 1e-8
    pred = pred_soh.detach()
    y = true_soh.detach()
    mae = torch.mean(torch.abs(pred - y)).item()
    mse = torch.mean((pred - y) ** 2).item()
    rmspe = torch.sqrt(torch.mean(((pred - y) / (y + eps)) ** 2)).item()
    ss_res = torch.sum((y - pred) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    r2 = (1 - ss_res / (ss_tot + eps)).item()
    return {"MAE": mae, "MSE": mse, "RMSPE": rmspe, "R2": r2}


def _ensure_results_dir(save_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _write_readable_log(save_path: str, cfg: "TrainConfig", metrics: dict, started_at: datetime.datetime, finished_at: datetime.datetime) -> str:
    base, _ = os.path.splitext(save_path)
    txt_path = f"{base}.txt" if base else f"{save_path}.txt"

    duration_s = (finished_at - started_at).total_seconds()
    metrics_lines = [f"{k}={v:.6f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items()]

    content = "\n".join(
        [
            f"save_path: {os.path.abspath(save_path)}",
            f"started_at: {started_at.isoformat()}",
            f"finished_at: {finished_at.isoformat()}",
            f"duration_seconds: {duration_s:.3f}",
            "",
            f"model_type: {cfg.model_type}",
            f"train_cells: {cfg.train_cells}",
            f"test_cells: {cfg.test_cells}",
            "",
            "metrics:",
            "  " + "  ".join(metrics_lines) if metrics_lines else "  (empty)",
            "",
            "metrics_json:",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "",
            "config:",
            str(cfg),
            "",
        ]
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    return txt_path


def _train_epochwise(
    *,
    cfg: TrainConfig,
    model: nn.Module,
    train_loader: DataLoader,
    inputs_val: torch.Tensor,
    targets_val: torch.Tensor,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[nn.Module, Dict[str, Any]]:
    device = next(model.parameters()).device
    log_sigma_u = torch.zeros((), device=device)
    log_sigma_f = torch.zeros((), device=device)
    log_sigma_f_t = torch.zeros((), device=device)

    history: Dict[str, Any] = {
        "epoch": [],
        "lr": [],
        "loss_train": [],
        "loss_u_train": [],
        "loss_f_train": [],
        "loss_f_t_train": [],
        "loss_val": [],
        "loss_u_val": [],
        "loss_f_val": [],
        "loss_f_t_val": [],
        "mae_val": [],
        "mse_val": [],
        "rmspe_val": [],
        "r2_val": [],
    }

    # Use cudnn for better performance
    # torch.backends.cudnn.benchmark = True # Optional: might help if input sizes are constant
    with torch.backends.cudnn.flags(enabled=True):
        for epoch_idx in range(int(cfg.epochs)):
            if should_cancel is not None and should_cancel():
                raise TrainingCancelled()
            model.train()
            loss_sum = 0.0
            loss_u_sum = 0.0
            loss_f_sum = 0.0
            loss_f_t_sum = 0.0
            seen = 0

            for period_idx, (inputs_train_batch, targets_train_batch) in enumerate(train_loader):
                if should_cancel is not None and should_cancel():
                    raise TrainingCancelled()
                inputs_train_batch = inputs_train_batch.to(device)
                targets_train_batch = targets_train_batch.to(device)

                U_pred_train, F_pred_train, F_t_pred_train = model(inputs=inputs_train_batch)
                loss_raw = criterion(
                    outputs_U=U_pred_train,
                    targets_U=targets_train_batch,
                    outputs_F=F_pred_train,
                    outputs_F_t=F_t_pred_train,
                    log_sigma_u=log_sigma_u,
                    log_sigma_f=log_sigma_f,
                    log_sigma_f_t=log_sigma_f_t,
                )
                denom = max(1, int(targets_train_batch.numel()))
                loss = loss_raw / denom
                loss_u = criterion.loss_U / denom
                loss_f = criterion.loss_F / denom
                loss_f_t = criterion.loss_F_t / denom

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch_idx + 1} period={period_idx + 1}: "
                        f"loss={float(loss.detach().item())}"
                    )

                optimizer.zero_grad()
                loss.backward()
                if cfg.model_type == "DeepHPM":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_sum += float(loss.detach().item())
                loss_u_sum += float(loss_u.detach().item())
                loss_f_sum += float(loss_f.detach().item())
                loss_f_t_sum += float(loss_f_t.detach().item())
                seen += 1

            scheduler.step()

            if should_cancel is not None and should_cancel():
                raise TrainingCancelled()
            model.eval()
            U_pred_val, F_pred_val, F_t_pred_val = model(inputs=inputs_val)
            loss_val_raw = criterion(
                outputs_U=U_pred_val,
                targets_U=targets_val,
                outputs_F=F_pred_val,
                outputs_F_t=F_t_pred_val,
                log_sigma_u=log_sigma_u,
                log_sigma_f=log_sigma_f,
                log_sigma_f_t=log_sigma_f_t,
            )
            denom_val = max(1, int(targets_val.numel()))
            loss_val = loss_val_raw / denom_val
            loss_u_val = criterion.loss_U / denom_val
            loss_f_val = criterion.loss_F / denom_val
            loss_f_t_val = criterion.loss_F_t / denom_val

            if not torch.isfinite(loss_val):
                raise FloatingPointError(f"Non-finite val loss at epoch={epoch_idx + 1}: loss_val={float(loss_val.detach().item())}")

            U_pred_val_soh = 1.0 - U_pred_val
            targets_val_soh = 1.0 - targets_val
            m = _eval_metrics(U_pred_val_soh, targets_val_soh)

            avg_loss = loss_sum / max(1, seen)
            avg_loss_u = loss_u_sum / max(1, seen)
            avg_loss_f = loss_f_sum / max(1, seen)
            avg_loss_f_t = loss_f_t_sum / max(1, seen)

            history["epoch"].append(epoch_idx + 1)
            history["lr"].append(float(optimizer.param_groups[0].get("lr", 0.0)))
            history["loss_train"].append(avg_loss)
            history["loss_u_train"].append(avg_loss_u)
            history["loss_f_train"].append(avg_loss_f)
            history["loss_f_t_train"].append(avg_loss_f_t)
            history["loss_val"].append(float(loss_val.detach().item()))
            history["loss_u_val"].append(float(loss_u_val.detach().item()))
            history["loss_f_val"].append(float(loss_f_val.detach().item()))
            history["loss_f_t_val"].append(float(loss_f_t_val.detach().item()))
            history["mae_val"].append(float(m["MAE"]))
            history["mse_val"].append(float(m["MSE"]))
            history["rmspe_val"].append(float(m["RMSPE"]))
            history["r2_val"].append(float(m["R2"]))

            payload: Dict[str, Any] = {
                "epoch": epoch_idx + 1,
                "epochs": int(cfg.epochs),
                "lr": history["lr"][-1],
                "loss_train": avg_loss,
                "loss_u_train": avg_loss_u,
                "loss_f_train": avg_loss_f,
                "loss_f_t_train": avg_loss_f_t,
                "loss_val": history["loss_val"][-1],
                "loss_u_val": history["loss_u_val"][-1],
                "loss_f_val": history["loss_f_val"][-1],
                "loss_f_t_val": history["loss_f_t_val"][-1],
                "metrics_val": m,
            }
            if on_epoch_end is not None:
                on_epoch_end(payload)

            print(
                f"Epoch: {epoch_idx + 1}, "
                f"Loss: {avg_loss:.6f}, Loss_U: {avg_loss_u:.6f}, Loss_F: {avg_loss_f:.6f}, Loss_F_t: {avg_loss_f_t:.6f}, "
                f"MAE: {m['MAE']:.6f}, MSE: {m['MSE']:.6f}, RMSPE: {m['RMSPE']:.6f}, R2: {m['R2']:.6f}"
            )

    return model, history


def train_unified(
    cfg: TrainConfig,
    *,
    on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    print(f"Starting training with model: {cfg.model_type}")
    started_at = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    _set_seed(cfg.seed)
    dev = _device(cfg.device)
    
    # 1. Load Data
    seq_len = 1 # 目前所有模型默认使用 seq_len=1
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
    
    # Target: Capacity Loss (PCL) -> We convert to SOH later
    targets_train = targets_dict["train"][:, :, 0:1].to(dev)
    targets_val = targets_dict["val"][:, :, 0:1].to(dev)
    targets_test = targets_dict["test"][:, :, 0:1].to(dev)

    # Standardize
    _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
    _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

    inputs_dim = inputs_train.shape[2]
    outputs_dim = 1

    # 2. Build Model
    model = None
    criterion = None
    loss_mode = "Baseline" # Default

    if cfg.model_type == "Baseline":
        # DataDrivenNN (wrapper around Neural_Net)
        model = func.DataDrivenNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=cfg.layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
        ).to(dev)
        # Replace surrogateNN with our CustomNeuralNet for flexible activation
        model.surrogateNN = CustomNeuralNet(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=cfg.layers,
            activation_name=cfg.activation
        ).to(dev)
        loss_mode = "Baseline"

    elif cfg.model_type == "BiLSTM":
        core_model = BiLSTMModel(
            input_dim=inputs_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=outputs_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        ).to(dev)
        model = BiLSTMWrapper(core_model).to(dev)
        loss_mode = "Baseline" # BiLSTM treats physics loss as 0

    elif cfg.model_type == "DeepHPM":
        # DeepHPMNN
        model = func.DeepHPMNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=cfg.layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
            inputs_dynamical=cfg.inputs_dynamical,
            inputs_dim_dynamical=str(cfg.inputs_dim_dynamical)
        ).to(dev)
        model.surrogateNN = CustomNeuralNet(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=cfg.layers,
            activation_name=cfg.activation
        ).to(dev)
        model.dynamicalNN = CustomNeuralNet(
            seq_len=seq_len,
            inputs_dim=cfg.inputs_dim_dynamical,
            outputs_dim=1,
            layers=cfg.layers,
            activation_name=cfg.activation
        ).to(dev)
        loss_mode = "Sum" # Use physics loss

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    # 3. Setup Optimizer & Loss
    optimizer = _build_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    criterion = func.My_loss(mode=loss_mode)

    # Placeholders for uncertainty weighting (used in AdpBal mode, ignored in others)
    log_sigma_u = torch.zeros((), device=dev)
    log_sigma_f = torch.zeros((), device=dev)
    log_sigma_f_t = torch.zeros((), device=dev)

    # 4. Data Loader
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    # 5. Training Loop
    # Disable cuDNN for reproducibility if needed (as per original code)
    torch.backends.cudnn.enabled = False
    try:
        model, results_epoch = _train_epochwise(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            inputs_val=inputs_val,
            targets_val=targets_val,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            should_cancel=should_cancel,
            on_epoch_end=on_epoch_end,
        )
    except TrainingCancelled:
        print("Training cancelled.")
        raise

    # 6. Evaluation
    model.eval()
    
    # Unpack model if wrapped (for BiLSTM we use the wrapper's output directly, 
    # but for saving/predicting we just call the model as is)
    U_pred_test, _, _ = model(inputs=inputs_test)
    
    # Convert PCL to SoH (1 - PCL)
    U_pred_test_soh = 1.0 - U_pred_test
    targets_test_soh = 1.0 - targets_test
    
    metrics = _eval_metrics(U_pred_test_soh, targets_test_soh)
    print("Test Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # 7. Save Results
    _ensure_results_dir(cfg.save_path)
    results = {
        "model_state_dict": model.state_dict(),  # 保存模型权重
        "U_true": targets_test_soh.detach().cpu().numpy().squeeze(),
        "U_pred": U_pred_test_soh.detach().cpu().numpy().squeeze(),
        "Cycles": inputs_test[:, :, -1:].detach().cpu().numpy().squeeze(),
        "metric": metrics,
        "config": str(cfg),
        "train_config": cfg.__dict__, # 保存完整配置字典，方便后续解析
        "history": results_epoch
    }
    
    # 保存 Scaler 信息 (如果存在)
    if 'mean_inputs_train' in locals():
        results['scaler_inputs'] = (mean_inputs_train, std_inputs_train)
        results['scaler_targets'] = (mean_targets_train, std_targets_train)

    torch.save(results, cfg.save_path)
    print(f"Results saved to: {cfg.save_path}")
    txt_path = _write_readable_log(cfg.save_path, cfg, metrics, started_at=started_at, finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))))
    print(f"Readable log saved to: {txt_path}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    
    # Config arguments with defaults from USER_CONFIG
    parser.add_argument("--model_type", type=str, default=USER_CONFIG["model_type"], choices=["Baseline", "BiLSTM", "DeepHPM"])
    
    parser.add_argument("--data_path", type=str, default=USER_CONFIG["data_path"])
    parser.add_argument("--save_path", type=str, default=USER_CONFIG["save_path"])
    parser.add_argument("--train_cells", type=str, default=USER_CONFIG["train_cells"])
    parser.add_argument("--test_cells", type=str, default=USER_CONFIG["test_cells"])
    parser.add_argument("--perc_val", type=float, default=USER_CONFIG["perc_val"])
    
    # Common Structure
    parser.add_argument("--layers", type=str, default=USER_CONFIG["layers"])
    parser.add_argument("--activation", type=str, default=USER_CONFIG["activation"])
    
    # BiLSTM Structure
    parser.add_argument("--hidden_dim", type=int, default=USER_CONFIG["hidden_dim"])
    parser.add_argument("--num_layers", type=int, default=USER_CONFIG["num_layers"])
    parser.add_argument("--dropout", type=float, default=USER_CONFIG["dropout"])
    
    # DeepHPM Structure
    parser.add_argument("--inputs_dynamical", type=str, default=USER_CONFIG["inputs_dynamical"])
    parser.add_argument("--inputs_dim_dynamical", type=int, default=USER_CONFIG["inputs_dim_dynamical"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=USER_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=USER_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=USER_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=USER_CONFIG["weight_decay"])
    parser.add_argument("--step_size", type=int, default=USER_CONFIG["step_size"])
    parser.add_argument("--gamma", type=float, default=USER_CONFIG["gamma"])
    parser.add_argument("--optimizer", type=str, default=USER_CONFIG["optimizer"])
    
    parser.add_argument("--device", type=str, default=USER_CONFIG["device"])
    parser.add_argument("--seed", type=int, default=USER_CONFIG["seed"])

    args = parser.parse_args(argv)

    save_path = args.save_path
    if not save_path:
        timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d%H%M%S")
        filename = f"{args.model_type}_{timestamp}.pth"
        save_path = os.path.join(ROOT, "results", filename)

    cfg = TrainConfig(
        model_type=args.model_type,
        data_path=args.data_path,
        save_path=save_path,
        train_cells=_parse_int_list(args.train_cells),
        test_cells=_parse_int_list(args.test_cells),
        perc_val=args.perc_val,
        layers=_parse_int_list(args.layers),
        activation=args.activation,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        inputs_dynamical=args.inputs_dynamical,
        inputs_dim_dynamical=args.inputs_dim_dynamical,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        optimizer=args.optimizer,
        device=args.device,
        seed=args.seed,
    )

    train_unified(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
