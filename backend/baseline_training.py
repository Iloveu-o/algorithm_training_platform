import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


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
    "save_path": os.path.join(ROOT, "results", "SoH_CaseA_Baseline.pth"), # 结果保存路径
    "settings_path": os.path.join(ROOT, "Settings", "settings_SoH_CaseA.pth"), # 默认设置文件路径(优先级低于本配置)

    # 2. 数据集划分
    "train_cells": "91,100",  # 训练集电池编号
    "test_cells": "124",      # 测试集电池编号
    "perc_val": 0.2,          # 验证集比例

    # 3. 网络结构参数
    "layers": "32,32",        # 隐藏层结构：逗号分隔，表示每层的节点数。例如 "32,32" 表示两层，每层32个节点
    "activation": "Tanh",     # 激活函数：支持 "Tanh", "ReLU", "Sigmoid", "LeakyReLU", "Sin"

    # 4. 训练超参数
    "epochs": 100,            # 训练轮数
    "batch_size": 128,        # 批大小
    "lr": 1e-3,               # 初始学习率
    "weight_decay": 0.0,      # 权重衰减 (L2正则化)

    # 5. 优化器与学习率调度
    "optimizer": "SGD",      # 优化器选择：支持 "Adam", "SGD"
    "step_size": 50,          # 学习率衰减步长 (每多少个epoch衰减一次)
    "gamma": 0.1,             # 学习率衰减率

    # 6. 其他
    "device": "cuda",         # 设备："auto", "cpu", "cuda"
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


def _parse_layers(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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


def _eval_metrics(pred_soh: torch.Tensor, true_soh: torch.Tensor) -> dict[str, float]:
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


@dataclass
class TrainConfig:
    data_path: str
    settings_path: str
    seq_len: int
    perc_val: float
    train_cells: list[int]
    test_cells: list[int]
    epochs: int | None
    batch_size: int | None
    lr: float | None
    optimizer: str
    weight_decay: float
    step_size: int | None
    gamma: float | None
    layers: list[int] | None
    activation: str
    device: str
    seed: int
    save_path: str


def _load_settings(path: str) -> dict:
    if os.path.exists(path):
        return torch.load(path)
    return {}


def _ensure_results_dir(save_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def train_baseline(cfg: TrainConfig) -> dict[str, dict[str, float]]:
    _set_seed(cfg.seed)
    dev = _device(cfg.device)

    settings = _load_settings(cfg.settings_path)
    epochs = int(cfg.epochs if cfg.epochs is not None else settings.get("num_epoch", 100))
    batch_size = int(cfg.batch_size if cfg.batch_size is not None else settings.get("batch_size", 128))
    lr = float(cfg.lr if cfg.lr is not None else settings.get("lr", 1e-3))
    step_size = int(cfg.step_size if cfg.step_size is not None else settings.get("step_size", 50))
    gamma = float(cfg.gamma if cfg.gamma is not None else settings.get("gamma", 0.1))

    if cfg.layers is None:
        num_layers_list = settings.get("num_layers", [2])
        num_neurons_list = settings.get("num_neurons", [32])
        num_layers = int(num_layers_list[0]) if len(num_layers_list) > 0 else 2
        num_neurons = int(num_neurons_list[0]) if len(num_neurons_list) > 0 else 32
        layers = num_layers * [num_neurons]
    else:
        layers = cfg.layers

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
    targets_train = targets_dict["train"][:, :, 0:1].to(dev)
    targets_val = targets_dict["val"][:, :, 0:1].to(dev)
    targets_test = targets_dict["test"][:, :, 0:1].to(dev)

    _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode="fit")
    _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode="fit")

    model = func.DataDrivenNN(
        seq_len=cfg.seq_len,
        inputs_dim=inputs_train.shape[2],
        outputs_dim=1,
        layers=layers,
        scaler_inputs=(mean_inputs_train, std_inputs_train),
        scaler_targets=(mean_targets_train, std_targets_train),
    ).to(dev)

    if cfg.activation.strip().lower() in {"tanh", "sin"}:
        model.surrogateNN = func.Neural_Net(
            seq_len=cfg.seq_len,
            inputs_dim=inputs_train.shape[2],
            outputs_dim=1,
            layers=layers,
            activation=cfg.activation,
        ).to(dev)
    else:
        raise ValueError(f"Unsupported activation for functions.Neural_Net: {cfg.activation}")

    optimizer = _build_optimizer(cfg.optimizer, model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = func.My_loss(mode="Baseline")

    log_sigma_u = torch.zeros((), device=dev)
    log_sigma_f = torch.zeros((), device=dev)
    log_sigma_f_t = torch.zeros((), device=dev)

    train_set = func.TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model, _ = func.train(
        num_epoch=epochs,
        batch_size=batch_size,
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
        log_sigma_f_t=log_sigma_f_t,
    )

    model.eval()
    pred_train, _, _ = model(inputs=inputs_train)
    pred_val, _, _ = model(inputs=inputs_val)
    pred_test, _, _ = model(inputs=inputs_test)

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

    _ensure_results_dir(cfg.save_path)
    results = {
        "U_true": true_test_soh.detach().cpu().numpy().squeeze(),
        "U_pred": pred_test_soh.detach().cpu().numpy().squeeze(),
        "U_t_pred": getattr(model, "U_t").detach().cpu().numpy().squeeze() if hasattr(model, "U_t") else None,
        "Cycles": inputs_test[:, :, -1:].detach().cpu().numpy().squeeze(),
        "metric": report,
        "layers": layers,
        "settings_path": cfg.settings_path,
    }
    torch.save(results, cfg.save_path)

    return report


def _print_report(report: dict[str, dict[str, float]]) -> None:
    for split in ["train", "val", "test"]:
        m = report[split]
        print(f"{split.upper()}  RMSPE={m['RMSPE']:.6f}  MSE={m['MSE']:.6f}  R2={m['R2']:.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--layers", type=str, default=USER_CONFIG["layers"])
    parser.add_argument("--activation", type=str, default=USER_CONFIG["activation"], choices=["Tanh", "Sin", "ReLU", "Sigmoid", "LeakyReLU"])

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
        layers=_parse_layers(args.layers) if args.layers is not None else None,
        activation=args.activation,
        device=args.device,
        seed=args.seed,
        save_path=args.save_path,
    )

    report = train_baseline(cfg)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
