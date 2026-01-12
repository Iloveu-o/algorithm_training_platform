# -*- coding: utf-8 -*-

import argparse
import datetime
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import functions as func


@dataclass
class LoadedArtifact:
    model_type: str
    config: Dict[str, Any]
    model_state_dict: Dict[str, Any]
    scaler_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]]
    scaler_targets: Optional[Tuple[torch.Tensor, torch.Tensor]]


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)


def _safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_model_path(model_path: str) -> str:
    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        candidates = []
        for name in os.listdir(model_path):
            if name.lower().endswith(".pth"):
                candidates.append(os.path.join(model_path, name))
        if not candidates:
            raise FileNotFoundError(f"no .pth files found in directory: {model_path}")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    return model_path


def _load_local_training_artifact(model_path: str) -> LoadedArtifact:
    artifact = _safe_torch_load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("invalid artifact format: expected dict")

    cfg = artifact.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("invalid artifact: missing 'config' dict")

    model_type = str(cfg.get("model_type") or "").strip()
    if model_type not in {"Baseline", "BiLSTM", "DeepHPM"}:
        raise ValueError(f"invalid model_type in artifact config: {model_type}")

    sd = artifact.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError("invalid artifact: missing 'model_state_dict' dict")

    scaler_inputs = artifact.get("scaler_inputs")
    scaler_targets = artifact.get("scaler_targets")
    si = None
    st = None
    if isinstance(scaler_inputs, tuple) and len(scaler_inputs) == 2:
        si = (torch.as_tensor(scaler_inputs[0]), torch.as_tensor(scaler_inputs[1]))
    if isinstance(scaler_targets, tuple) and len(scaler_targets) == 2:
        st = (torch.as_tensor(scaler_targets[0]), torch.as_tensor(scaler_targets[1]))

    return LoadedArtifact(
        model_type=model_type,
        config=cfg,
        model_state_dict=sd,
        scaler_inputs=si,
        scaler_targets=st,
    )


def _apply_neural_net_patch(activation: str) -> None:
    class PatchedNeuralNet(nn.Module):
        def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation="Tanh"):
            super().__init__()

            act = activation_name
            if act not in ("Tanh", "Sin"):
                raise ValueError(f"Unsupported activation: {act}")

            self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim

            blocks = []
            blocks.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
            nn.init.xavier_normal_(blocks[-1].weight)
            if act == "Tanh":
                blocks.append(nn.Tanh())
            else:
                blocks.append(func.Sin())
            blocks.append(nn.Dropout(p=0.2))

            for i in range(len(layers) - 1):
                blocks.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
                nn.init.xavier_normal_(blocks[-1].weight)
                if act == "Tanh":
                    blocks.append(nn.Tanh())
                else:
                    blocks.append(func.Sin())
                blocks.append(nn.Dropout(p=0.2))

            blocks.append(nn.Linear(in_features=layers[-1], out_features=outputs_dim))
            nn.init.xavier_normal_(blocks[-1].weight)
            self.NN = nn.Sequential(*blocks)

        def forward(self, x):
            self.x = x
            self.x.requires_grad_(True)
            self.x_2D = self.x.contiguous().view((-1, self.inputs_dim))
            NN_out_2D = self.NN(self.x_2D)
            self.u_pred = NN_out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))
            return self.u_pred

    activation_name = activation
    func.Neural_Net = PatchedNeuralNet


def _build_model_from_local_artifact(la: LoadedArtifact, device: torch.device) -> nn.Module:
    cfg = la.config
    activation = str(cfg.get("activation") or "Tanh")
    _apply_neural_net_patch(activation)

    if la.model_type in {"Baseline", "DeepHPM"}:
        if la.scaler_inputs is None or la.scaler_targets is None:
            raise ValueError("artifact missing scalers for Baseline/DeepHPM")

    seq_len = int(cfg.get("seq_len") or 1)

    if la.model_type == "BiLSTM":
        input_dim = int(cfg.get("inputs_dim") or 0)
        if input_dim <= 0:
            raise ValueError("BiLSTM artifact config missing 'inputs_dim' (>0)")
        hidden_dim = int(cfg.get("lstm_hidden_dim") or 128)
        num_layers = int(cfg.get("lstm_layers") or 1)
        model = BiLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
        model.load_state_dict(la.model_state_dict, strict=True)
        model.to(device).eval()
        return model

    layers = list(cfg.get("hidden_layers") or (int(cfg.get("num_layers") or 1) * [int(cfg.get("num_neurons") or 128)]))
    if not layers:
        raise ValueError("invalid hidden_layers: empty")

    mean_in, std_in = la.scaler_inputs
    mean_tg, std_tg = la.scaler_targets
    inputs_dim = int(mean_in.numel())

    if la.model_type == "Baseline":
        model = func.DataDrivenNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=1,
            layers=layers,
            scaler_inputs=(mean_in.to(device), std_in.to(device)),
            scaler_targets=(mean_tg.to(device), std_tg.to(device)),
        )
        model.load_state_dict(la.model_state_dict, strict=True)
        model.to(device).eval()
        return model

    inputs_dynamical = str(cfg.get("inputs_dynamical") or "U")
    inputs_dim_dynamical = int(cfg.get("inputs_dim_dynamical") or 1)
    model = func.DeepHPMNN(
        seq_len=seq_len,
        inputs_dim=inputs_dim,
        outputs_dim=1,
        layers=layers,
        scaler_inputs=(mean_in.to(device), std_in.to(device)),
        scaler_targets=(mean_tg.to(device), std_tg.to(device)),
        inputs_dynamical=inputs_dynamical,
        inputs_dim_dynamical=str(inputs_dim_dynamical),
    )
    model.load_state_dict(la.model_state_dict, strict=True)
    model.to(device).eval()
    return model


def _extract_cell_series(data: func.SeversonBattery, cell_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx_true = int(cell_id) - 1
    inputs_tmp = None
    targets_tmp = None
    if idx_true in data.idx_train_units:
        idx_tmp = int((data.idx_train_units == idx_true).nonzero()[0][0])
        inputs_tmp = data.inputs_train_slices[idx_tmp]
        targets_tmp = data.targets_train_slices[idx_tmp]
    elif idx_true in data.idx_val_units:
        idx_tmp = int((data.idx_val_units == idx_true).nonzero()[0][0])
        inputs_tmp = data.inputs_val_slices[idx_tmp]
        targets_tmp = data.targets_val_slices[idx_tmp]
    elif idx_true in data.idx_test_units:
        idx_tmp = int((data.idx_test_units == idx_true).nonzero()[0][0])
        inputs_tmp = data.inputs_test_slices[idx_tmp]
        targets_tmp = data.targets_test_slices[idx_tmp]
    if inputs_tmp is None or targets_tmp is None:
        raise ValueError(f"cell_id not found in dataset: {cell_id}")
    inputs_t = torch.from_numpy(inputs_tmp).type(torch.float32)
    targets_t = torch.from_numpy(targets_tmp).type(torch.float32)
    return inputs_t, targets_t


def _compute_rul_curve(cycles: torch.Tensor, soh: torch.Tensor, threshold_soh: float) -> torch.Tensor:
    cyc = cycles.detach().view(-1)
    s = soh.detach().view(-1)
    if cyc.numel() == 0:
        return torch.zeros_like(cyc)
    crossed = (s <= float(threshold_soh)).nonzero()
    if crossed.numel() > 0:
        eol_cycle = cyc[int(crossed[0].item())]
    else:
        eol_cycle = cyc[-1]
    rul = eol_cycle - cyc
    return torch.clamp(rul, min=0.0)


def _eval_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    p = pred.detach().view(-1).float()
    t = true.detach().view(-1).float()
    mae = torch.mean(torch.abs(p - t)).item()
    mse = torch.mean((p - t) ** 2).item()
    rmspe = torch.sqrt(torch.mean(((p - t) / torch.clamp(t, min=1e-8)) ** 2)).item()
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - torch.mean(t)) ** 2)
    r2 = (1 - ss_res / torch.clamp(ss_tot, min=1e-12)).item()
    return {"MAE": float(mae), "MSE": float(mse), "RMSPE": float(rmspe), "R2": float(r2)}


def _plot_and_save(out_path: str, cycles: np.ndarray, true: np.ndarray, pred: np.ndarray, title: str, y_label: str) -> None:
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(cycles, true, label="True")
    plt.plot(cycles, pred, label="Pred", linestyle="--")
    plt.xlabel("Cycle")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(__file__), "local_results"))
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT, "SeversonBattery.mat"))
    parser.add_argument("--cell_id", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--threshold_soh", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "local_test_outputs"))
    args = parser.parse_args(argv)

    model_path = _resolve_model_path(args.model_path)
    la = _load_local_training_artifact(model_path)

    dev = torch.device(args.device)
    data = func.SeversonBattery(args.data_path, seq_len=int(la.config.get("seq_len") or 1))
    inputs_t, targets_t = _extract_cell_series(data, int(args.cell_id))

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{la.model_type}_cell{int(args.cell_id)}_{run_id}"

    model = _build_model_from_local_artifact(la, dev)
    inputs_dev = inputs_t.to(dev)
    targets_dev = targets_t.to(dev)

    with torch.no_grad():
        if la.model_type == "BiLSTM":
            pcl_pred = model(inputs_dev).detach().cpu().view(-1)
        else:
            pcl_pred, _, _ = model(inputs=inputs_dev)
            pcl_pred = pcl_pred.detach().cpu().view(-1)

    cycles = inputs_t[:, :, -1].detach().cpu().view(-1)
    pcl_true = targets_t[:, :, 0].detach().cpu().view(-1)
    soh_pred = 1.0 - pcl_pred
    soh_true = 1.0 - pcl_true

    rul_true = None
    if targets_t.shape[-1] >= 2:
        rul_true = targets_t[:, :, 1].detach().cpu().view(-1)
    rul_pred = _compute_rul_curve(cycles, soh_pred, float(args.threshold_soh))

    step = max(1, int(args.step))
    idx = slice(None, None, step)

    metrics_pcl = _eval_metrics(pcl_pred, pcl_true)
    metrics_soh = _eval_metrics(soh_pred, soh_true)
    metrics_rul = _eval_metrics(rul_pred, rul_true) if rul_true is not None else None

    out_pcl = os.path.join(args.out_dir, f"{base}_pcl.png")
    out_soh = os.path.join(args.out_dir, f"{base}_soh.png")
    out_rul = os.path.join(args.out_dir, f"{base}_rul.png")

    cyc_np = cycles[idx].numpy()
    _plot_and_save(out_pcl, cyc_np, pcl_true[idx].numpy(), pcl_pred[idx].numpy(), f"PCL Curve ({la.model_type})", "PCL")
    _plot_and_save(out_soh, cyc_np, soh_true[idx].numpy(), soh_pred[idx].numpy(), f"SOH Curve ({la.model_type})", "SOH")
    if rul_true is not None:
        _plot_and_save(out_rul, cyc_np, rul_true[idx].numpy(), rul_pred[idx].numpy(), f"RUL Curve ({la.model_type})", "RUL")

    report = {
        "model_path": model_path,
        "model_type": la.model_type,
        "cell_id": int(args.cell_id),
        "step": step,
        "threshold_soh": float(args.threshold_soh),
        "metrics_pcl": metrics_pcl,
        "metrics_soh": metrics_soh,
        "metrics_rul": metrics_rul,
        "plots": {
            "pcl": out_pcl,
            "soh": out_soh,
            "rul": out_rul if rul_true is not None else None,
        },
    }

    txt_path = os.path.join(args.out_dir, f"{base}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print(f"model_path: {model_path}")
    print(f"model_type: {la.model_type}")
    print(f"cell_id: {int(args.cell_id)}  step: {step}  threshold_soh: {float(args.threshold_soh)}")
    print(f"metrics_pcl: {metrics_pcl}")
    print(f"metrics_soh: {metrics_soh}")
    if metrics_rul is not None:
        print(f"metrics_rul: {metrics_rul}")
    print(f"saved: {out_pcl}")
    print(f"saved: {out_soh}")
    if rul_true is not None:
        print(f"saved: {out_rul}")
    print(f"saved: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

