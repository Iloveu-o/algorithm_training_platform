# -*- coding: utf-8 -*-
import csv
import datetime
import os
import threading
import traceback
import uuid
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import unified_training as ut
import functions as func


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _project_root()
RESULTS_DIR = os.path.join(ROOT, "results")


def _now_cn() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))


def _parse_int_list(value: Union[str, List[int], None]) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
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


def _safe_save_path(model_type: str) -> str:
    timestamp = _now_cn().strftime("%Y%m%d%H%M%S")
    filename = f"{model_type}_{timestamp}.pth"
    return os.path.join(RESULTS_DIR, filename)


def _artifact_urls(save_path: str) -> Dict[str, Optional[str]]:
    pth_name = os.path.basename(save_path)
    base, _ = os.path.splitext(pth_name)
    txt_name = f"{base}.txt"

    pth_url = f"/results/{pth_name}" if pth_name else None
    txt_url = f"/results/{txt_name}" if txt_name else None

    return {"pth": pth_url, "txt": txt_url}


def _cleanup_artifacts(save_path: Optional[str]) -> None:
    if not save_path:
        return
    try:
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception:
        pass
    try:
        base, _ = os.path.splitext(save_path)
        txt_path = f"{base}.txt" if base else ""
        if txt_path and os.path.exists(txt_path):
            os.remove(txt_path)
    except Exception:
        pass


class UnifiedTrainRequest(BaseModel):
    model_type: str
    epochs: Optional[int] = None
    lr: Optional[float] = None
    optimizer: Optional[str] = None
    layers: Optional[Union[str, List[int]]] = None
    activation: Optional[str] = None
    batch_size: Optional[int] = None

    model_config = {"extra": "forbid"}


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    canceled_at: Optional[str] = None
    cancel_requested: Optional[bool] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    progress: Optional[Dict[str, Any]] = None
    history: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    save_path: Optional[str] = None
    artifacts: Optional[Dict[str, Optional[str]]] = None


_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _resolve_results_file(ref: str) -> str:
    s = str(ref or "").strip()
    if not s:
        raise ValueError("model_file is required")
    if s.startswith("/results/"):
        s = s[len("/results/") :]
    base = os.path.basename(s)
    if not base:
        raise ValueError("invalid model_file")
    abs_path = os.path.join(RESULTS_DIR, base)
    if not os.path.exists(abs_path):
        raise ValueError(f"model_file not found: {base}")
    if not abs_path.lower().endswith(".pth"):
        raise ValueError("model_file must be a .pth file")
    return abs_path


def _list_result_pth_files() -> List[str]:
    if not os.path.exists(RESULTS_DIR):
        return []
    out: List[str] = []
    for name in os.listdir(RESULTS_DIR):
        if name.lower().endswith(".pth"):
            out.append(name)
    out.sort(reverse=True)
    return out


def _eval_metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    eps = 1e-8
    pred_ = pred.detach()
    y_ = y.detach()
    mae = torch.mean(torch.abs(pred_ - y_)).item()
    mse = torch.mean((pred_ - y_) ** 2).item()
    rmspe = torch.sqrt(torch.mean(((pred_ - y_) / (y_ + eps)) ** 2)).item()
    ss_res = torch.sum((y_ - pred_) ** 2)
    ss_tot = torch.sum((y_ - torch.mean(y_)) ** 2)
    r2 = (1 - ss_res / (ss_tot + eps)).item()
    return {"MAE": float(mae), "MSE": float(mse), "RMSPE": float(rmspe), "R2": float(r2)}


def _extract_cell_series(data: func.SeversonBattery, cell_id: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _build_model_from_artifact(artifact: Dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    train_cfg = artifact.get("train_config")
    if not isinstance(train_cfg, dict):
        raise ValueError("该模型文件缺少配置信息，无法用于预测。如果是旧版本生成的模型，请尝试重新训练。")

    model_type = str(artifact.get("model_type") or train_cfg.get("model_type") or "").strip()
    if model_type not in {"Baseline", "BiLSTM", "DeepHPM"}:
        raise ValueError(f"unsupported model_type in artifact: {model_type}")

    sd = artifact.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError("该模型文件不包含模型权重，无法用于预测。请重新训练模型。")

    scaler_inputs = artifact.get("scaler_inputs") or {}
    scaler_targets = artifact.get("scaler_targets") or {}
    mean_in = scaler_inputs.get("mean")
    std_in = scaler_inputs.get("std")
    mean_tg = scaler_targets.get("mean")
    std_tg = scaler_targets.get("std")

    if not torch.is_tensor(mean_in) or not torch.is_tensor(std_in):
        raise ValueError("该模型文件缺少输入归一化参数，无法用于预测。请重新训练模型。")
    if not torch.is_tensor(mean_tg) or not torch.is_tensor(std_tg):
        raise ValueError("该模型文件缺少目标归一化参数，无法用于预测。请重新训练模型。")

    # Ensure scalers are on the correct device
    mean_in = mean_in.to(device)
    std_in = std_in.to(device)
    mean_tg = mean_tg.to(device)
    std_tg = std_tg.to(device)

    # Update scaler dictionaries with device-moved tensors
    scaler_inputs_dev = {"mean": mean_in, "std": std_in}
    scaler_targets_dev = {"mean": mean_tg, "std": std_tg}

    seq_len = 1
    inputs_dim = int(mean_in.numel())
    outputs_dim = 1
    layers = [int(x) for x in (train_cfg.get("layers") or [])]
    activation = str(train_cfg.get("activation") or "Tanh")

    if model_type == "Baseline":
        model = func.DataDrivenNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_in, std_in),
            scaler_targets=(mean_tg, std_tg),
        )
        model.surrogateNN = ut.CustomNeuralNet(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            activation_name=activation,
        )
    elif model_type == "BiLSTM":
        hidden_dim = int(train_cfg.get("hidden_dim") or 32)
        num_layers = int(train_cfg.get("num_layers") or 2)
        dropout = float(train_cfg.get("dropout") or 0.0)
        core = ut.BiLSTMModel(
            input_dim=inputs_dim,
            hidden_dim=hidden_dim,
            output_dim=outputs_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = ut.BiLSTMWrapper(core)
    else:
        inputs_dynamical = str(train_cfg.get("inputs_dynamical") or "U")
        inputs_dim_dynamical = int(train_cfg.get("inputs_dim_dynamical") or 1)
        model = func.DeepHPMNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_in, std_in),
            scaler_targets=(mean_tg, std_tg),
            inputs_dynamical=inputs_dynamical,
            inputs_dim_dynamical=str(inputs_dim_dynamical),
        )
        model.surrogateNN = ut.CustomNeuralNet(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            activation_name=activation,
        )
        model.dynamicalNN = ut.CustomNeuralNet(
            seq_len=seq_len,
            inputs_dim=inputs_dim_dynamical,
            outputs_dim=1,
            layers=layers,
            activation_name=activation,
        )

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, train_cfg, scaler_inputs_dev, scaler_targets_dev


def _create_cfg_from_request(req: UnifiedTrainRequest) -> ut.TrainConfig:
    cfg_dict: Dict[str, Any] = dict(ut.USER_CONFIG)

    model_type = str(req.model_type).strip()
    if model_type not in {"Baseline", "BiLSTM", "DeepHPM"}:
        raise ValueError("model_type must be one of: Baseline, BiLSTM, DeepHPM")

    save_path = _safe_save_path(model_type)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    data_path = cfg_dict.get("data_path")
    if not data_path:
        raise ValueError("data_path is required")

    train_cells = _parse_int_list(cfg_dict.get("train_cells"))
    test_cells = _parse_int_list(cfg_dict.get("test_cells"))
    if not train_cells:
        raise ValueError("train_cells is required in backend USER_CONFIG")
    if not test_cells:
        raise ValueError("test_cells is required in backend USER_CONFIG")

    if req.layers is not None:
        cfg_dict["layers"] = req.layers
    if req.activation is not None:
        cfg_dict["activation"] = req.activation
    if req.epochs is not None:
        cfg_dict["epochs"] = req.epochs
    if req.lr is not None:
        cfg_dict["lr"] = req.lr
    if req.optimizer is not None:
        cfg_dict["optimizer"] = req.optimizer
    if req.batch_size is not None:
        cfg_dict["batch_size"] = req.batch_size

    layers = _parse_int_list(cfg_dict.get("layers"))

    return ut.TrainConfig(
        model_type=model_type,
        data_path=str(data_path),
        save_path=save_path,
        train_cells=train_cells,
        test_cells=test_cells,
        perc_val=float(cfg_dict.get("perc_val", 0.2)),
        layers=layers,
        activation=str(cfg_dict.get("activation", "Tanh")),
        hidden_dim=int(cfg_dict.get("hidden_dim", 32)),
        num_layers=int(cfg_dict.get("num_layers", 2)),
        dropout=float(cfg_dict.get("dropout", 0.0)),
        inputs_dynamical=str(cfg_dict.get("inputs_dynamical", "U")),
        inputs_dim_dynamical=int(cfg_dict.get("inputs_dim_dynamical", 1)),
        epochs=int(cfg_dict.get("epochs", 100)),
        batch_size=int(cfg_dict.get("batch_size", 128)),
        lr=float(cfg_dict.get("lr", 1e-3)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.0)),
        step_size=int(cfg_dict.get("step_size", 50)),
        gamma=float(cfg_dict.get("gamma", 0.1)),
        optimizer=str(cfg_dict.get("optimizer", "SGD")),
        device=str(cfg_dict.get("device", "auto")),
        seed=int(cfg_dict.get("seed", 1234)),
    )


def _run_job(job_id: str, cfg: ut.TrainConfig) -> None:
    started_at = _now_cn()
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if job.get("status") == "canceled" or job.get("cancel_requested"):
            finished_at = _now_cn()
            _cleanup_artifacts(cfg.save_path)
            job["status"] = "canceled"
            job["finished_at"] = finished_at.isoformat()
            job["canceled_at"] = finished_at.isoformat()
            job["save_path"] = None
            job["artifacts"] = None
            job["metrics"] = None
            job["error"] = None
            return

        job["status"] = "running"
        job["started_at"] = started_at.isoformat()
        job["save_path"] = cfg.save_path
        job["artifacts"] = _artifact_urls(cfg.save_path)

    try:
        def _should_cancel() -> bool:
            with _LOCK:
                j = _JOBS.get(job_id)
                if j is None:
                    return True
                return bool(j.get("cancel_requested")) or str(j.get("status")) == "canceling"

        def _on_epoch_end(payload: Dict[str, Any]) -> None:
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is None:
                    return
                job["progress"] = payload
                h = job.get("history")
                if not isinstance(h, dict):
                    h = {}
                    job["history"] = h

                def _append(key: str, value: Any) -> None:
                    arr = h.get(key)
                    if not isinstance(arr, list):
                        arr = []
                        h[key] = arr
                    arr.append(value)

                _append("epoch", payload.get("epoch"))
                _append("lr", payload.get("lr"))
                _append("loss_train", payload.get("loss_train"))
                _append("loss_u_train", payload.get("loss_u_train"))
                _append("loss_f_train", payload.get("loss_f_train"))
                _append("loss_f_t_train", payload.get("loss_f_t_train"))
                _append("loss_val", payload.get("loss_val"))
                _append("loss_u_val", payload.get("loss_u_val"))
                _append("loss_f_val", payload.get("loss_f_val"))
                _append("loss_f_t_val", payload.get("loss_f_t_val"))
                metrics_val = payload.get("metrics_val") or {}
                _append("mae_val", metrics_val.get("MAE"))
                _append("mse_val", metrics_val.get("MSE"))
                _append("rmspe_val", metrics_val.get("RMSPE"))
                _append("r2_val", metrics_val.get("R2"))

        ut.train_unified(cfg, on_epoch_end=_on_epoch_end, should_cancel=_should_cancel)
        finished_at = _now_cn()
        metrics: Optional[Dict[str, Any]] = None
        try:
            results = torch.load(cfg.save_path, map_location="cpu")
            metrics = results.get("metric")
        except Exception:
            metrics = None

        with _LOCK:
            _JOBS[job_id]["status"] = "succeeded"
            _JOBS[job_id]["finished_at"] = finished_at.isoformat()
            _JOBS[job_id]["metrics"] = metrics
    except ut.TrainingCancelled:
        finished_at = _now_cn()
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["status"] = "canceled"
                job["finished_at"] = finished_at.isoformat()
                job["canceled_at"] = finished_at.isoformat()
                job["cancel_requested"] = True
                job["metrics"] = None
                job["error"] = None
                job["save_path"] = None
                job["artifacts"] = None
        _cleanup_artifacts(cfg.save_path)
    except Exception as e:
        finished_at = _now_cn()
        err = f"{e}\n{traceback.format_exc()}"
        with _LOCK:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["finished_at"] = finished_at.isoformat()
            _JOBS[job_id]["error"] = err


router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("/unified", response_model=TrainJobResponse)
async def start_unified_training(req: UnifiedTrainRequest) -> TrainJobResponse:
    try:
        cfg = _create_cfg_from_request(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = uuid.uuid4().hex
    created_at = _now_cn().isoformat()
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "canceled_at": None,
        "cancel_requested": False,
        "error": None,
        "config": {
            "model_type": cfg.model_type,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "optimizer": cfg.optimizer,
            "layers": cfg.layers if cfg.model_type in {"Baseline", "DeepHPM"} else None,
            "activation": cfg.activation if cfg.model_type in {"Baseline", "DeepHPM"} else None,
        },
        "progress": None,
        "history": {
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
        },
        "metrics": None,
        "save_path": cfg.save_path,
        "artifacts": _artifact_urls(cfg.save_path),
    }

    with _LOCK:
        _JOBS[job_id] = job

    t = threading.Thread(target=_run_job, args=(job_id, cfg), daemon=True)
    t.start()

    return TrainJobResponse(**job)


@router.get("/unified/{job_id}", response_model=TrainJobResponse)
async def get_unified_training_job(job_id: str) -> TrainJobResponse:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return TrainJobResponse(**job)


@router.post("/unified/{job_id}/cancel", response_model=TrainJobResponse)
async def cancel_unified_training_job(job_id: str) -> TrainJobResponse:
    now = _now_cn().isoformat()
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        status = str(job.get("status") or "")
        if status in {"succeeded", "failed", "canceled"}:
            raise HTTPException(status_code=400, detail=f"job already finished: {status}")

        job["cancel_requested"] = True

        if status == "queued":
            job["status"] = "canceled"
            job["finished_at"] = now
            job["canceled_at"] = now
            job["metrics"] = None
            job["error"] = None
            job["save_path"] = None
            job["artifacts"] = None
        else:
            job["status"] = "canceling"

        save_path = job.get("save_path")

    _cleanup_artifacts(save_path)
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return TrainJobResponse(**job)


predict_router = APIRouter(prefix="/api/predict", tags=["predict"])


class PredictRequest(BaseModel):
    model_file: str
    cells: Union[str, List[int]]
    step: int = 1
    threshold_soh: float = 0.8
    device: Optional[str] = "cpu"
    data_path: Optional[str] = None

    model_config = {"extra": "forbid"}


@predict_router.get("/models")
async def list_models() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for name in _list_result_pth_files():
        abs_path = os.path.join(RESULTS_DIR, name)
        info: Dict[str, Any] = {"file": name, "url": f"/results/{name}", "model_type": None, "has_weights": False}
        try:
            # PyTorch 2.6+ safe load workaround
            try:
                artifact = torch.load(abs_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                artifact = torch.load(abs_path, map_location="cpu")
            
            if isinstance(artifact, dict):
                info["model_type"] = artifact.get("model_type") or None
                info["has_weights"] = isinstance(artifact.get("model_state_dict"), dict)
        except Exception:
            pass
        items.append(info)
    return {"models": items}


@predict_router.post("/run")
async def run_predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        abs_model_path = _resolve_results_file(req.model_file)
        
        # PyTorch 2.6+ safe load workaround
        try:
            artifact = torch.load(abs_model_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            artifact = torch.load(abs_model_path, map_location="cpu")

        if not isinstance(artifact, dict):
            raise ValueError("invalid model artifact")
        dev = ut._device(str(req.device or "cpu"))
        model, train_cfg, scaler_inputs, scaler_targets = _build_model_from_artifact(artifact, dev)
        data_path = str(req.data_path or train_cfg.get("data_path") or ut.USER_CONFIG.get("data_path") or "").strip()
        if not data_path:
            raise ValueError("data_path is required")
        data = func.SeversonBattery(data_path, seq_len=1)
        cell_ids = _parse_int_list(req.cells)
        if not cell_ids:
            raise ValueError("cells is required")
        step = max(1, int(req.step))
        threshold_soh = float(req.threshold_soh)

        results_cells: List[Dict[str, Any]] = []
        for cid in cell_ids:
            inputs_t, targets_t = _extract_cell_series(data, cid)
            inputs_dev = inputs_t.to(dev)
            with torch.no_grad():
                if artifact.get("model_type") == "BiLSTM":
                    mean_in = scaler_inputs["mean"]
                    std_in = scaler_inputs["std"]
                    mean_tg = scaler_targets["mean"]
                    std_tg = scaler_targets["std"]

                    inputs_norm, _, _ = func.standardize_tensor(inputs_dev, mode='transform', mean=mean_in, std=std_in)
                    u_pred_norm, _, _ = model(inputs=inputs_norm)
                    u_pred = func.inverse_standardize_tensor(u_pred_norm, mean=mean_tg, std=std_tg)
                else:
                    u_pred, _, _ = model(inputs=inputs_dev)
            pcl_pred = u_pred.detach().cpu().view(-1)
            cycles = inputs_t[:, :, -1].detach().cpu().view(-1)
            pcl_true = targets_t[:, :, 0].detach().cpu().view(-1)
            soh_pred = 1.0 - pcl_pred
            soh_true = 1.0 - pcl_true

            rul_true = None
            if targets_t.shape[-1] >= 2:
                rul_true = targets_t[:, :, 1].detach().cpu().view(-1)
            rul_pred = _compute_rul_curve(cycles, soh_pred, threshold_soh)

            idx = slice(None, None, step)
            cell_payload: Dict[str, Any] = {
                "cell_id": int(cid),
                "cycles": cycles[idx].tolist(),
                "pcl_true": pcl_true[idx].tolist(),
                "pcl_pred": pcl_pred[idx].tolist(),
                "soh_true": soh_true[idx].tolist(),
                "soh_pred": soh_pred[idx].tolist(),
                "rul_pred": rul_pred[idx].tolist(),
                "metrics_soh": _eval_metrics(soh_pred, soh_true),
            }
            if rul_true is not None:
                cell_payload["rul_true"] = rul_true[idx].tolist()
            results_cells.append(cell_payload)

        return {
            "model_file": os.path.basename(abs_model_path),
            "model_url": f"/results/{os.path.basename(abs_model_path)}",
            "model_type": artifact.get("model_type"),
            "threshold_soh": threshold_soh,
            "step": step,
            "cells": results_cells,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ExportRequest(PredictRequest):
    fmt: str = "csv"

    model_config = {"extra": "forbid"}


@predict_router.post("/export")
async def export_predict(req: ExportRequest) -> Dict[str, Any]:
    try:
        payload = await run_predict(PredictRequest(**req.model_dump(exclude={"fmt"})))
        fmt = str(req.fmt or "csv").strip().lower()
        if fmt not in {"csv", "excel", "xlsx"}:
            raise ValueError("fmt must be one of: csv, excel")

        timestamp = _now_cn().strftime("%Y%m%d%H%M%S")
        base = os.path.splitext(os.path.basename(payload["model_file"]))[0]
        ext = "csv" if fmt == "csv" else "xlsx"
        out_name = f"predict_{base}_{timestamp}_{uuid.uuid4().hex[:8]}.{ext}"
        out_path = os.path.join(RESULTS_DIR, out_name)

        rows: List[Dict[str, Any]] = []
        for cell in payload.get("cells", []):
            cid = cell.get("cell_id")
            cycles = cell.get("cycles") or []
            n = len(cycles)
            for i in range(n):
                row = {
                    "cell_id": cid,
                    "cycle": cycles[i],
                    "pcl_true": (cell.get("pcl_true") or [None] * n)[i],
                    "pcl_pred": (cell.get("pcl_pred") or [None] * n)[i],
                    "soh_true": (cell.get("soh_true") or [None] * n)[i],
                    "soh_pred": (cell.get("soh_pred") or [None] * n)[i],
                    "rul_true": (cell.get("rul_true") or [None] * n)[i],
                    "rul_pred": (cell.get("rul_pred") or [None] * n)[i],
                }
                rows.append(row)

        if fmt == "csv":
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["cell_id", "cycle", "pcl_true", "pcl_pred", "soh_true", "soh_pred", "rul_true", "rul_pred"],
                )
                w.writeheader()
                w.writerows(rows)
        else:
            try:
                from openpyxl import Workbook
            except Exception as e:
                raise ValueError(f"missing dependency for excel export: {e}")
            wb = Workbook()
            ws = wb.active
            ws.title = "prediction"
            header = ["cell_id", "cycle", "pcl_true", "pcl_pred", "soh_true", "soh_pred", "rul_true", "rul_pred"]
            ws.append(header)
            for r in rows:
                ws.append([r.get(k) for k in header])
            wb.save(out_path)

        return {"file": out_name, "url": f"/results/{out_name}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
