# -*- coding: utf-8 -*-
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


predict_router = APIRouter(prefix="/api/predict", tags=["predict"])


class PredictionRequest(BaseModel):
    model_name: str
    input_data: Optional[List[float]] = None
    # Add other fields as necessary


@predict_router.post("/run")
async def run_prediction(req: PredictionRequest):
    # This is a placeholder to fix the AttributeError.
    # TODO: Implement actual prediction logic or restore it if it was deleted.
    return {"message": "Prediction endpoint is under construction", "model": req.model_name}


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
