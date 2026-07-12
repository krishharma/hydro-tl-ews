"""Rolling-origin (walk-forward) backtester.

Simulates real-time operational forecasting by repeatedly fine-tuning on an
expanding training window and producing predictions for the next horizon.
Eliminates the temporal leakage that plagues random-split evaluations of
autocorrelated hydrological time series.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data.camels import BasinData, DYNAMIC_FEATURES, STATIC_ATTRIBUTES
from ..data.preprocessing import (
    Normalizer,
    StaticNormalizer,
    align_forcing_streamflow,
    make_sequences,
    quality_control,
)
from ..models.ealstm import EALSTM
from ..utils.device import training_device
from ..utils.logging import get_logger
from .trainer import Trainer
from .transfer import FineTuneConfig, fine_tune_conservative

log = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    initial_train_end: str          # e.g. "2010-12-31" (warmup end)
    eval_end: str                   # e.g. "2014-12-31"
    refit_every_days: int = 90      # full fine-tune cadence
    online_bias_correction: bool = True
    sequence_length: int = 365
    batch_size: int = 256
    val_tail_days: int = 90         # held-out tail of the training window used
                                    # as the refit validation signal; 0 disables
                                    # (falls back to train-loss early stopping)
    refit_train_start: str | None = None
                                    # first date refits may train on.  None uses
                                    # the full loaded record — which can reach
                                    # decades before the warmup and silently
                                    # break a data-scarce simulation.  Set to
                                    # the warmup start (e.g. "2009-01-01") so
                                    # refits only ever see warmup + observed
                                    # evaluation data.
    fine_tune_cfg: FineTuneConfig = field(default_factory=FineTuneConfig)


@dataclass
class WalkForwardResult:
    dates: pd.DatetimeIndex
    observed: np.ndarray
    predicted: np.ndarray
    bias_corrections: np.ndarray
    refit_dates: list[pd.Timestamp]


def _build_loader(forcings: pd.DataFrame, streamflow: pd.Series,
                  attrs: pd.Series, dyn_norm: Normalizer,
                  static_norm: StaticNormalizer, cfg: WalkForwardConfig,
                  shuffle: bool = True) -> DataLoader | None:
    f, q = quality_control(forcings, streamflow)
    f, q = align_forcing_streamflow(f, q)
    f_norm = dyn_norm.transform(f)[DYNAMIC_FEATURES]
    X, y = make_sequences(f_norm.to_numpy(), q.to_numpy(),
                          sequence_length=cfg.sequence_length)
    if len(X) == 0:
        return None
    statics = static_norm.transform(attrs.to_frame().T).reindex(
        columns=STATIC_ATTRIBUTES).to_numpy().astype(np.float32)[0]
    S = np.tile(statics, (len(X), 1))
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(S),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_window(model: EALSTM, loader: DataLoader,
                    device: str | None = None) -> np.ndarray:
    device = device or training_device()
    model.eval()
    out = []
    for X, S, _ in loader:
        out.append(model(X.to(device), S.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def walk_forward(
    model: EALSTM,
    target_basin: BasinData,
    dyn_norm: Normalizer,
    static_norm: StaticNormalizer,
    cfg: WalkForwardConfig,
    device: str | None = None,
    refit_fn: Callable | None = None,
) -> WalkForwardResult:
    """Run a rolling-origin backtest on the target basin.

    Parameters
    ----------
    model:
        Pre-trained EA-LSTM.  The function makes a deepcopy each refit so the
        caller's weights remain unchanged across calls.
    target_basin:
        BasinData with daily forcings and streamflow covering both warmup
        and evaluation periods.
    refit_fn:
        Optional override for the fine-tuning routine; defaults to
        :func:`fine_tune_conservative` (Approach A).
    """
    device = device or training_device()
    refit_fn = refit_fn or fine_tune_conservative
    # Deep-copy so each refit modifies an internal copy; caller's weights intact.
    model = copy.deepcopy(model)
    log.info("Walk-forward on device: %s", device)

    # Establish full date range
    full_dates = target_basin.streamflow.dropna().index
    full_dates = full_dates.intersection(target_basin.forcings.dropna().index)
    full_dates = pd.DatetimeIndex(full_dates)

    init_end = pd.Timestamp(cfg.initial_train_end)
    eval_end = pd.Timestamp(cfg.eval_end)

    refit_dates: list[pd.Timestamp] = []
    bias_correction = 0.0
    total_bias_sum = 0.0
    total_bias_count = 0
    all_dates, all_obs, all_pred, all_bias = [], [], [], []

    # Refits train on [train_start, cur_start).  A None train_start replicates
    # the historical behaviour (full record); a warmup-start date keeps the
    # data-scarce simulation honest.
    train_start = (pd.Timestamp(cfg.refit_train_start)
                   if cfg.refit_train_start else None)

    cur_start = init_end + pd.Timedelta(days=1)
    while cur_start <= eval_end:
        chunk_end = min(cur_start + pd.Timedelta(days=cfg.refit_every_days - 1),
                        eval_end)
        # ---- Refit on data up to cur_start - 1 ----------------------------
        train_end = cur_start - pd.Timedelta(days=1)
        val_loader = None
        fit_end = train_end
        if cfg.val_tail_days > 0:
            # Hold out the most recent val_tail_days of the window as the
            # refit validation signal (fixes the historical val=nan issue:
            # refit_fn previously always received val_loader=None).
            val_start = train_end - pd.Timedelta(days=cfg.val_tail_days - 1)
            val_hist_start = val_start - pd.Timedelta(days=cfg.sequence_length - 1)
            if train_start is not None:
                val_hist_start = max(val_hist_start, train_start)
            val_loader = _build_loader(
                target_basin.forcings.loc[val_hist_start:train_end],
                target_basin.streamflow.loc[val_hist_start:train_end],
                target_basin.attributes, dyn_norm, static_norm, cfg,
                shuffle=False,
            )
            if val_loader is not None:
                fit_end = val_start - pd.Timedelta(days=1)
        train_loader = _build_loader(
            target_basin.forcings.loc[train_start:fit_end],
            target_basin.streamflow.loc[train_start:fit_end],
            target_basin.attributes, dyn_norm, static_norm, cfg, shuffle=True,
        )
        if train_loader is None and fit_end != train_end:
            # Holding out the tail starved training: fall back to the full
            # window with train-loss early stopping (Trainer handles
            # val_loader=None gracefully).
            val_loader = None
            fit_end = train_end
            train_loader = _build_loader(
                target_basin.forcings.loc[train_start:fit_end],
                target_basin.streamflow.loc[train_start:fit_end],
                target_basin.attributes, dyn_norm, static_norm, cfg,
                shuffle=True,
            )
        if train_loader is not None:
            log.info("Refit at %s | train %s..%s | val %s", cur_start.date(),
                     train_start.date() if train_start is not None else "0",
                     fit_end.date(),
                     "none (train-loss stopping)" if val_loader is None
                     else f"last {cfg.val_tail_days}d of window")
            refit_fn(model, train_loader, val_loader, cfg.fine_tune_cfg,
                     device=device)
            refit_dates.append(cur_start)

        # ---- Predict next chunk -----------------------------------------
        eval_forcings = target_basin.forcings.loc[
            cur_start - pd.Timedelta(days=cfg.sequence_length - 1):chunk_end
        ]
        eval_flow = target_basin.streamflow.loc[
            cur_start - pd.Timedelta(days=cfg.sequence_length - 1):chunk_end
        ]
        eval_loader = _build_loader(
            eval_forcings, eval_flow, target_basin.attributes,
            dyn_norm, static_norm, cfg, shuffle=False,
        )
        if eval_loader is None:
            cur_start = chunk_end + pd.Timedelta(days=1)
            continue
        preds = _predict_window(model, eval_loader, device=device)
        # Sequence target dates correspond to last day of each window,
        # i.e. the days between cur_start and chunk_end inclusive.
        target_dates = pd.date_range(cur_start, periods=len(preds), freq="D")
        target_dates = target_dates[target_dates <= chunk_end]
        preds = preds[: len(target_dates)]
        obs = target_basin.streamflow.reindex(target_dates).to_numpy()

        if cfg.online_bias_correction:
            preds = preds + bias_correction
            valid = ~np.isnan(obs)
            if valid.any():
                chunk_err = obs[valid] - (preds[valid] - bias_correction)
                total_bias_sum += float(np.sum(chunk_err))
                total_bias_count += int(np.sum(valid))
                bias_correction = total_bias_sum / total_bias_count
        all_dates.extend(target_dates)
        all_obs.extend(obs.tolist())
        all_pred.extend(preds.tolist())
        all_bias.extend([bias_correction] * len(target_dates))

        cur_start = chunk_end + pd.Timedelta(days=1)

    return WalkForwardResult(
        dates=pd.DatetimeIndex(all_dates),
        observed=np.array(all_obs),
        predicted=np.array(all_pred),
        bias_corrections=np.array(all_bias),
        refit_dates=refit_dates,
    )
