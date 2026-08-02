"""Hydrological and early-warning evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _drop_nan(y_true: np.ndarray, y_pred: np.ndarray):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    y_true, y_pred = _drop_nan(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009)."""
    y_true, y_pred = _drop_nan(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan")
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(np.std(y_pred) / (np.std(y_true) + 1e-12))
    beta = float(np.mean(y_pred) / (np.mean(y_true) + 1e-12))
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percent bias [%]."""
    y_true, y_pred = _drop_nan(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return 100.0 * float(np.sum(y_pred - y_true) / np.sum(y_true))


def auc_roc(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Area under ROC for a binary label and a probability/score vector.

    Uses midranks so tied scores (common for saturated EWS probabilities)
    yield the correct Mann–Whitney AUC (all ties → 0.5).
    """
    y_true, p_pred = _drop_nan(y_true.astype(float), p_pred)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p_pred, kind="mergesort")
    ranks = np.empty(len(p_pred), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and p_pred[order[j]] == p_pred[order[i]]:
            j += 1
        # Average ranks for a tied block (1-based ranks).
        mid = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = mid
        i = j
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def f1_at_threshold(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_true, p_pred = _drop_nan(y_true.astype(float), p_pred)
    pred = (p_pred >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    if tp + fp == 0 or tp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y_true, p_pred = _drop_nan(y_true.astype(float), p_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((p_pred - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, observed_freq, count_per_bin) for a reliability diagram."""
    y_true, p_pred = _drop_nan(y_true.astype(float), p_pred)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    freq = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (p_pred >= edges[i]) & (p_pred < edges[i + 1] if i < n_bins - 1
                                       else p_pred <= edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            freq[i] = float(y_true[mask].mean())
        else:
            freq[i] = np.nan
    return centers, freq, counts


@dataclass
class MetricsBundle:
    nse: float
    kge: float
    pbias: float
    auc: float | None = None
    f1: float | None = None
    brier: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {"NSE": self.nse, "KGE": self.kge, "PBIAS": self.pbias,
                "AUC": self.auc, "F1": self.f1, "Brier": self.brier}


def compute_all(y_true: np.ndarray, y_pred: np.ndarray,
                event_label: np.ndarray | None = None,
                event_score: np.ndarray | None = None,
                threshold: float = 0.5) -> MetricsBundle:
    return MetricsBundle(
        nse=nse(y_true, y_pred),
        kge=kge(y_true, y_pred),
        pbias=pbias(y_true, y_pred),
        auc=auc_roc(event_label, event_score) if event_label is not None else None,
        f1=f1_at_threshold(event_label, event_score, threshold)
            if event_label is not None else None,
        brier=brier_score(event_label, event_score)
            if event_label is not None else None,
    )
