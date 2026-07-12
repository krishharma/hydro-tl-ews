"""Long-record (at-site climatological) extreme event thresholds.

Calculating site-specific percentiles from a 2-year warmup window biases
threshold estimates (e.g. a drought year masquerading as normal).  We
instead estimate extreme quantiles from the *full* available record of the
target basin, providing stable Q5/Q95/Q99 references.

TERMINOLOGY NOTE: this is an AT-SITE frequency analysis using the long
historical record, not a true Regional Frequency Analysis (RFA sensu
Hosking & Wallis: pooling multiple donor sites via index-flood/L-moments).
The function name ``regional_thresholds`` is kept for backwards
compatibility; a genuine donor-pooled RFA is future work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExtremeThresholds:
    q5: float
    q95: float
    q99: float


def regional_thresholds(streamflow: pd.Series,
                        years_required: int = 30) -> ExtremeThresholds:
    """Compute Q5/Q95/Q99 from the full available record."""
    s = streamflow.dropna()
    n_years = len(s) / 365.25
    if n_years < years_required:
        raise ValueError(
            f"At least {years_required} years required; got {n_years:.1f}.")
    return ExtremeThresholds(
        q5=float(np.quantile(s, 0.05)),
        q95=float(np.quantile(s, 0.95)),
        q99=float(np.quantile(s, 0.99)),
    )


def warning_labels(observed_flow: pd.Series,
                   thresholds: ExtremeThresholds,
                   kind: str = "flood",
                   percentile: str = "q95",
                   lead_times: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Build binary early-warning labels at multiple lead times.

    A label at date *t* with lead-time *L* is 1 if any day in
    ``[t+1, t+L]`` exceeds (flood) or falls below (drought) the threshold.
    """
    if kind == "flood":
        thr = getattr(thresholds, percentile)
        cmp = lambda x: x >= thr
    elif kind == "drought":
        thr = thresholds.q5
        cmp = lambda x: x <= thr
    else:
        raise ValueError(f"Unknown kind: {kind}")

    out = pd.DataFrame(index=observed_flow.index)
    for L in lead_times:
        # shift(-L) aligns future values so that rolling(L) collects exactly
        # [t+1, ..., t+L] — the correct forward-looking window for lead time L.
        # (shift(-1).rolling(L) was incorrect: it mixed past and future values.)
        future = observed_flow.rolling(L, min_periods=1).apply(
            lambda w: float(cmp(w).any()), raw=False
        ).shift(-L)
        out[f"{kind}_{percentile}_lead{L}d"] = (future > 0).astype(float).fillna(0.0)
    return out


def predicted_warning_probabilities(predicted_flow: pd.Series,
                                    thresholds: ExtremeThresholds,
                                    kind: str = "flood",
                                    percentile: str = "q95",
                                    sigma: float | None = None,
                                    lead_times: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Convert deterministic predictions to warning probabilities.

    A simple operational mapping: assume Gaussian residual std ``sigma``
    (default = 25% of the threshold) and integrate over the threshold for
    each lead-time max (flood) or min (drought).
    """
    from math import erf, sqrt
    sigma = sigma or 0.25 * abs(getattr(thresholds, percentile))
    if kind == "flood":
        thr = getattr(thresholds, percentile)
        prob_one_day = 0.5 * (1 - np.array([
            erf((thr - x) / (sigma * sqrt(2))) for x in predicted_flow.values
        ]))
    else:
        thr = thresholds.q5
        prob_one_day = 0.5 * (1 + np.array([
            erf((thr - x) / (sigma * sqrt(2))) for x in predicted_flow.values
        ]))
    p = pd.Series(prob_one_day, index=predicted_flow.index)
    log1m = np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
    out = pd.DataFrame(index=predicted_flow.index)
    for L in lead_times:
        # P(any event in [t+1, ..., t+L]) = 1 - prod(1-p_i) for i in t+1..t+L
        # shift(-L).rolling(L) collects exactly log(1-p[t+1])..log(1-p[t+L])
        any_event = 1.0 - np.exp(
            log1m.rolling(L, min_periods=1).sum().shift(-L)
        )
        out[f"{kind}_{percentile}_lead{L}d"] = any_event.fillna(0.0)
    return out
