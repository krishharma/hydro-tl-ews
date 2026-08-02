"""Long-record (at-site climatological) extreme event thresholds.

Calculating site-specific percentiles from a 2-year warmup window biases
threshold estimates (e.g. a drought year masquerading as normal).  We
instead estimate extreme quantiles from a long historical record of the
target basin, providing stable Q5/Q95/Q99 references.

TERMINOLOGY NOTE: this is an AT-SITE frequency analysis using the long
historical record, not a true Regional Frequency Analysis (RFA sensu
Hosking & Wallis). The function name ``regional_thresholds`` is kept for
backwards compatibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_erf = np.vectorize(math.erf, otypes=[float])


@dataclass
class ExtremeThresholds:
    q5: float
    q95: float
    q99: float


def regional_thresholds(streamflow: pd.Series,
                        years_required: int = 30) -> ExtremeThresholds:
    """Compute Q5/Q95/Q99 from the provided record."""
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
        event = (observed_flow >= thr).astype(float)
    elif kind == "drought":
        thr = thresholds.q5
        event = (observed_flow <= thr).astype(float)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    out = pd.DataFrame(index=observed_flow.index)
    for L in lead_times:
        # rolling(L) at t+L covers [t+1, ..., t+L]; shift(-L) aligns to t.
        future = event.rolling(L, min_periods=L).max().shift(-L)
        out[f"{kind}_{percentile}_lead{L}d"] = future.fillna(0.0)
    return out


def predicted_warning_probabilities(predicted_flow: pd.Series,
                                    thresholds: ExtremeThresholds,
                                    kind: str = "flood",
                                    percentile: str = "q95",
                                    sigma: float | None = None,
                                    lead_times: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Convert deterministic predictions to warning probabilities.

    Assume Gaussian residual std ``sigma`` (default = 25% of the threshold)
    and compute P(any exceedance in the lead window) under day-independence:
    ``1 - prod(1 - p_i)`` for ``i in [t+1, t+L]``.
    """
    if kind == "flood":
        thr = getattr(thresholds, percentile)
    elif kind == "drought":
        thr = thresholds.q5
    else:
        raise ValueError(f"Unknown kind: {kind}")

    sigma = float(sigma if sigma is not None else 0.25 * abs(thr))
    sigma = max(sigma, 1e-6)
    x = predicted_flow.to_numpy(dtype=float)
    z = (thr - x) / (sigma * math.sqrt(2.0))
    erf_z = _erf(z)
    if kind == "flood":
        prob_one_day = 0.5 * (1.0 - erf_z)
    else:
        prob_one_day = 0.5 * (1.0 + erf_z)

    p = pd.Series(prob_one_day, index=predicted_flow.index)
    log1m = np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
    out = pd.DataFrame(index=predicted_flow.index)
    for L in lead_times:
        any_event = 1.0 - np.exp(
            log1m.rolling(L, min_periods=L).sum().shift(-L)
        )
        out[f"{kind}_{percentile}_lead{L}d"] = any_event.fillna(0.0)
    return out
