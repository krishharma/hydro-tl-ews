"""Time-series preprocessing utilities for EA-LSTM training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Normalizer:
    """Z-score normalizer fit on the training period only (no look-ahead)."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Normalizer":
        return cls(mean=df.mean(), std=df.std().replace(0, 1.0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std

    def inverse_transform_series(self, x: np.ndarray, col: str) -> np.ndarray:
        return x * float(self.std[col]) + float(self.mean[col])


@dataclass
class StaticNormalizer:
    """Min-max normalizer for static catchment attributes (across basins)."""
    min: pd.Series
    max: pd.Series

    @classmethod
    def fit(cls, attrs: pd.DataFrame) -> "StaticNormalizer":
        mn, mx = attrs.min(), attrs.max()
        rng = (mx - mn).replace(0, 1.0)
        return cls(min=mn, max=mn + rng)

    def transform(self, attrs: pd.DataFrame) -> pd.DataFrame:
        rng = (self.max - self.min).replace(0, 1.0)
        return (attrs - self.min) / rng


def quality_control(forcings: pd.DataFrame, streamflow: pd.Series,
                    max_gap_days: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Drop non-physical values and linearly interpolate short gaps."""
    f = forcings.copy()
    if "prcp(mm/day)" in f.columns:
        f.loc[f["prcp(mm/day)"] < 0, "prcp(mm/day)"] = np.nan
    f = f.interpolate(method="linear", limit=max_gap_days, limit_direction="both")
    q = streamflow.where(streamflow >= 0)
    q = q.interpolate(method="linear", limit=max_gap_days, limit_direction="both")
    return f, q


def align_forcing_streamflow(
    forcings: pd.DataFrame, streamflow: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Use only dates present in both series (CAMELS forcing/flow indices can differ)."""
    idx = forcings.index.intersection(streamflow.index)
    idx = idx.sort_values()
    return forcings.loc[idx], streamflow.loc[idx]


def make_sequences(forcings: np.ndarray, streamflow: np.ndarray,
                   sequence_length: int = 365,
                   index: np.ndarray | pd.DatetimeIndex | None = None,
                   ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (T, F) forcings + (T,) streamflow into (N, L, F) / (N,) windows.

    Each sample uses the previous ``sequence_length`` forcings to predict the
    streamflow at the *current* day (i.e. the target is right-aligned).

    When ``index`` is provided, also return the target dates of surviving
    windows (after NaN filtering) so callers can align predictions correctly.
    """
    T = forcings.shape[0]
    if T < sequence_length:
        empty_x = np.empty((0, sequence_length, forcings.shape[1]), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        if index is None:
            return empty_x, empty_y
        return empty_x, empty_y, np.empty((0,), dtype="datetime64[ns]")
    X = np.lib.stride_tricks.sliding_window_view(
        forcings, window_shape=sequence_length, axis=0
    ).transpose(0, 2, 1).astype(np.float32)
    y = streamflow[sequence_length - 1:].astype(np.float32)
    # Drop NaN-target windows
    # Avoid materialising a (N, L, F) bool intermediate (~22 MiB): NaN propagates
    # through sum, so X.sum() is NaN iff any element in the sample is NaN.
    mask = ~np.isnan(y) & ~np.isnan(X.sum(axis=(1, 2)))
    X_out, y_out = X[mask], y[mask]
    if index is None:
        return X_out, y_out
    dates = np.asarray(index[sequence_length - 1:])[mask]
    return X_out, y_out, dates
