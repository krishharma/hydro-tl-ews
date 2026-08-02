"""PyTorch ``Dataset`` wrappers for multi-basin EA-LSTM training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # torch optional during static analysis
    torch = None
    Dataset = object  # type: ignore[misc,assignment]

from .camels import BasinData, DYNAMIC_FEATURES, STATIC_ATTRIBUTES
from .preprocessing import (
    Normalizer,
    StaticNormalizer,
    align_forcing_streamflow,
    quality_control,
)


@dataclass
class BasinSample:
    basin_id: str
    forcings: np.ndarray  # (L, F_dyn)
    statics: np.ndarray   # (F_static,)
    target: float


class MultiBasinSequenceDataset(Dataset):
    """Concatenates sliding-window samples from many basins for EA-LSTM training.

    Each item yields ``(forcings (L, F_dyn), statics (F_static,), target,
    basin_std)`` — static attributes are *not* repeated along time inside the
    dataset; the EA-LSTM model handles that internally via its embedding gate.

    Windows are cut **lazily** in ``__getitem__``: only one normalized forcing
    array per basin is held in memory (~raw forcing size), and each ``L``-day
    window is sliced on demand. This avoids materialising the full
    ``(N_samples, L, F_dyn)`` tensor, which for the whole CAMELS-US set is tens
    of GB and previously exhausted RAM.
    """

    def __init__(
        self,
        basins: Dict[str, BasinData],
        period: tuple[str, str],
        dyn_normalizer: Normalizer,
        static_normalizer: StaticNormalizer,
        sequence_length: int = 365,
    ):
        self.basin_ids: List[str] = list(basins.keys())
        self.sequence_length = sequence_length
        self.dyn_normalizer = dyn_normalizer
        self.static_normalizer = static_normalizer

        L = sequence_length
        # Per-basin stores (indexed by ``basin_pos``); samples reference them.
        self._forcings: List[np.ndarray] = []   # each (T_b, F_dyn) float32
        self._statics: List[np.ndarray] = []     # each (F_static,) float32
        sample_basin, sample_end = [], []        # int32 arrays per basin
        y_parts, std_parts, all_basin = [], [], []
        date_parts = []                          # datetime64 target dates

        start, end = period
        for bid, bd in basins.items():
            f, q = quality_control(bd.forcings, bd.streamflow)
            f = f.loc[start:end]
            q = q.loc[start:end]
            f, q = align_forcing_streamflow(f, q)
            if len(f) == 0:
                continue
            # Per-basin std for NSELoss weighting (Kratzert 2019): prevents
            # high-flow basins from dominating the gradient.
            basin_std_val = float(np.nanstd(q.values)) if len(q) > 1 else 1.0
            basin_std_val = max(basin_std_val, 0.01)

            f_norm = (dyn_normalizer.transform(f)[DYNAMIC_FEATURES]
                      .to_numpy().astype(np.float32))
            q_arr = q.to_numpy().astype(np.float32)
            T = f_norm.shape[0]
            if T < L:
                continue

            # Valid window ends: no NaN feature anywhere in the L-day span and a
            # non-NaN target on the (right-aligned) prediction day. Computed via a
            # cumulative count of bad rows — O(T) memory, no (N, L, F) intermediate.
            row_bad = np.isnan(f_norm).any(axis=1)               # (T,)
            csum = np.concatenate(([0], np.cumsum(row_bad)))     # (T + 1,)
            ends = np.arange(L - 1, T)
            bad_in_window = csum[ends + 1] - csum[ends - L + 1]
            valid = (bad_in_window == 0) & ~np.isnan(q_arr[ends])
            valid_ends = ends[valid]
            if valid_ends.size == 0:
                continue

            statics = static_normalizer.transform(
                bd.attributes.to_frame().T
            ).reindex(columns=STATIC_ATTRIBUTES).to_numpy().astype(np.float32)[0]

            bpos = len(self._forcings)
            # Contiguous once at build time so __getitem__ avoids a copy per sample.
            self._forcings.append(np.ascontiguousarray(f_norm))
            self._statics.append(np.ascontiguousarray(statics))
            sample_basin.append(np.full(valid_ends.size, bpos, dtype=np.int32))
            sample_end.append(valid_ends.astype(np.int32))
            y_parts.append(q_arr[valid_ends])
            std_parts.append(np.full(valid_ends.size, basin_std_val, dtype=np.float32))
            all_basin.extend([bid] * valid_ends.size)
            date_parts.append(f.index.to_numpy()[valid_ends])

        if y_parts:
            self._sample_basin = np.concatenate(sample_basin)
            self._sample_end = np.concatenate(sample_end)
            self.y = np.concatenate(y_parts)
            self.basin_std = np.concatenate(std_parts)
            self.dates = np.concatenate(date_parts)
        else:
            self._sample_basin = np.empty((0,), dtype=np.int32)
            self._sample_end = np.empty((0,), dtype=np.int32)
            self.y = np.empty((0,), dtype=np.float32)
            self.basin_std = np.empty((0,), dtype=np.float32)
            self.dates = np.empty((0,), dtype="datetime64[ns]")
        self.basin_index = all_basin

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if torch is None:
            raise RuntimeError("torch is required to iterate the dataset.")
        L = self.sequence_length
        bpos = int(self._sample_basin[idx])
        end = int(self._sample_end[idx])
        window = self._forcings[bpos][end - L + 1:end + 1]  # (L, F_dyn) contiguous view
        return (
            torch.from_numpy(window),
            torch.from_numpy(self._statics[bpos]),
            torch.tensor(float(self.y[idx]), dtype=torch.float32),
            torch.tensor(float(self.basin_std[idx]), dtype=torch.float32),
        )
