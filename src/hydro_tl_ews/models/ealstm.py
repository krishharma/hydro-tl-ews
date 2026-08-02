"""Entity-Aware LSTM (EA-LSTM) implementation.

Reference:
    Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., &
    Nearing, G. (2019). Towards learning universal, regional, and local
    hydrological behaviors via machine learning applied to large-sample
    datasets. *Hydrology and Earth System Sciences*, 23(12), 5089-5110.
    https://doi.org/10.5194/hess-23-5089-2019

The EA-LSTM augments the standard LSTM input gate with a *static* gate
computed once from catchment attributes — this is what lets a single model
learn a regional rainfall-runoff representation while remaining identifiable
per-basin.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EALSTMConfig:
    dynamic_input_size: int
    static_input_size: int
    hidden_size: int = 256
    dropout: float = 0.4
    output_size: int = 1
    initial_forget_bias: float = 3.0


class EALSTMCell(nn.Module):
    """Custom EA-LSTM cell.

    The input gate ``i`` is a function of the *static* attributes only and is
    therefore time-invariant; the forget, cell-candidate, and output gates
    depend on the dynamic forcings and the previous hidden state.
    """

    def __init__(self, cfg: EALSTMConfig):
        super().__init__()
        self.cfg = cfg
        H, D, S = cfg.hidden_size, cfg.dynamic_input_size, cfg.static_input_size

        # Static input gate: i_t = sigmoid(W_i s + b_i)
        self.W_i = nn.Linear(S, H)

        # Dynamic gates (forget, candidate, output) -- combined for efficiency
        self.W_x = nn.Linear(D, 3 * H, bias=True)
        self.W_h = nn.Linear(H, 3 * H, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        # Forget-gate bias initialization (gates ordered: f, g, o)
        H = self.cfg.hidden_size
        with torch.no_grad():
            self.W_x.bias[0:H].fill_(self.cfg.initial_forget_bias)

    def forward(self, x_t: torch.Tensor, s: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor],
                i_t: torch.Tensor | None = None):
        h_prev, c_prev = state
        # Static input gate is time-invariant — callers may precompute ``i_t``.
        if i_t is None:
            i_t = torch.sigmoid(self.W_i(s))
        gates = self.W_x(x_t) + self.W_h(h_prev)
        f_t, g_t, o_t = gates.chunk(3, dim=-1)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class EALSTM(nn.Module):
    """Single-layer EA-LSTM with a dense streamflow head."""

    def __init__(self, cfg: EALSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.cell = EALSTMCell(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x: torch.Tensor, s: torch.Tensor,
                return_sequence: bool = False) -> torch.Tensor:
        """
        x: (B, L, D) dynamic forcings
        s: (B, S)    static attributes
        Returns the final-step prediction (B, 1) by default.
        """
        B, L, _ = x.shape
        H = self.cfg.hidden_size
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        # Precompute static gate once per sequence (was recomputed every timestep).
        i_t = torch.sigmoid(self.cell.W_i(s))
        outputs = []
        for t in range(L):
            h, c = self.cell(x[:, t, :], s, (h, c), i_t=i_t)
            if return_sequence:
                outputs.append(h)
        last_h = self.dropout(h)
        y = self.head(last_h)
        if return_sequence:
            return torch.stack(outputs, dim=1), y
        return y

    # --------------------------------------------------------------- helpers
    def freeze_lstm(self) -> None:
        """Freeze every parameter except the dense head (Approach A)."""
        for p in self.cell.parameters():
            p.requires_grad = False

    def unfreeze_lstm(self, fraction: float = 1.0) -> None:
        """Unfreeze a fraction of the LSTM parameters (Approach B).

        ``fraction`` selects the *last* ``fraction`` of parameters by index;
        this targets higher-level abstractions while keeping low-level
        precip/temp feature extraction frozen.
        """
        params = list(self.cell.parameters())
        n_unfreeze = max(1, int(round(len(params) * fraction)))
        for p in params[:-n_unfreeze]:
            p.requires_grad = False
        for p in params[-n_unfreeze:]:
            p.requires_grad = True

    def trainable_parameter_groups(self, head_lr: float, lstm_lr: float):
        """Return param groups for differential learning rates."""
        head_params = list(self.head.parameters())
        lstm_params = [p for p in self.cell.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if lstm_params:
            groups.append({"params": lstm_params, "lr": lstm_lr})
        return groups
