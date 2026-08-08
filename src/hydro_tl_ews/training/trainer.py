"""Training loops for pre-training, fine-tuning, and zero-shot evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..utils.device import training_device

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.ealstm import EALSTM
from ..models.losses import NSELoss
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_no_improve: int = 0
    history: list[dict] = field(default_factory=list)


class Trainer:
    """Generic trainer for EA-LSTM pre-training and fine-tuning.

    Behavioural switches:
        * ``mode='pretrain'``    - all parameters trainable, NSELoss
        * ``mode='conservative'``- LSTM frozen, head only, MSE
        * ``mode='progressive'`` - last-25% LSTM unfrozen, differential LR
        * ``mode='local'``       - random init, all params trainable, MSE
    """

    def __init__(
        self,
        model: EALSTM,
        device: str | None = None,
        mode: str = "pretrain",
        head_lr: float = 1e-3,
        lstm_lr: float = 1e-5,
        weight_decay: float = 0.0,
        clip_grad_norm: Optional[float] = 1.0,
        unfreeze_fraction: float = 0.25,
    ):
        self.device = device or training_device()
        self.model = model.to(self.device)
        self.mode = mode
        log.info("Training on device: %s", self.device)
        self.clip = clip_grad_norm

        if mode == "conservative":
            self.model.freeze_lstm()
        elif mode == "progressive":
            self.model.unfreeze_lstm(fraction=unfreeze_fraction)
        elif mode in {"pretrain", "local", "zero_shot"}:
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.optimizer = torch.optim.Adam(
            self.model.trainable_parameter_groups(head_lr=head_lr, lstm_lr=lstm_lr),
            weight_decay=weight_decay,
        )
        self.nse_loss = NSELoss()
        self.state = TrainState()

    # --------------------------------------------------------------- helpers
    def _step(self, X: torch.Tensor, S: torch.Tensor, y: torch.Tensor,
              basin_std: torch.Tensor, train: bool) -> float:
        X = X.to(self.device)
        S = S.to(self.device)
        y = y.to(self.device)
        basin_std = basin_std.to(self.device)

        if train:
            self.optimizer.zero_grad()
        y_hat = self.model(X, S)

        if self.mode in {"pretrain"}:
            loss = self.nse_loss(y_hat, y, basin_std)
        else:
            loss = torch.mean((y_hat.squeeze(-1) - y) ** 2)

        if train:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.clip,
                )
            self.optimizer.step()
        return float(loss.item())

    # ---------------------------------------------------------------- epoch
    def run_epoch(self, loader: DataLoader, train: bool, basin_std: float = 1.0) -> float:
        self.model.train(mode=train)
        total, n = 0.0, 0
        n_batches = len(loader)
        log_every = max(1, n_batches // 5) if n_batches >= 10 else n_batches + 1
        for i, batch in enumerate(loader, start=1):
            if len(batch) == 4:
                X, S, y, std_t = batch
            else:
                X, S, y = batch
                std_t = torch.full_like(y, fill_value=basin_std)
            with torch.set_grad_enabled(train):
                loss = self._step(X, S, y, std_t, train=train)
            total += loss * len(y)
            n += len(y)
            if train and (i == 1 or i % log_every == 0 or i == n_batches):
                log.info("  batch %d/%d | running_loss=%.4f", i, n_batches, total / max(n, 1))
        return total / max(n, 1)

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int = 50,
        patience: int = 10,
        basin_std: float = 1.0,
        checkpoint_path: Optional[str | Path] = None,
        restore_best_weights: bool = True,
    ) -> TrainState:
        best_state: Optional[dict] = None
        for epoch in range(1, epochs + 1):
            self.state.epoch = epoch
            train_loss = self.run_epoch(train_loader, train=True, basin_std=basin_std)
            val_loss = (self.run_epoch(val_loader, train=False, basin_std=basin_std)
                        if val_loader is not None else float("nan"))
            self.state.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            log.info("Epoch %02d | train=%.4f | val=%.4f", epoch, train_loss, val_loss)

            cur = val_loss if val_loader is not None else train_loss
            if cur < self.state.best_val_loss - 1e-6:
                self.state.best_val_loss = cur
                self.state.best_epoch = epoch
                self.state.epochs_no_improve = 0
                if restore_best_weights:
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                self.state.epochs_no_improve += 1
                if self.state.epochs_no_improve >= patience:
                    log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                             epoch, patience)
                    break
        if best_state is not None and self.state.epoch != self.state.best_epoch:
            self.model.load_state_dict(best_state)
            log.info("Restored best weights from epoch %d (loss=%.4f)",
                     self.state.best_epoch, self.state.best_val_loss)
        return self.state

    # ----------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.model.cfg.__dict__,
            "epoch": self.state.epoch,
        }, path)

    @staticmethod
    def load_model(path: str | Path, device: str | None = None) -> EALSTM:
        from ..models.ealstm import EALSTMConfig
        device = device or training_device()
        ckpt = torch.load(path, map_location=device)
        cfg = EALSTMConfig(**ckpt["config"])
        model = EALSTM(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, obs = [], []
        for batch in loader:
            X, S, y = batch[0], batch[1], batch[2]
            X = X.to(self.device); S = S.to(self.device)
            yh = self.model(X, S).squeeze(-1).cpu().numpy()
            preds.append(yh)
            obs.append(y.numpy())
        return np.concatenate(preds), np.concatenate(obs)
