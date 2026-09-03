"""
src/trainer.py — Fixed-epoch training loop and diagnostics.
"""

import math
from numbers import Real
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import report
from src.utils import save_checkpoint


def resolve_checkpoint_epochs(config: dict, epochs: int) -> tuple[int, ...]:
    """Validate and normalize the explicit fixed-epoch checkpoint schedule."""
    raw = config.get("training", {}).get("checkpoint_epochs")
    if raw is None:
        raise ValueError(
            "training.checkpoint_epochs is required; there is no implicit default."
        )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            f"training.checkpoint_epochs must be a non-empty list of epochs; got {raw!r}."
        )
    resolved = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"training.checkpoint_epochs entries must be positive integers; got {value!r}."
            )
        if value > epochs:
            raise ValueError(
                f"training.checkpoint_epochs entry {value} exceeds training.epochs {epochs}."
            )
        resolved.append(value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"training.checkpoint_epochs contains duplicates: {raw!r}.")
    return tuple(sorted(resolved))


def resolve_balanced_mixup_alpha(config: dict) -> float | None:
    """Validates training.balanced_mixup_alpha, or None when absent.

    Balanced-MixUp (Galdran et al., MICCAI 2021) draws one batch under natural
    sampling and one under class-balanced sampling, then blends them:

        image = (1 - lam) * natural + lam * balanced
        label = (1 - lam) * y_natural + lam * y_balanced,   lam ~ Beta(alpha, 1)

    The configured method uses alpha=0.2 and treats lambda as the contribution
    of the class-balanced branch.
    """
    raw = config.get("training", {}).get("balanced_mixup_alpha")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ValueError(
            f"training.balanced_mixup_alpha must be a number; got {raw!r}."
        )
    alpha = float(raw)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"training.balanced_mixup_alpha must be finite and positive; got {raw!r}."
        )
    return alpha


def balanced_mixup_batch(natural, balanced, alpha: float, rng) -> tuple:
    """Blends one natural batch with one class-balanced batch.

    Returns (images, soft_labels, lam). A single lambda is drawn per batch, as
    in the reference implementation, not per sample.

    The paired loaders must agree on batch size at every step; a mismatch is
    an orchestration bug, not something to silently truncate around.
    """
    x_nat, y_nat = natural
    x_bal, y_bal = balanced
    n_nat = int(x_nat.shape[0])
    n_bal = int(x_bal.shape[0])
    if n_nat != n_bal:
        raise ValueError(
            "Balanced-MixUp paired loaders produced different batch sizes: "
            f"natural={n_nat}, balanced={n_bal}."
        )
    if int(y_nat.shape[0]) != n_nat or int(y_bal.shape[0]) != n_bal:
        raise ValueError(
            "Balanced-MixUp image/label batch sizes disagree within a branch."
        )

    lam = float(rng.beta(alpha, 1.0))
    images = (1.0 - lam) * x_nat + lam * x_bal
    labels = (1.0 - lam) * y_nat + lam * y_bal
    return images, labels, lam


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 loss_fn: nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                 device: torch.device, config: dict, device_config: dict = None,
                 val_row_ids=None, balanced_loader: DataLoader = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.epochs = config["training"]["epochs"]
        self.checkpoint_epochs = resolve_checkpoint_epochs(config, self.epochs)
        # Manifest row positions behind the val loader (never shuffled), recorded
        # so the per-epoch curve can pool across folds without re-deriving splits.
        self.val_row_ids = (
            None if val_row_ids is None
            else np.asarray(val_row_ids, dtype=np.int64)
        )
        # validate() writes raw scores here for fit() to accumulate -- kept off
        # its return value, which gets serialised into the checkpoint sidecar.
        self._last_val_scores = None
        self._last_val_targets = None

        # Refuses to silently train an ordinary run when the config asked for
        # Balanced-MixUp but didn't provide the second, class-balanced loader.
        self.balanced_mixup_alpha = resolve_balanced_mixup_alpha(config)
        self.balanced_loader = balanced_loader
        if self.balanced_mixup_alpha is not None:
            if balanced_loader is None:
                raise ValueError(
                    "training.balanced_mixup_alpha is set but no balanced_loader "
                    "was provided; Balanced-MixUp needs both branches."
                )
            if len(self.train_loader) != len(balanced_loader):
                raise ValueError(
                    "Balanced-MixUp loaders must have the same number of batches: "
                    f"natural={len(self.train_loader)}, balanced={len(balanced_loader)}."
                )
            # Seeded off the run's global seed rather than a fixed constant, so
            # the three seeds differ in their lambda stream too.
            self._mixup_rng = np.random.default_rng(
                np.random.SeedSequence([int(torch.initial_seed() % (2 ** 31)), 7])
            )

        device_config = device_config or {}
        self.use_amp = device_config.get("use_amp", False)
        self.channels_last = device_config.get("channels_last", False)
        # GradScaler is CUDA-specific; use_amp is only ever True for device_config["cuda"]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _prepare_inputs(self, batch: dict) -> torch.Tensor:
        inputs = batch["image"].to(self.device)
        if self.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        return inputs

    def _gradient_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Forward, backward and optimizer step for one batch. Returns the loss."""
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def train_one_epoch(self) -> float:
        self.model.train()
        if self.balanced_mixup_alpha is not None:
            return self._train_one_epoch_balanced_mixup()

        total_loss = 0.0
        n_batches = len(self.train_loader)
        for batch in self.train_loader:
            inputs = self._prepare_inputs(batch)
            targets = batch["label"].to(self.device).unsqueeze(1)
            total_loss += self._gradient_step(inputs, targets)

        return total_loss / n_batches

    def _train_one_epoch_balanced_mixup(self) -> float:
        """One epoch of Balanced-MixUp: every batch is a blend of both branches.

        No image reaches the model unmixed, but at alpha=0.2 the median lambda
        is 0.03, so most of an epoch is the natural batch essentially untouched
        with a minority of strongly blended batches. Labels move with the
        images, making the targets soft.
        """
        total_loss = 0.0
        n_batches = 0

        for natural, balanced in zip(self.train_loader, self.balanced_loader):
            images, labels, _ = balanced_mixup_batch(
                (natural["image"], natural["label"]),
                (balanced["image"], balanced["label"]),
                alpha=self.balanced_mixup_alpha,
                rng=self._mixup_rng,
            )
            inputs = images.to(self.device)
            if self.channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)
            # BCEWithLogitsLoss takes these soft labels directly; pos_weight
            # still multiplies the positive term (see src/losses.py for 18.6).
            targets = labels.to(self.device).unsqueeze(1)
            total_loss += self._gradient_step(inputs, targets)
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("Balanced-MixUp epoch produced no batches.")
        expected_batches = len(self.train_loader)
        if n_batches != expected_batches:
            raise RuntimeError(
                "Balanced-MixUp stopped before consuming both loaders: "
                f"processed={n_batches}, expected={expected_batches}."
            )
        return total_loss / n_batches

    def validate(self) -> tuple[float, dict]:
        """Runs validation and evaluates using the diagnostics metrics report."""
        self.model.eval()

        total_loss = 0.0
        all_targets = []
        all_scores = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = self._prepare_inputs(batch)
                targets = batch["label"].to(self.device).unsqueeze(1)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                scores = torch.sigmoid(outputs.float())

                all_targets.extend(targets.cpu().numpy().flatten())
                all_scores.extend(scores.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)

        y_true = np.array(all_targets)
        y_score = np.array(all_scores)
        self._last_val_targets = y_true
        self._last_val_scores = y_score

        metrics_report = report(y_true, y_score)

        return avg_loss, metrics_report

    def _epoch_checkpoint_path(self, checkpoint_path: Path, epoch: int) -> Path:
        """Where one fixed-epoch checkpoint goes.

        The last listed epoch takes the canonical path so get_oof_predictions()
        and downstream readers use one stable path; the other listed epochs get
        an _ep{N} suffix. Nothing is written twice.
        """
        if epoch == self.checkpoint_epochs[-1]:
            return checkpoint_path
        return checkpoint_path.with_name(f"{checkpoint_path.stem}_ep{epoch}{checkpoint_path.suffix}")

    def fit(self, checkpoint_path: Path) -> dict:
        """Trains every epoch, checkpoints the listed ones, records the curve."""
        print(f"\nTrainer: checkpointing epochs "
              f"{list(self.checkpoint_epochs)} of {self.epochs}.")

        curve, reports = [], {}
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.validate()
            if self.scheduler:
                self.scheduler.step()

            curve.append(self._last_val_scores)
            print(f"Epoch {epoch:02d}/{self.epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | FPR@90: {val_metrics['fpr@90']:.4f} | "
                  f"AUROC: {val_metrics['auroc']:.4f}")

            if epoch in self.checkpoint_epochs:
                target = self._epoch_checkpoint_path(checkpoint_path, epoch)
                meta = {
                    "epoch": epoch,
                    "selection": "fixed_epoch",
                    "checkpoint_epochs": list(self.checkpoint_epochs),
                    "val_metrics": val_metrics,
                    "config": self.config,
                }
                save_checkpoint(self.model, target, meta=meta)
                reports[epoch] = val_metrics
                print(f"  ✓ Epoch {epoch} checkpoint saved to {target}")

        # Every epoch's scores, not just the checkpointed ones: this is what shows
        # whether the checkpointed epochs bracket the peak; costs kilobytes.
        curve_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_valcurve.npz")
        payload = {
            "epochs": np.arange(1, self.epochs + 1, dtype=np.int64),
            "scores": np.asarray(curve, dtype=np.float32),
            "targets": np.asarray(self._last_val_targets, dtype=np.int64),
            "checkpoint_epochs": np.asarray(self.checkpoint_epochs, dtype=np.int64),
        }
        if self.val_row_ids is not None:
            payload["val_row_ids"] = self.val_row_ids
        np.savez_compressed(curve_path, **payload)
        print(f"  ✓ Per-epoch validation curve saved to {curve_path}")

        return reports[self.checkpoint_epochs[-1]]


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over loader and returns (y_true, y_score) as NumPy arrays."""
    model.eval()
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            targets = batch["label"]
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)

            all_targets.extend(targets.numpy().flatten())
            all_scores.extend(scores.cpu().numpy().flatten())

    return np.array(all_targets), np.array(all_scores)
