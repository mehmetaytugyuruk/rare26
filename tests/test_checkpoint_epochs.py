"""Tests for explicit fixed-epoch checkpointing."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.trainer import Trainer, resolve_checkpoint_epochs


def _config(**training):
    base = {"epochs": 6, "batch_size": 2}
    base.update(training)
    return {"training": base, "model": {}, "loss": {"name": "bce"}}


class ResolveCheckpointEpochsTests(unittest.TestCase):
    def test_absent_key_is_rejected(self):
        """There is no fallback: a config without the key must not train."""
        with self.assertRaisesRegex(ValueError, "checkpoint_epochs is required"):
            resolve_checkpoint_epochs(_config(), 30)

    def test_sorted_and_deduplicated_into_a_tuple(self):
        self.assertEqual(resolve_checkpoint_epochs(_config(checkpoint_epochs=[30, 20, 25]), 30),
                         (20, 25, 30))

    def test_rejects_an_epoch_beyond_the_schedule(self):
        """Silently clamping would checkpoint an epoch that never runs."""
        with self.assertRaises(ValueError):
            resolve_checkpoint_epochs(_config(checkpoint_epochs=[20, 40]), 30)

    def test_rejects_duplicates_empty_and_non_integers(self):
        for bad in ([20, 20], [], [0], [-5], "20", [2.5], [True]):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                resolve_checkpoint_epochs(_config(checkpoint_epochs=bad), 30)


class _TinyDataset(Dataset):
    def __init__(self, n=8):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, generator=g)
        self.y = torch.tensor([0.0, 1.0] * (n // 2))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"image": self.x[i], "label": self.y[i]}


def _trainer(**training):
    model = nn.Linear(3, 1)
    loader = DataLoader(_TinyDataset(), batch_size=2)
    return Trainer(
        model=model, train_loader=loader, val_loader=loader,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None, device=torch.device("cpu"), config=_config(**training),
        val_row_ids=[10, 11, 12, 13, 14, 15, 16, 17],
    )


class FixedEpochFitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_one_checkpoint_per_listed_epoch_last_one_canonical(self):
        """Downstream readers such as get_oof_predictions resolve the
        canonical path, so the last listed epoch has to land there."""
        path = self.tmp / "exp_fold0_seed45.pth"
        _trainer(checkpoint_epochs=[2, 4, 6]).fit(path)

        self.assertTrue(path.exists(), "canonical path missing")
        self.assertTrue((self.tmp / "exp_fold0_seed45_ep2.pth").exists())
        self.assertTrue((self.tmp / "exp_fold0_seed45_ep4.pth").exists())
        self.assertFalse((self.tmp / "exp_fold0_seed45_ep6.pth").exists(),
                         "last epoch must not be written twice")

    def test_runs_every_epoch_when_only_the_last_is_checkpointed(self):
        """Training never stops short: the curve needs every epoch, not just the
        checkpointed ones, to show whether those epochs bracket the optimum."""
        path = self.tmp / "exp.pth"
        _trainer(checkpoint_epochs=[6]).fit(path)
        curve = np.load(self.tmp / "exp_valcurve.npz")
        self.assertEqual(len(curve["epochs"]), 6)
        self.assertEqual(curve["scores"].shape, (6, 8))

    def test_curve_records_every_epoch_not_just_checkpointed_ones(self):
        """The curve is what says whether the checkpointed epochs bracket the
        peak, so it has to cover epochs that were not checkpointed."""
        path = self.tmp / "exp.pth"
        _trainer(checkpoint_epochs=[4, 6]).fit(path)
        curve = np.load(self.tmp / "exp_valcurve.npz")
        np.testing.assert_array_equal(curve["epochs"], np.arange(1, 7))
        np.testing.assert_array_equal(curve["checkpoint_epochs"], [4, 6])
        np.testing.assert_array_equal(curve["val_row_ids"], [10, 11, 12, 13, 14, 15, 16, 17])
        np.testing.assert_array_equal(curve["targets"], [0, 1, 0, 1, 0, 1, 0, 1])

    def test_scores_differ_across_epochs(self):
        """A curve of identical rows would mean the scores were captured once
        and reused, which no shape assertion would catch."""
        path = self.tmp / "exp.pth"
        _trainer(checkpoint_epochs=[6]).fit(path)
        scores = np.load(self.tmp / "exp_valcurve.npz")["scores"]
        self.assertGreater(np.abs(np.diff(scores, axis=0)).sum(), 0.0)

    def test_returned_metrics_come_from_the_canonical_epoch(self):
        path = self.tmp / "exp.pth"
        returned = _trainer(checkpoint_epochs=[2, 6]).fit(path)
        saved = json.loads((self.tmp / "exp.json").read_text())
        self.assertEqual(saved["epoch"], 6)
        self.assertEqual(saved["selection"], "fixed_epoch")
        self.assertAlmostEqual(returned["auroc"], saved["val_metrics"]["auroc"])
        # The non-canonical epoch keeps its own sidecar.
        self.assertEqual(json.loads((self.tmp / "exp_ep2.json").read_text())["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
