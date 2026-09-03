"""Tests for Balanced-MixUp configuration, sampling, and batch alignment."""

import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import EvenSpreadBatchSampler, class_balanced_sample_weights
from src.trainer import Trainer, balanced_mixup_batch, resolve_balanced_mixup_alpha


def _config(**training):
    return {"training": training, "model": {}, "loss": {"name": "bce"}}


class ResolveAlphaTests(unittest.TestCase):
    def test_absent_key_disables_mixup(self):
        self.assertIsNone(resolve_balanced_mixup_alpha(_config()))

    def test_accepts_a_positive_number(self):
        self.assertAlmostEqual(resolve_balanced_mixup_alpha(_config(balanced_mixup_alpha=0.2)), 0.2)

    def test_rejects_zero_negative_nan_and_non_numbers(self):
        for bad in (0, -0.2, float("nan"), float("inf"), "0.2", True, [0.2]):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                resolve_balanced_mixup_alpha(_config(balanced_mixup_alpha=bad))


class ClassBalancedWeightTests(unittest.TestCase):
    def test_each_class_carries_equal_total_weight(self):
        df = pd.DataFrame({"label": [0] * 2937 + [1] * 158})
        w = class_balanced_sample_weights(df)
        label = df["label"].to_numpy()
        self.assertAlmostEqual(w[label == 0].sum(), w[label == 1].sum(), places=10)

    def test_a_rare_positive_is_drawn_far_more_often_than_a_common_negative(self):
        df = pd.DataFrame({"label": [0] * 2937 + [1] * 158})
        w = class_balanced_sample_weights(df)
        label = df["label"].to_numpy()
        self.assertAlmostEqual(w[label == 1][0] / w[label == 0][0], 2937 / 158, places=6)

    def test_drawn_batches_are_about_half_positive(self):
        """The whole point of the second branch: ~50% positive against 5.1%."""
        df = pd.DataFrame({"label": [0] * 2937 + [1] * 158})
        w = class_balanced_sample_weights(df)
        g = torch.Generator().manual_seed(0)
        drawn = torch.multinomial(torch.as_tensor(w), 20000, replacement=True, generator=g)
        rate = df["label"].to_numpy()[drawn.numpy()].mean()
        self.assertAlmostEqual(rate, 0.5, delta=0.02)


class EvenSpreadBatchSamplerTests(unittest.TestCase):
    def test_only_incomplete_batch_is_yielded_last(self):
        df = pd.DataFrame(
            {
                "center": ["center_1"] * 10,
                "label": [0] * 10,
                "group_id": np.arange(10),
            }
        )
        sampler = EvenSpreadBatchSampler(df, batch_size=4)

        for _ in range(20):
            batches = list(iter(sampler))
            self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
            self.assertEqual(sorted(index for batch in batches for index in batch), list(range(10)))

    def test_divisible_epoch_has_only_full_batches(self):
        df = pd.DataFrame(
            {
                "center": ["center_1"] * 8,
                "label": [0] * 8,
                "group_id": np.arange(8),
            }
        )
        batches = list(EvenSpreadBatchSampler(df, batch_size=4))
        self.assertEqual([len(batch) for batch in batches], [4, 4])
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(8)))

    def test_smaller_than_batch_size_is_one_short_batch(self):
        df = pd.DataFrame(
            {
                "center": ["center_1"] * 3,
                "label": [0] * 3,
                "group_id": np.arange(3),
            }
        )
        batches = list(EvenSpreadBatchSampler(df, batch_size=4))
        self.assertEqual([len(batch) for batch in batches], [3])
        self.assertEqual(sorted(batches[0]), list(range(3)))

    def test_batch_size_one_preserves_exact_coverage(self):
        df = pd.DataFrame(
            {
                "center": ["center_1"] * 4,
                "label": [0] * 4,
                "group_id": np.arange(4),
            }
        )
        batches = list(EvenSpreadBatchSampler(df, batch_size=1))
        self.assertEqual([len(batch) for batch in batches], [1, 1, 1, 1])
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(4)))


class MixBatchTests(unittest.TestCase):
    def test_lambda_weights_the_balanced_branch_not_the_natural_one(self):
        """Lambda is the contribution of the class-balanced sample."""
        natural = (torch.zeros(4, 3, 2, 2), torch.zeros(4))
        balanced = (torch.ones(4, 3, 2, 2), torch.ones(4))

        class _FixedRng:
            def __init__(self, value):
                self.value = value

            def beta(self, a, b):
                return self.value

        images, labels, lam = balanced_mixup_batch(natural, balanced, 0.2, _FixedRng(0.25))
        self.assertAlmostEqual(lam, 0.25)
        # lam is the balanced share, so a lam of 0.25 leaves 75% of the natural
        # (all-zero) sample: the result is 0.25, not 0.75.
        self.assertAlmostEqual(float(images.mean()), 0.25, places=6)
        self.assertAlmostEqual(float(labels.mean()), 0.25, places=6)

    def test_rejects_different_branch_sizes_instead_of_discarding_samples(self):
        natural = (torch.zeros(3, 3, 2, 2), torch.zeros(3))
        balanced = (torch.ones(8, 3, 2, 2), torch.ones(8))
        with self.assertRaisesRegex(ValueError, "different batch sizes"):
            balanced_mixup_batch(natural, balanced, 0.2, np.random.default_rng(0))

    def test_rejects_image_label_size_mismatch_within_a_branch(self):
        natural = (torch.zeros(4, 3, 2, 2), torch.zeros(3))
        balanced = (torch.ones(4, 3, 2, 2), torch.ones(4))
        with self.assertRaisesRegex(ValueError, "image/label"):
            balanced_mixup_batch(natural, balanced, 0.2, np.random.default_rng(0))

    def test_equal_short_branches_are_valid(self):
        natural = (torch.zeros(3, 3, 2, 2), torch.zeros(3))
        balanced = (torch.ones(3, 3, 2, 2), torch.ones(3))
        images, labels, _ = balanced_mixup_batch(
            natural, balanced, 0.2, np.random.default_rng(0)
        )
        self.assertEqual(images.shape[0], 3)
        self.assertEqual(labels.shape[0], 3)

    def test_labels_track_the_images(self):
        """A blend whose label did not move with its pixels would be mislabelled
        training data, and nothing downstream would notice."""
        rng = np.random.default_rng(0)
        natural = (torch.zeros(6, 3, 2, 2), torch.zeros(6))
        balanced = (torch.ones(6, 3, 2, 2), torch.ones(6))
        for _ in range(20):
            images, labels, lam = balanced_mixup_batch(natural, balanced, 0.2, rng)
            self.assertAlmostEqual(float(images.mean()), float(labels.mean()), places=5)
            self.assertAlmostEqual(float(labels.mean()), lam, places=5)

    def test_one_lambda_per_batch_not_per_sample(self):
        """Per-sample lambdas would be a different variance regime, not the
        published method."""
        natural = (torch.zeros(16, 3, 2, 2), torch.zeros(16))
        balanced = (torch.ones(16, 3, 2, 2), torch.ones(16))
        _, labels, _ = balanced_mixup_batch(
            natural, balanced, 0.2, np.random.default_rng(0)
        )
        self.assertAlmostEqual(float(labels.std()), 0.0, places=6)

    def test_alpha_controls_the_expected_blend(self):
        """E[lambda] = alpha/(alpha+1) is what sets the effective positive rate,
        and therefore the pos_weight that balances it."""
        for alpha in (0.1, 0.2, 0.3):
            rng = np.random.default_rng(0)
            natural = (torch.zeros(2, 1, 1, 1), torch.zeros(2))
            balanced = (torch.ones(2, 1, 1, 1), torch.ones(2))
            lams = [balanced_mixup_batch(natural, balanced, alpha, rng)[2] for _ in range(40000)]
            self.assertAlmostEqual(float(np.mean(lams)), alpha / (alpha + 1), delta=0.01)


class PairedLoaderContractTests(unittest.TestCase):
    def test_trainer_rejects_different_loader_lengths(self):
        natural_loader = DataLoader(
            TensorDataset(torch.zeros(8, 1), torch.zeros(8)), batch_size=2
        )
        balanced_loader = DataLoader(
            TensorDataset(torch.zeros(6, 1), torch.zeros(6)), batch_size=2
        )
        model = nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "same number of batches"):
            Trainer(
                model=model,
                train_loader=natural_loader,
                val_loader=natural_loader,
                loss_fn=nn.BCEWithLogitsLoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                scheduler=None,
                device=torch.device("cpu"),
                config={
                    "training": {"epochs": 1, "balanced_mixup_alpha": 0.2,
                                 "checkpoint_epochs": [1]},
                    "model": {},
                    "loss": {"name": "bce"},
                },
                balanced_loader=balanced_loader,
            )


class DerivedPosWeightTests(unittest.TestCase):
    def test_configured_pos_weight_relative_to_the_balancing_value(self):
        """At alpha=0.2, pos_weight=18.6 gives 2.68x positive loss mass."""
        p_i, alpha = 158 / 3095, 0.2
        e_lam = alpha / (alpha + 1)
        e_y = (1 - e_lam) * p_i + e_lam * 0.5

        self.assertAlmostEqual(e_y, 0.1259, places=4)
        self.assertAlmostEqual(e_y / p_i, 2.47, places=2)
        self.assertAlmostEqual((1 - e_y) / e_y, 6.94, places=2)
        for pos_weight, expected in ((1.0, 0.14), (6.94, 1.00), (18.6, 2.68)):
            self.assertAlmostEqual(pos_weight * e_y / (1 - e_y), expected, places=2)


if __name__ == "__main__":
    unittest.main()
