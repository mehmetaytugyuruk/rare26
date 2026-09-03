import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.utils import load_config
from train import (
    _build_scheduler,
    main,
    run_full_experiment,
    train_single_fold,
    validate_experiment_plan,
)


class StopAfterResolvedFold(RuntimeError):
    """Test sentinel used to stop orchestration before OOF extraction."""


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {"split_mode": "group_cv", "n_folds": 3},
            "seeds": [42, 43],
        }
        self.manifest = pd.DataFrame({
            "center": ["center_1"] * 6,
            "group_id": [f"group_{i}" for i in range(6)],
            "fold_k3_seed42": [0, 1, 2, 0, 1, 2],
            "fold_k3_seed43": [2, 0, 1, 2, 0, 1],
        })

    def test_all_requested_seed_fold_columns_are_resolved(self):
        seeds, splits = validate_experiment_plan(self.config, self.manifest)

        self.assertEqual(seeds, [42, 43])
        self.assertEqual(set(splits), {42, 43})
        self.assertTrue(all(len(seed_splits) == 3 for seed_splits in splits.values()))

    def test_missing_later_seed_is_rejected_before_orchestration(self):
        manifest = self.manifest.drop(columns="fold_k3_seed43")

        with self.assertRaisesRegex(ValueError, "fold_k3_seed43 column is missing"):
            validate_experiment_plan(self.config, manifest)

    def test_non_positive_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "limit_seeds"):
            validate_experiment_plan(self.config, self.manifest, limit_seeds=0)
        with self.assertRaisesRegex(ValueError, "limit_folds"):
            run_full_experiment("configs/base.yaml", checkpoint_dir="models", limit_folds=0)

    def test_duplicate_seeds_are_rejected(self):
        config = {**self.config, "seeds": [42, 42]}

        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_experiment_plan(config, self.manifest)

    def test_group_split_across_folds_is_rejected(self):
        manifest = self.manifest.copy()
        manifest.loc[0, "group_id"] = "shared_group"
        manifest.loc[1, "group_id"] = "shared_group"

        with self.assertRaisesRegex(ValueError, "splits group_id"):
            validate_experiment_plan(self.config, manifest)

    def test_missing_group_id_is_rejected(self):
        manifest = self.manifest.copy()
        manifest.loc[0, "group_id"] = None

        with self.assertRaisesRegex(ValueError, "group_id contains missing"):
            validate_experiment_plan(self.config, manifest)

    def test_submitted_config_resolves_all_expected_cv_splits(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(str(project_root / "configs/resnet50_fold.yaml"))
        manifest = pd.DataFrame({
            "center": ["center_1"] * 10,
            "group_id": [f"group_{i}" for i in range(10)],
            "fold_k5_seed45": [0, 1, 2, 3, 4] * 2,
            "fold_k5_seed46": [1, 2, 3, 4, 0] * 2,
            "fold_k5_seed47": [2, 3, 4, 0, 1] * 2,
        })

        seeds, splits = validate_experiment_plan(config, manifest)

        self.assertEqual(seeds, [45, 46, 47])
        self.assertEqual([len(splits[seed]) for seed in seeds], [5, 5, 5])

    @patch("train.get_device_config", return_value={"num_workers": 0})
    @patch("train.get_device", return_value="cpu")
    @patch("train.get_splits")
    @patch("train.pd.read_csv")
    @patch("train.load_config")
    @patch("train._train_resolved_fold", return_value={"status": "ok"})
    def test_single_fold_wrapper_resolves_inputs_once_and_passes_them_to_core(
            self, train_fold, load_config_mock, read_csv, get_splits,
            _get_device, _get_device_config):
        resolved_split = ([1, 2], [0])
        load_config_mock.return_value = self.config
        read_csv.return_value = self.manifest
        get_splits.return_value = [resolved_split]

        result = train_single_fold(
            "configs/base.yaml",
            fold_idx=0,
            seed=42,
            checkpoint_dir=Path("models"),
        )

        self.assertEqual(result, {"status": "ok"})
        self.assertIs(train_fold.call_args.kwargs["config"], self.config)
        self.assertIs(train_fold.call_args.kwargs["df"], self.manifest)
        self.assertIs(train_fold.call_args.kwargs["split"], resolved_split)
        get_splits.assert_called_once_with(self.manifest, "group_cv", 3, 42)

    @patch("train.get_device")
    @patch("train._train_resolved_fold")
    def test_missing_later_seed_stops_before_device_or_training(self, train_fold, get_device):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "missing_seed.yaml"
            config_path.write_text("seeds: [45, 999]\n")

            with self.assertRaisesRegex(ValueError, "fold_k5_seed999 column is missing"):
                run_full_experiment(str(config_path), checkpoint_dir="models")

        train_fold.assert_not_called()
        get_device.assert_not_called()

    @patch("train.get_device_config", return_value={})
    @patch("train.get_device", return_value="cpu")
    @patch("train._train_resolved_fold", side_effect=StopAfterResolvedFold)
    @patch("train.validate_experiment_plan")
    def test_orchestration_passes_validated_split_snapshot_to_training(
            self, validate_plan, train_fold, _get_device, _get_device_config):
        resolved_split = ([1, 2], [0])
        validate_plan.return_value = ([42], {42: [resolved_split]})

        with self.assertRaises(StopAfterResolvedFold):
            run_full_experiment(
                "configs/base.yaml",
                checkpoint_dir="models",
                limit_folds=1,
                limit_seeds=1,
            )

        validate_plan.assert_called_once()
        self.assertIs(train_fold.call_args.kwargs["split"], resolved_split)
        self.assertIs(
            train_fold.call_args.kwargs["df"],
            validate_plan.call_args.kwargs["df"],
        )

    def test_limited_run_scores_the_folds_it_actually_trained(self):
        """A --limit-folds run must still pool OOF over exactly the folds it
        ran, against the validation rows of the split it was handed."""
        config = {
            "experiment_name": "limited_test",
            "data": {"split_mode": "group_cv", "n_folds": 3},
            "seeds": [42],
        }
        manifest = pd.DataFrame({"center": ["c1", "c1", "c1", "c1"]})
        resolved_split = ([2, 3], [0, 1])
        targets = np.array([0, 1])
        scores = np.array([0.1, 0.9])

        with (
            patch("train.load_config", return_value=config),
            patch("train.pd.read_csv", return_value=manifest),
            patch(
                "train.validate_experiment_plan",
                return_value=([42], {42: [resolved_split]}),
            ),
            patch("train.get_device", return_value="cpu"),
            patch("train.get_device_config", return_value={}),
            patch("train._train_resolved_fold") as train_fold,
            patch(
                "train.get_oof_predictions",
                autospec=True,
                return_value=(targets, scores),
            ) as get_oof,
            patch("train.official_score", return_value={"PPV@90RECALL": 0.5}),
            patch(
                "train.center_normalized_official_score",
                return_value={"PPV@90RECALL": 0.5},
            ),
        ):
            run_full_experiment(
                "configs/base.yaml",
                checkpoint_dir="models",
                limit_folds=1,
            )

        self.assertIs(train_fold.call_args.kwargs["split"], resolved_split)
        self.assertEqual(get_oof.call_count, 1)
        pd.testing.assert_frame_equal(
            get_oof.call_args.kwargs["val_df"],
            manifest.iloc[[0, 1]].reset_index(drop=True),
        )
        self.assertEqual(get_oof.call_args.kwargs["checkpoint_dir"], Path("models"))

    def test_incompatible_cli_combinations_are_rejected_before_any_work(self):
        """--preflight-only and --fold each exclude the full-experiment limits,
        and main() must refuse before dispatching either entry point."""
        cases = {
            "preflight_experiment": dict(fold=None, limit_folds=1, limit_seeds=None,
                                         preflight_only=True),
            "train_single_fold": dict(fold=0, limit_folds=None, limit_seeds=1,
                                      preflight_only=False),
        }
        for entry_point, args in cases.items():
            with self.subTest(entry_point=entry_point):
                with (
                    patch(f"train.{entry_point}") as dispatched,
                    patch("train.parse_args", return_value=SimpleNamespace(
                        config="configs/base.yaml", seed=42,
                        checkpoint_dir="models", **args)),
                    self.assertRaisesRegex(ValueError, "cannot be combined"),
                ):
                    main()
                dispatched.assert_not_called()


class SchedulerTests(unittest.TestCase):
    """Guards the warmup path. It runs only for the ViT arm, so a regression here
    would show up as a silently different schedule on the run that uses it."""

    @staticmethod
    def _lrs(warmup, epochs=30, lr=1e-5):
        import torch, warnings
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = _build_scheduler(opt, {"training": {"epochs": epochs,
                                                    "warmup_epochs": warmup}})
        out = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(epochs):
                out.append(opt.param_groups[0]["lr"])
                sched.step()
        return out

    def test_absent_warmup_reproduces_plain_cosine(self):
        """A zero warmup must match the configured plain cosine schedule."""
        import torch, warnings
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        ref_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        ref = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(30):
                ref.append(opt.param_groups[0]["lr"])
                ref_sched.step()
        for a, b in zip(ref, self._lrs(0, 30, 5e-5)):
            self.assertAlmostEqual(a, b, places=15)

    def test_warmup_ramps_to_the_configured_lr_then_decays(self):
        lrs = self._lrs(3, epochs=30, lr=1e-5)
        self.assertAlmostEqual(max(lrs), 1e-5, places=12)
        self.assertEqual(lrs.index(max(lrs)), 3, "peak must land at the end of the ramp")
        self.assertTrue(all(lrs[i] < lrs[i + 1] for i in range(3)), "ramp must rise")
        self.assertTrue(all(lrs[i] >= lrs[i + 1] for i in range(3, 29)), "then decay")
        self.assertLess(lrs[0], 1e-5)

    def test_out_of_range_warmup_is_rejected(self):
        for bad in (-1, 30, 31):
            with self.assertRaises(ValueError, msg=f"accepted warmup_epochs={bad}"):
                self._lrs(bad, epochs=30)


class FullDataModeTests(unittest.TestCase):
    """Guards the full-data split mode and, more importantly, that adding it did
    not weaken the leakage guards every other mode depends on."""

    def setUp(self):
        self.df = pd.DataFrame({
            "center": ["center_1"] * 4 + ["center_2"] * 2,
            "label": [0, 1, 0, 1, 0, 1],
            "group_id": [f"g{i}" for i in range(6)],
            "fold_k3_seed42": [0, 1, 2, 0, 1, 2],
        })

    def test_full_mode_is_one_fold_over_every_row_with_val_mirroring_train(self):
        from src.splits import get_splits
        splits = get_splits(self.df, "full", 5, 42)
        self.assertEqual(len(splits), 1)
        train_idx, val_idx = splits[0]
        self.assertEqual(sorted(train_idx), list(range(6)))
        self.assertEqual(sorted(val_idx), list(range(6)))

    def test_plan_accepts_full_mode_despite_total_overlap(self):
        config = {"data": {"split_mode": "full", "n_folds": 5}, "seeds": [45, 46]}
        seeds, splits = validate_experiment_plan(config, self.df)
        self.assertEqual(seeds, [45, 46])
        self.assertEqual([len(splits[s]) for s in seeds], [1, 1])

    def test_group_leakage_is_still_rejected_for_cross_validation(self):
        """The full-data branch must not have loosened this for group_cv."""
        df = self.df.copy()
        df.loc[0, "group_id"] = "shared"
        df.loc[1, "group_id"] = "shared"
        config = {"data": {"split_mode": "group_cv", "n_folds": 3}, "seeds": [42]}
        with self.assertRaisesRegex(ValueError, "splits group_id"):
            validate_experiment_plan(config, df)

    def test_train_val_overlap_is_still_rejected_for_cross_validation(self):
        config = {"data": {"split_mode": "group_cv", "n_folds": 3}, "seeds": [42]}
        overlapping = [([0, 1, 2], [2, 3, 4, 5])]
        with patch("train.get_splits", return_value=overlapping):
            with self.assertRaisesRegex(ValueError, "train/validation index overlap"):
                validate_experiment_plan(config, self.df)

    def test_full_mode_rejects_a_split_that_is_not_the_whole_manifest(self):
        config = {"data": {"split_mode": "full", "n_folds": 5}, "seeds": [45]}
        partial = [([0, 1, 2], [0, 1, 2])]
        with patch("train.get_splits", return_value=partial):
            with self.assertRaisesRegex(ValueError, "every manifest row"):
                validate_experiment_plan(config, self.df)

    def test_full_mode_rejects_more_than_one_fold(self):
        config = {"data": {"split_mode": "full", "n_folds": 5}, "seeds": [45]}
        two = [(list(range(6)), list(range(6))), (list(range(6)), list(range(6)))]
        with patch("train.get_splits", return_value=two):
            with self.assertRaisesRegex(ValueError, "exactly one fold"):
                validate_experiment_plan(config, self.df)


if __name__ == "__main__":
    unittest.main()
