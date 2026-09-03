"""
train.py — Unified Training and Orchestration Entry Point

Usage:
    # 1. Run a single fold training:
    python train.py --config configs/resnet50_fold.yaml --fold 0 --seed 45

    # 2. Run the full cross-validation and seeds experiment sweep:
    python train.py --config configs/resnet50_fold.yaml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import rankdata

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils import (load_config, set_seed, load_checkpoint, get_device,
                       get_device_config)
from src.splits import get_splits
from src.dataset import (ImageDataset, get_transforms, EvenSpreadBatchSampler,
                         class_balanced_sample_weights)
from src.models import create_model
from src.losses import create_loss
from src.trainer import Trainer, predict_loader, resolve_balanced_mixup_alpha
from src.metrics import (center_normalized_official_score, fpr_at_tpr,
                         official_score, report)


OFFICIAL_BOOTSTRAP_SEED = 20260807


def positive_int(value: str) -> int:
    """Argparse type for options that only make sense above zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RARE26 model (Single Fold or Full Experiment)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--fold", type=int, default=None, help="Index of fold to validate on (None runs full CV sweep)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Reproducibility random seed (required with --fold)")
    parser.add_argument("--checkpoint-dir", type=str, default="models", help="Directory to save weights")
    parser.add_argument(
        "--limit-folds", type=positive_int, default=None,
        help="Diagnostic fold limit for full experiments",
    )
    parser.add_argument(
        "--limit-seeds", type=positive_int, default=None,
        help="Diagnostic seed limit for full experiments",
    )
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate config, complete seed x fold plan, and pretrained weights without training")
    return parser.parse_args()


def _checkpoint_path(checkpoint_dir: Path, exp_name: str, fold_idx: int, seed: int) -> Path:
    return checkpoint_dir / exp_name / f"{exp_name}_fold{fold_idx}_seed{seed}.pth"


def get_oof_predictions(config: dict, fold_idx: int, seed: int, val_df: pd.DataFrame,
                        checkpoint_dir: Path, device, device_config) -> tuple[np.ndarray, np.ndarray]:
    """Loads this fold/seed's canonical checkpoint and scores its validation rows."""
    exp_name = config.get("experiment_name", "experiment")
    checkpoint_path = _checkpoint_path(checkpoint_dir, exp_name, fold_idx, seed)

    model = create_model(config).to(device)

    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    val_transform = get_transforms(config, "val")
    val_dataset = ImageDataset(val_df, transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=device_config["num_workers"],
        pin_memory=device_config["pin_memory"]
    )

    return predict_loader(model, val_loader, device)


def _build_scheduler(optimizer, config: dict):
    """Cosine, optionally preceded by a linear warmup.

    warmup_epochs defaults to 0. When configured, a linear ramp precedes a
    cosine schedule spanning the remaining epochs.
    """
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"].get("warmup_epochs", 0) or 0)
    if warmup < 0 or warmup >= epochs:
        raise ValueError(
            f"training.warmup_epochs must be in [0, epochs); got {warmup} with epochs={epochs}."
        )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    if warmup == 0:
        return cosine
    ramp = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / (warmup + 1), end_factor=1.0, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[ramp, cosine], milestones=[warmup])


def _train_resolved_fold(*, config: dict, df: pd.DataFrame,
                         split: tuple[list[int], list[int]], fold_idx: int,
                         seed: int, checkpoint_dir: Path, device,
                         device_config: dict) -> dict:
    """Trains one already-resolved fold without re-reading runtime inputs."""
    set_seed(seed)

    exp_name = config.get("experiment_name", "experiment")
    print("=" * 60)
    print(f"EXPERIMENT: {exp_name} | FOLD: {fold_idx} | SEED: {seed}")
    print("=" * 60)

    print(f"Device: {device} | Config: {device_config}")

    train_idx, val_idx = split
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print("Dataset split completed:")
    print(f"  Train samples: {len(train_df)} (NEO: {np.sum(train_df['label'] == 1)}, NDBE: {np.sum(train_df['label'] == 0)})")
    print(f"  Val samples:   {len(val_df)} (NEO: {np.sum(val_df['label'] == 1)}, NDBE: {np.sum(val_df['label'] == 0)})")

    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")

    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=val_transform)

    # Avoids respawning workers every epoch; only valid when num_workers > 0.
    use_persistent_workers = device_config["num_workers"] > 0

    train_batch_sampler = EvenSpreadBatchSampler(
        train_df,
        batch_size=config["training"]["batch_size"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=device_config["num_workers"],
        pin_memory=device_config["pin_memory"],
        persistent_workers=use_persistent_workers
    )

    # Balanced-MixUp's second branch: same rows, drawn with replacement so every
    # class is equally likely; same length and batch size, so the two zip batch-for-batch.
    balanced_loader = None
    mixup_alpha = resolve_balanced_mixup_alpha(config)
    if mixup_alpha is not None:
        weights = class_balanced_sample_weights(train_df)
        balanced_loader = DataLoader(
            ImageDataset(train_df, transform=train_transform),
            batch_size=config["training"]["batch_size"],
            sampler=WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(train_df),
                replacement=True,
            ),
            num_workers=device_config["num_workers"],
            pin_memory=device_config["pin_memory"],
            persistent_workers=use_persistent_workers,
        )
        n_pos = int((train_df["label"] == 1).sum())
        p_i = n_pos / len(train_df)
        e_lam = mixup_alpha / (mixup_alpha + 1.0)
        e_y = (1.0 - e_lam) * p_i + e_lam * 0.5
        pos_weight = float(config["loss"].get("pos_weight", 1.0))
        # Report the effective class weighting without altering the config.
        print(f"  Balanced-MixUp: alpha={mixup_alpha}, E[lambda]={e_lam:.4f}, "
              f"effective positive rate {p_i:.4f} -> {e_y:.4f} ({e_y / p_i:.2f}x)")
        print(f"    pos_weight={pos_weight} gives positive/negative loss mass "
              f"{pos_weight * e_y / (1 - e_y):.2f} (1.00 = balanced, "
              f"{(1 - e_y) / e_y:.2f} would balance it)")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=device_config["num_workers"],
        pin_memory=device_config["pin_memory"],
        persistent_workers=use_persistent_workers
    )

    model = create_model(config).to(device)

    if device_config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    loss_fn = create_loss(config)
    loss_fn = loss_fn.to(device)

    # AdamW + cosine only, raises rather than silently substituting -- AdamW
    # normalises gradient scale, so loss-magnitude conclusions need it fixed.
    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    scheduler_name = config["training"].get("scheduler", "cosine").lower()
    if optimizer_name != "adamw" or scheduler_name != "cosine":
        raise ValueError(
            f"Only optimizer 'adamw' with scheduler 'cosine' is supported, got "
            f"{optimizer_name!r}/{scheduler_name!r}."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = _build_scheduler(optimizer, config)

    checkpoint_path = _checkpoint_path(checkpoint_dir, exp_name, fold_idx, seed)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        device_config=device_config,
        # Manifest row positions behind val_loader, in its (never shuffled) order --
        # lets the per-epoch curve pool across folds without re-deriving the split.
        val_row_ids=list(val_idx),
        balanced_loader=balanced_loader,
    )

    return trainer.fit(checkpoint_path)


def train_single_fold(config_path: str, fold_idx: int, seed: int,
                      checkpoint_dir: Path) -> dict:
    """Standalone single-fold entry point that resolves its own runtime inputs."""
    config = load_config(config_path)
    manifest_path = Path("data/data_manifest.csv")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    split_mode = config["data"].get("split_mode", "group_cv")
    n_folds = config["data"].get("n_folds", 5)
    splits = get_splits(df, split_mode, n_folds, seed)
    if fold_idx < 0 or fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for split mode "
            f"'{split_mode}' (has {len(splits)} folds)"
        )

    device = get_device()
    device_config = get_device_config(device)
    return _train_resolved_fold(
        config=config,
        df=df,
        split=splits[fold_idx],
        fold_idx=fold_idx,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        device=device,
        device_config=device_config,
    )


def validate_experiment_plan(config: dict, df: pd.DataFrame,
                             limit_seeds: int = None) -> tuple[list[int], dict[int, list]]:
    """Validates the complete seed x fold plan before any training starts.

    Full experiments must fail before fold 0 if a later requested seed lacks a
    pinned split column. The returned snapshot is used unchanged by training
    and OOF extraction throughout the orchestration run.
    """
    configured_seeds = config.get("seeds")
    if not isinstance(configured_seeds, list) or not configured_seeds:
        raise ValueError("Config 'seeds' must be a non-empty list of integers.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in configured_seeds):
        raise ValueError(f"Config 'seeds' contains a non-integer value: {configured_seeds}")
    if len(set(configured_seeds)) != len(configured_seeds):
        raise ValueError(f"Config 'seeds' contains duplicates: {configured_seeds}")

    if limit_seeds is not None:
        if isinstance(limit_seeds, bool) or not isinstance(limit_seeds, int) or limit_seeds <= 0:
            raise ValueError("limit_seeds must be a positive integer or None.")
        seeds = configured_seeds[:limit_seeds]
    else:
        seeds = configured_seeds

    split_mode = config["data"].get("split_mode", "group_cv")
    n_folds = config["data"].get("n_folds", 5)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds <= 0:
        raise ValueError(f"data.n_folds must be a positive integer, got {n_folds!r}.")
    if "group_id" in df.columns and df["group_id"].isna().any():
        raise ValueError("Manifest group_id contains missing values.")

    splits_by_seed = {}
    for seed in seeds:
        splits = get_splits(df, split_mode, n_folds, seed)
        if not splits:
            raise ValueError(f"No splits were produced for seed {seed} ({split_mode}).")

        all_indices = set(range(len(df)))

        if split_mode == "full":
            # There is no held-out data here by construction, so the overlap and
            # group-leakage guards below cannot apply. They are what makes every
            # other mode trustworthy and stay in force for those; this branch
            # checks the only thing that can go wrong in full mode instead --
            # that it really is one fold over the entire manifest.
            if len(splits) != 1:
                raise ValueError(
                    f"Full-data mode must produce exactly one fold; got {len(splits)}.")
            train_idx, val_idx = splits[0]
            if set(train_idx) != all_indices or set(val_idx) != all_indices:
                raise ValueError(
                    "Full-data mode must train on every manifest row and mirror "
                    "them as its validation set.")
            splits_by_seed[seed] = splits
            continue

        validation_counts = np.zeros(len(df), dtype=np.int64)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_set = set(train_idx)
            val_set = set(val_idx)
            if not train_set or not val_set:
                raise ValueError(f"Seed {seed} fold {fold_idx} has an empty train or validation split.")
            if len(train_set) != len(train_idx) or len(val_set) != len(val_idx):
                raise ValueError(f"Seed {seed} fold {fold_idx} contains duplicate row indices.")
            if train_set & val_set:
                raise ValueError(f"Seed {seed} fold {fold_idx} has train/validation index overlap.")
            if train_set | val_set != all_indices:
                raise ValueError(f"Seed {seed} fold {fold_idx} does not cover the complete manifest.")

            np.add.at(validation_counts, val_idx, 1)
            if "group_id" in df.columns:
                train_groups = set(df.iloc[train_idx]["group_id"])
                val_groups = set(df.iloc[val_idx]["group_id"])
                leaked_groups = train_groups & val_groups
                if leaked_groups:
                    preview = sorted(map(str, leaked_groups))[:10]
                    raise ValueError(
                        f"Seed {seed} fold {fold_idx} splits group_id values across train/validation: "
                        f"{preview}"
                    )

        if not np.all(validation_counts == 1):
            raise ValueError(
                f"Seed {seed} does not hold out every manifest row exactly once across its folds."
            )
        splits_by_seed[seed] = splits

    return seeds, splits_by_seed


def preflight_experiment(config_path: str, limit_seeds: int = None) -> None:
    """Runs all checks that can fail before an experiment, without training."""
    config = load_config(config_path)
    manifest_path = Path("data/data_manifest.csv")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    seeds, splits_by_seed = validate_experiment_plan(
        config=config,
        df=df,
        limit_seeds=limit_seeds,
    )

    # This verifies custom checkpoint existence, tensor shapes, and complete
    # backbone coverage. The model stays on CPU and no optimizer is created.
    create_model(config)

    fold_counts = {seed: len(splits_by_seed[seed]) for seed in seeds}
    print(f"Preflight passed: config={config_path} | seeds={seeds} | folds={fold_counts}")


def run_full_experiment(config_path: str, checkpoint_dir: str, limit_folds: int = None, limit_seeds: int = None):
    config = load_config(config_path)
    exp_name = config.get("experiment_name", "experiment")
    checkpoint_dir = Path(checkpoint_dir)
    is_limited_run = limit_folds is not None or limit_seeds is not None
    if limit_folds is not None:
        if isinstance(limit_folds, bool) or not isinstance(limit_folds, int) or limit_folds <= 0:
            raise ValueError("limit_folds must be a positive integer or None.")

    # Validate the complete execution plan before creating a device/model or
    # training the first fold.
    split_mode = config["data"].get("split_mode", "group_cv")
    n_folds = config["data"].get("n_folds", 5)
    manifest_path = Path("data/data_manifest.csv")
    if not manifest_path.exists():
        print("Manifest not found.")
        sys.exit(1)
    df = pd.read_csv(manifest_path)
    seeds, splits_by_seed = validate_experiment_plan(
        config=config,
        df=df,
        limit_seeds=limit_seeds,
    )
    center_arr = df["center"].to_numpy()

    device = get_device()
    device_config = get_device_config(device)

    print("=" * 60)
    print(f"STARTING EXPERIMENT ORCHESTRATION: {exp_name}")
    print(f"Folds: {n_folds} ({split_mode}) | Seeds: {seeds}")
    print("=" * 60)

    # seed -> {y_true, y_score, center, fpr_pooled, per_fold_fprs}
    seed_oof_scores = {}

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")

        # Every requested seed was resolved before orchestration started.
        splits = splits_by_seed[seed]
        folds_to_run = len(splits)
        if limit_folds is not None:
            folds_to_run = min(folds_to_run, limit_folds)

        seed_preds = np.zeros(len(df))
        seed_targets = np.zeros(len(df))
        validated_mask = np.zeros(len(df), dtype=bool)

        per_fold_fprs = []

        for fold in range(folds_to_run):
            _train_resolved_fold(
                config=config,
                df=df,
                split=splits[fold],
                fold_idx=fold,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                device=device,
                device_config=device_config,
            )

            _, val_idx = splits[fold]
            val_df = df.iloc[val_idx].reset_index(drop=True)

            targets, scores = get_oof_predictions(
                config=config,
                fold_idx=fold,
                seed=seed,
                val_df=val_df,
                checkpoint_dir=checkpoint_dir,
                device=device,
                device_config=device_config
            )

            fold_fpr = fpr_at_tpr(targets, scores, 0.90)
            per_fold_fprs.append(fold_fpr)

            # Rank-normalise within the fold before pooling, so folds with
            # different score scales cannot dominate the pooled curve.
            ranks = rankdata(scores) / len(scores)

            seed_preds[val_idx] = ranks
            seed_targets[val_idx] = targets
            validated_mask[val_idx] = True

        y_true_seed = seed_targets[validated_mask]
        y_score_seed = seed_preds[validated_mask]
        center_seed = center_arr[validated_mask]

        seed_fpr = fpr_at_tpr(y_true_seed, y_score_seed, 0.90)

        seed_oof_scores[seed] = {
            "y_true": y_true_seed,
            "y_score": y_score_seed,
            "center": center_seed,
            "fpr_pooled": seed_fpr,
            "per_fold_fprs": per_fold_fprs
        }

        print(f"\nSeed {seed} complete. OOF Pooled FPR@90: {seed_fpr:.6f}")
        print(f"Per-fold FPRs: {['%.4f' % f for f in per_fold_fprs]}")

    # Aggregate across all seeds (noise band & final stats).
    all_seed_fprs = [data["fpr_pooled"] for data in seed_oof_scores.values()]
    mean_oof_fpr = float(np.mean(all_seed_fprs))
    noise_band = float(np.std(all_seed_fprs)) if len(all_seed_fprs) > 1 else 0.0

    print("\n" + "=" * 60)
    print("LIMITED DIAGNOSTIC RUN COMPLETE" if is_limited_run else "EXPERIMENT ORCHESTRATION COMPLETE")
    if split_mode == "full":
        # Every number below is measured on training rows.
        print("  *** FULL-DATA MODE: metrics below are on TRAINING rows, NOT held out.")
        print("  *** They are diagnostics, not a generalization estimate.")
    print(f"  Experiment: {exp_name}")
    print(f"  OOF Pooled FPR@90: {mean_oof_fpr:.6f} ± {noise_band:.6f} (Noise Band)")
    print("=" * 60)

    # Report the mean over every configured seed.
    per_seed_reports = [report(seed_oof_scores[s]["y_true"], seed_oof_scores[s]["y_score"])
                        for s in seeds]
    eval_report = {k: float(np.mean([r[k] for r in per_seed_reports]))
                   for k in per_seed_reports[0]}

    print(f"  AUROC: {eval_report['auroc']:.6f} | AUPRC: {eval_report['auprc']:.6f}")
    print(f"  FPR@90: {eval_report['fpr@90']:.6f} | "
          f"composite: {eval_report['fpr_composite']:.6f}")

    def _score_all_seeds(fn):
        """Mean and spread over seeds, all sharing one bootstrap draw so
        comparisons between rows are paired and reproducible."""
        values = []
        for seed in seeds:
            np.random.seed(OFFICIAL_BOOTSTRAP_SEED)
            values.append(fn(seed_oof_scores[seed])["PPV@90RECALL"])
        return float(np.mean(values)), float(np.std(values))

    try:
        official_ppv, official_ppv_std = _score_all_seeds(
            lambda d: official_score(d["y_true"], d["y_score"]))
    except Exception as e:
        print(f"  ⚠️  Could not run official score script: {e}")
        official_ppv, official_ppv_std = None, None

    # Reported alongside official_ppv, never instead of it -- see
    # center_normalized_official_score()'s docstring.
    try:
        official_ppv_center_norm, _ = _score_all_seeds(
            lambda d: center_normalized_official_score(d["y_true"], d["y_score"], d["center"]))
    except Exception as e:
        print(f"  ⚠️  Could not run center-normalized official score: {e}")
        official_ppv_center_norm = None

    if official_ppv is None:
        print("  Official PPV@90Recall: unavailable")
    else:
        print(f"  Official PPV@90Recall: {official_ppv:.6f} ± {official_ppv_std:.6f} "
              f"(mean over {len(seeds)} seed(s))")
    if official_ppv_center_norm is None:
        print("  Center-normalized:     unavailable")
    else:
        print(f"  Center-normalized:     {official_ppv_center_norm:.6f}")

    # Each checkpoint's .json sidecar carries the merged config, epoch and
    # validation metrics.
    print(f"  Checkpoints: {checkpoint_dir}/{exp_name}")
    print("=" * 60)


def main():
    args = parse_args()

    if args.preflight_only:
        if args.fold is not None or args.limit_folds is not None:
            raise ValueError(
                "--preflight-only validates every fold and cannot be combined "
                "with --fold or --limit-folds."
            )
        preflight_experiment(
            config_path=args.config,
            limit_seeds=args.limit_seeds,
        )
    elif args.fold is not None:
        if args.limit_folds is not None or args.limit_seeds is not None:
            raise ValueError(
                "--fold selects one fold and cannot be combined with "
                "--limit-folds or --limit-seeds."
            )
        if args.seed is None:
            raise ValueError("--fold needs an explicit --seed; there is no default split.")
        train_single_fold(
            config_path=args.config,
            fold_idx=args.fold,
            seed=args.seed,
            checkpoint_dir=Path(args.checkpoint_dir)
        )
    else:
        run_full_experiment(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            limit_folds=args.limit_folds,
            limit_seeds=args.limit_seeds
        )


if __name__ == "__main__":
    main()
