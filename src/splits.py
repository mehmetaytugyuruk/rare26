"""
src/splits.py — Cross-validation split schemes.

Two modes:
1. "group_cv" -> reads the pinned `fold_k{K}_seed{N}` column from
   data_manifest.csv (grouped by group_id, stratified by center x label).
2. "full"     -> one fold, every row trains, nothing is held out.

The group_cv mode consumes precomputed fold columns so every run uses the same
grouped assignment. Generate them with scripts/02_assign_fold_seed.py.
"""

import numpy as np
import pandas as pd


def get_splits(df: pd.DataFrame, mode: str, n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Returns (train_indices, validation_indices) pairs for the given mode.

    mode="group_cv" reads the pinned fold_k{n_splits}_seed{seed} column.
    mode="full" ignores n_splits and returns one fold covering every row.
    """
    mode = mode.lower()

    if mode == "group_cv":
        # k and seed are both in the column name, so different fold counts coexist.
        col = f"fold_k{n_splits}_seed{seed}"
        if col not in df.columns:
            raise ValueError(
                f"{col} column is missing from manifest. Run "
                f"scripts/02_assign_fold_seed.py --seed {seed} --n-splits {n_splits} first."
            )
        fold = df[col].to_numpy()
        available = sorted(set(fold.tolist()))
        if available != list(range(n_splits)):
            raise ValueError(
                f"{col} has folds {available}, which is not 0..{n_splits - 1} -- "
                f"the column is corrupt, regenerate it."
            )

        print(f"  Splits: Reading pinned {n_splits}-fold group_cv split from {col}")
        splits = []
        for f in range(n_splits):
            val_idx = np.flatnonzero(fold == f).tolist()
            train_idx = np.flatnonzero(fold != f).tolist()
            splits.append((train_idx, val_idx))
        return splits

    elif mode == "full":
        # "Validation" mirrors training here, so these numbers only show whether
        # the run is learning, never how well -- see the module docstring.
        print("  Splits: full-data mode (1 fold, all rows train, NO held-out set)")
        idx = list(range(len(df)))
        return [(idx, list(idx))]

    else:
        raise ValueError(
            f"Unknown split mode: {mode}. Must be 'group_cv' or 'full'")
