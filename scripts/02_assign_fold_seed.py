"""
scripts/02_assign_fold_seed.py — Pin a StratifiedGroupKFold split to disk.

Writes a StratifiedGroupKFold assignment into data_manifest.csv as a
`fold_k{K}_seed{N}` column. src/splits.py reads this pinned column during
training instead of recomputing it.

K and the seed are both in the column name, so different (K, seed) pairs
coexist rather than overwrite each other.

The final method configs use k=5 with seeds 45, 46, and 47. Generate all three
through scripts/00_prepare_manifest.py. The scikit-learn version is printed so
it can be recorded with the private manifest.

Usage:
    python -m scripts.02_assign_fold_seed --seed 45                 # k=5 method split
    python -m scripts.02_assign_fold_seed --seed 42 --n-splits 10   # k=10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

def parse_args():
    p = argparse.ArgumentParser(description="Pin a StratifiedGroupKFold split to data_manifest.csv")
    p.add_argument("--manifest", type=Path, default=Path("data/data_manifest.csv"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing fold_k{K}_seed{N} column")
    return p.parse_args()


def main():
    args = parse_args()
    col = f"fold_k{args.n_splits}_seed{args.seed}"

    df = pd.read_csv(args.manifest)
    if "group_id" not in df.columns:
        print("group_id column missing. Run scripts/01_build_groups.py first.")
        sys.exit(1)
    if col in df.columns and not args.force:
        print(f"{col} already exists in {args.manifest}. Pass --force to overwrite "
              f"(this will desync any checkpoints already trained with "
              f"k={args.n_splits}/seed={args.seed}).")
        sys.exit(1)

    key = df["center"].astype(str) + "_" + df["label"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold = np.full(len(df), -1, dtype=int)
    for i, (_, val_idx) in enumerate(sgkf.split(df, y=key, groups=df["group_id"])):
        fold[val_idx] = i
    if not (fold >= 0).all():
        raise RuntimeError(
            f"{int((fold < 0).sum())} rows were left unassigned by StratifiedGroupKFold."
        )

    df[col] = fold
    df.to_csv(args.manifest, index=False)

    print(f"Wrote {col} to {args.manifest}")
    print(f"Fold sizes: {np.bincount(fold).tolist()}")
    print(f"scikit-learn version used: {sklearn.__version__} "
          f"(record this with the generated manifest)")


if __name__ == "__main__":
    main()
