"""Build the private training manifest from an extracted RARE26 dataset.

Expected input layout::

    data/
      center_1/{ndbe,neo}/*.png
      center_2/{ndbe,neo}/*.png

By default the script creates ``data/data_manifest.csv``, computes DINOv2
near-duplicate groups, and pins the five folds used by the training configs for
seeds 45, 46, and 47. The generated CSV contains challenge-derived labels and
metadata and is intentionally excluded from Git.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


CENTERS = ("center_1", "center_2")
CLASS_LABELS = {"ndbe": 0, "neo": 1}
BASE_COLUMNS = [
    "path",
    "label",
    "center",
    "class_name",
    "filename",
    "width",
    "height",
    "format",
]


def build_base_manifest(data_root: Path, path_prefix: Path = Path("data")) -> pd.DataFrame:
    """Scan the expected dataset folders and return a stable manifest table."""
    rows = []
    for center in CENTERS:
        for class_name, label in CLASS_LABELS.items():
            class_dir = data_root / center / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected dataset directory is missing: {class_dir}")

            image_paths = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            )
            if not image_paths:
                raise FileNotFoundError(f"No PNG images found in: {class_dir}")

            for image_path in image_paths:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format

                relative_path = Path(center) / class_name / image_path.name
                rows.append({
                    "path": (path_prefix / relative_path).as_posix(),
                    "label": label,
                    "center": center,
                    "class_name": class_name,
                    "filename": image_path.name,
                    "width": width,
                    "height": height,
                    "format": image_format,
                })

    return pd.DataFrame(rows, columns=BASE_COLUMNS).sort_values("path").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the private RARE26 manifest, groups, and pinned folds"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("data/data_manifest.csv"))
    parser.add_argument(
        "--path-prefix",
        type=Path,
        default=Path("data"),
        help="Path prefix recorded in the manifest (default: data)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[45, 46, 47])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Create only the image metadata table; skip DINOv2 grouping and folds",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {args.manifest}. Pass --overwrite to regenerate it."
        )

    manifest = build_base_manifest(args.data_root, args.path_prefix)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} image rows to {args.manifest}")

    if args.base_only:
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.01_build_groups",
            "--manifest",
            str(args.manifest),
        ],
        check=True,
    )
    for seed in args.seeds:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.02_assign_fold_seed",
                "--manifest",
                str(args.manifest),
                "--seed",
                str(seed),
                "--n-splits",
                str(args.n_splits),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
