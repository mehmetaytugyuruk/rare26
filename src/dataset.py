"""
src/dataset.py — PyTorch Datasets and Samplers

Includes:
1. ImageDataset: Loads images and applies augmentations.
2. class_balanced_sample_weights: row weights that make a WeightedRandomSampler
   draw the two labels equally often, for Balanced-MixUp's second branch.
3. EvenSpreadBatchSampler: every image used once per epoch, center x label
   groups interleaved evenly across batches instead of left to random shuffling.
"""

import io
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image, ImageEnhance
import torchvision.transforms as T


class ImageDataset(Dataset):
    """Loads raw endoscopic images from their manifest path and augments them."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Only image and label are exposed; centre and group_id stay in the manifest.
        row = self.df.iloc[idx]
        img = Image.open(Path(row["path"])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(int(row["label"]), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Custom Augmentation helper
# ---------------------------------------------------------------------------

class RandomSharpness:
    """Two-sided blur/sharpen jitter (factor<1 softens, >1 sharpens) so training isn't offset from clean inference images."""
    def __init__(self, factor_range: tuple[float, float], probability: float):
        self.factor_range = factor_range
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        return ImageEnhance.Sharpness(img).enhance(random.uniform(*self.factor_range))


class RandomJpegCompression:
    """Re-encodes at a random JPEG quality, covering scopes that compress."""
    def __init__(self, quality_range: tuple[int, int], probability: float):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# Train-only augmentations, always applied in this order; no config selects
# among them. Resize/crop is handled separately in get_transforms() below.
_TRAIN_AUGMENTATIONS = [
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation((0, 360), interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03)], p=0.4),
    RandomSharpness(factor_range=(0.6, 1.6), probability=0.2),
    RandomJpegCompression(quality_range=(50, 95), probability=0.2),
]


def get_transforms(config: dict, split: str) -> T.Compose:
    """Builds the split's transform pipeline.

    Train augmentations span the axes endoscopy devices actually differ on and
    are kept moderate, centred on the identity transform.
    """
    img_size = config["data"]["img_size"]
    is_train = split == "train"

    transforms_pil = []

    if is_train:
        # Use a moderate crop in 40% of samples and a plain resize in 60%.
        transforms_pil.append(
            T.RandomChoice(
                [
                    T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.90, 1.20),
                                        interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                    T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                ],
                p=[0.4, 0.6],
            )
        )
        transforms_pil.extend(_TRAIN_AUGMENTATIONS)
    else:
        transforms_pil.append(T.Resize((img_size, img_size)))

    transforms_pil.append(T.ToTensor())
    transforms_pil.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms_pil)


def class_balanced_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-row weights (1/n_class) that make every class equally likely to be drawn.

    Feeds Balanced-MixUp's second loader (src/trainer.py): drawn with
    replacement, ~half of what it yields is positive against natural
    sampling's 5.1%. Sampling uses replacement and each draw receives a new
    partner and lambda.
    """
    if "label" not in df.columns:
        raise ValueError("class_balanced_sample_weights requires a 'label' column.")
    labels = df["label"].to_numpy()
    weights = np.zeros(len(df), dtype=np.float64)
    for value in np.unique(labels):
        mask = labels == value
        count = int(mask.sum())
        if count == 0:
            continue
        weights[mask] = 1.0 / count
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("class_balanced_sample_weights produced a degenerate weighting.")
    return weights


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class EvenSpreadBatchSampler(Sampler):
    """One epoch, every row exactly once.

    Quota per group, NEO groups round-robined across their near-duplicate
    clusters; the largest group fills the rest. Batches are full except for an
    optional final short batch, allowing the paired Balanced-MixUp loaders to
    stay aligned. Invalid layouts raise instead of truncating a pair.
    """

    def __init__(self, df: pd.DataFrame, batch_size: int, group_column: str = "group_id"):
        self.batch_size = batch_size
        self.n = len(df)
        self.n_batches = math.ceil(self.n / batch_size)

        keys = (df["center"].astype(str) + "_" + df["label"].astype(str)).values
        groups = {k: np.flatnonzero(keys == k) for k in sorted(set(keys))}

        # Largest group fills the remaining capacity, so only the smaller groups
        # need quota bookkeeping.
        self.fill_key = max(groups, key=lambda k: len(groups[k]))
        self.quota_groups = {k: v for k, v in groups.items() if k != self.fill_key}
        self.fill_indices = groups[self.fill_key]

        self.group_ids = (
            df[group_column].to_numpy() if group_column in df.columns
            else np.arange(self.n)
        )
        # Near-duplicate-aware interleaving only applies to the rare (label=1) groups.
        self.dup_aware_keys = {k for k in self.quota_groups if k.endswith("_1")}

    def _dup_aware_shuffle(self, indices: np.ndarray) -> np.ndarray:
        """Round-robins near-duplicate clusters so same-cluster members don't land at consecutive quota positions."""
        clusters = {}
        for i in indices:
            clusters.setdefault(self.group_ids[i], []).append(int(i))
        for members in clusters.values():
            np.random.shuffle(members)

        keys = list(clusters.keys())
        np.random.shuffle(keys)
        ordered = []
        while keys:
            next_keys = []
            for k in keys:
                members = clusters[k]
                if members:
                    ordered.append(members.pop())
                if members:
                    next_keys.append(k)
            np.random.shuffle(next_keys)
            keys = next_keys
        return np.asarray(ordered, dtype=int)

    def __iter__(self):
        batches = [[] for _ in range(self.n_batches)]

        for key, idx in self.quota_groups.items():
            ordered = self._dup_aware_shuffle(idx) if key in self.dup_aware_keys else np.random.permutation(idx)
            slot_order = np.random.permutation(self.n_batches)
            for position, sample_idx in enumerate(ordered):
                batches[slot_order[position % self.n_batches]].append(int(sample_idx))

        fill_pool = np.random.permutation(self.fill_indices)
        ptr = 0
        for batch in batches:
            capacity = self.batch_size - len(batch)
            batch.extend(int(x) for x in fill_pool[ptr:ptr + capacity])
            ptr += capacity

        used = sum(len(b) for b in batches)
        if used != self.n:
            raise RuntimeError(f"EvenSpreadBatchSampler used {used} samples, expected {self.n}")

        # Verify the batch-size contract from the class docstring, then reorder
        # -- membership and every-row-once coverage are already fixed above.
        sizes = [len(batch) for batch in batches]
        expected_short = self.n % self.batch_size
        full_batch_ids = [i for i, size in enumerate(sizes) if size == self.batch_size]
        short_batch_ids = [i for i, size in enumerate(sizes) if size < self.batch_size]
        if (max(sizes) > self.batch_size
                or len(short_batch_ids) != int(expected_short != 0)
                or any(sizes[i] != expected_short for i in short_batch_ids)):
            raise RuntimeError(
                "EvenSpreadBatchSampler produced an invalid batch-size layout: "
                f"expected {self.n_batches - int(expected_short != 0)} full batch(es) "
                f"and short_sizes={[expected_short] if expected_short else []}, "
                f"got sizes={sizes}"
            )

        batch_order = [int(x) for x in np.random.permutation(full_batch_ids)]
        batch_order.extend(short_batch_ids)
        for b in batch_order:
            batch = batches[b]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
