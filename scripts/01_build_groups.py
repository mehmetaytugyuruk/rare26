"""
scripts/01_build_groups.py — DINOv2 Embedding Extraction + Near-Duplicate Grouping

Extracts DINOv2 ViT-B/14 features for every image, then uses their cosine
similarity to detect near-duplicate frames (same lesion/scene shot multiple
times) and assigns each image a group_id -- used by StratifiedGroupKFold so
near-duplicates never land on both sides of a train/val split. Patient IDs
aren't available (filenames are anonymized hashes), so this is the proxy.

One-time step for this dataset: data/data_manifest.csv already has
group_id populated from this exact process -- only re-run if the data
changes.

Usage:
    python -m scripts.01_build_groups
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader

from src.utils import get_device, get_device_config, set_seed
from src.dataset import ImageDataset, get_transforms

MODEL_NAME = "vit_base_patch14_dinov2"
IMG_SIZE = 518


def extract_embeddings(df: pd.DataFrame, device, device_config) -> np.ndarray:
    """Extracts L2-normalized DINOv2 ViT-B/14 features for every image in df."""
    print(f"  Loading {MODEL_NAME} (LVD-142M self-supervised weights)...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=IMG_SIZE)
    model = model.to(device)
    model.eval()
    if device_config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    # Val transform = resize + normalize only, no augmentation.
    transform = get_transforms({"data": {"img_size": IMG_SIZE}}, "val")
    dataset = ImageDataset(df, transform=transform)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        num_workers=device_config["num_workers"], pin_memory=device_config["pin_memory"]
    )

    features_list = []
    print(f"  Extracting features for {len(df)} images...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = batch["image"].to(device)
            if device_config["channels_last"]:
                inputs = inputs.to(memory_format=torch.channels_last)
            feats = model(inputs)
            features_list.append(feats.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {min((batch_idx + 1) * loader.batch_size, len(df))}/{len(df)}")

    features = np.concatenate(features_list, axis=0)
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    return features / norms


def build_groups(df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    """Mutual-kNN + union-find near-duplicate grouping, within-center only,
    on thresholds calibrated for DINOv2's cosine similarity scale."""
    N = len(df)
    print("Computing cosine similarity matrix...")
    sim_matrix = embeddings @ embeddings.T   # cosine sim (L2-normalised in extraction)
    np.fill_diagonal(sim_matrix, -1.0)        # diagonal is not a neighbour

    # Within-center constraint: different-center similarities forced to -1.0
    centers = np.array(df["center"].tolist())
    same_center = (centers[:, None] == centers[None, :])
    sim_matrix = np.where(same_center, sim_matrix, -1.0)

    final_groups = np.arange(N)  # everyone starts as a singleton

    # Helper: mutual-kNN + union-find, recursively splitting oversized clusters
    def group_subset(indices, k, base_thr, max_size):
        sub_sim = sim_matrix[np.ix_(indices, indices)]

        def do_mutual_knn(thr):
            knn = np.argsort(-sub_sim, axis=1)[:, :k]
            parent = list(range(len(indices)))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            for i in range(len(indices)):
                for j_idx in range(k):
                    j = knn[i, j_idx]
                    if i in knn[j] and sub_sim[i, j] >= thr:
                        union(i, j)
            return [find(i) for i in range(len(indices))]

        group_raw = do_mutual_knn(base_thr)
        cnt = Counter(group_raw)
        subset_group_mapping = {}
        next_gid = 0

        for root, count in cnt.items():
            members = [i for i, r in enumerate(group_raw) if r == root]
            if count <= max_size:
                for m in members:
                    subset_group_mapping[m] = next_gid
                next_gid += 1
            else:
                # Group too large -- recurse with a stricter threshold
                new_thr = base_thr + 0.02
                if new_thr >= 0.99:
                    # Failsafe: threshold maxed out, assign as singletons
                    for m in members:
                        subset_group_mapping[m] = next_gid
                        next_gid += 1
                else:
                    global_members = [indices[m] for m in members]
                    sub_mapping = group_subset(global_members, k, new_thr, max_size)
                    for local_m, sub_gid in zip(members, sub_mapping):
                        subset_group_mapping[local_m] = next_gid + sub_gid
                    next_gid += max(sub_mapping) + 1 if sub_mapping else 0

        return [subset_group_mapping[i] for i in range(len(indices))]

    print("Processing NEO groups...")
    neo_idx = df[df["label"] == 1].index.tolist()
    # DINOv2 calibrated: thr=0.94 gives 7 multi-groups over 18 images
    neo_grouping = group_subset(neo_idx, k=3, base_thr=0.94, max_size=10)

    print("Processing NDBE groups...")
    ndbe_idx = df[df["label"] == 0].index.tolist()
    # DINOv2 calibrated: thr=0.95 gives 142 multi-groups over 378 images
    ndbe_grouping = group_subset(ndbe_idx, k=2, base_thr=0.95, max_size=20)

    gid_offset = 0
    if len(neo_grouping) > 0:
        for local_i, group_id in enumerate(neo_grouping):
            final_groups[neo_idx[local_i]] = group_id + gid_offset
        gid_offset += max(neo_grouping) + 1
    if len(ndbe_grouping) > 0:
        for local_i, group_id in enumerate(ndbe_grouping):
            final_groups[ndbe_idx[local_i]] = group_id + gid_offset

    return final_groups


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild group_id from DINOv2 near-duplicate clustering")
    p.add_argument("--manifest", type=Path, default=Path("data/data_manifest.csv"))
    p.add_argument("--force", action="store_true",
                   help="Rebuild group_id even though pinned fold columns depend on it")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("NEAR-DUPLICATE GROUPING (DINOv2 embeddings)")
    print("=" * 60)

    # Pinned fold columns were built from the current group_id. Rebuilding it
    # can shift borderline clustering decisions -- the thresholds sit at 0.94
    # and 0.95, and embeddings are not bit-identical across devices -- which
    # would leave those columns describing a group structure that no longer
    # exists, without anything failing.
    existing = pd.read_csv(args.manifest, nrows=0).columns
    pinned = [c for c in existing if c.startswith("fold_")]
    if "group_id" in existing and pinned and not args.force:
        print(f"group_id already exists and {len(pinned)} pinned fold column(s) depend on it: "
              f"{', '.join(pinned)}.\nRebuilding it would desync them from the grouping they "
              f"were generated against, and any checkpoints trained under them.\n"
              f"Pass --force if you are deliberately invalidating those columns.")
        sys.exit(1)

    set_seed(42)
    device = get_device()
    device_config = get_device_config(device)
    print(f"  Target device: {device}")

    df = pd.read_csv(args.manifest)
    embeddings = extract_embeddings(df, device, device_config)
    print(f"  Embeddings: {embeddings.shape}")

    df["group_id"] = build_groups(df, embeddings)

    print("\n=== FINAL PROTOCOL VERIFICATION ===")
    cnt = Counter(df["group_id"])
    multi = {k: v for k, v in cnt.items() if v > 1}
    sings = sum(1 for v in cnt.values() if v == 1)
    print(f"Total images:           {len(df)}")
    print(f"Multi-image groups:     {len(multi)}")
    print(f"Images in multi-groups: {sum(multi.values())}")
    print(f"Singletons (unique ID): {sings}")
    print(f"Total unique group_ids: {len(multi) + sings}")

    df.to_csv(args.manifest, index=False)
    print(f"\nUpdated {args.manifest} successfully.")

    print("\n" + "=" * 60)
    print("NEAR-DUPLICATE GROUPING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
