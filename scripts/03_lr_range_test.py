"""
scripts/03_lr_range_test.py — LR range test (Smith 2017) for a given config.

Sweeps the learning rate exponentially from --min-lr to --max-lr over one
short training pass, recording the loss at every step, instead of picking a
rate by inheritance or by reasoning about weight norms.

The loss-vs-log(lr) curve goes flat -> descending (usable band) -> minimum ->
sharp rise (divergence). The script prints the steepest-descent point and the
divergence point; pick a rate roughly an order of magnitude below divergence.

Needed here because GastroNet's ResNet-50 weights are ~17x smaller than
ImageNet's in mean absolute value, so an ImageNet-tuned rate would move them
too far and risk destroying the pretrained features.

Usage:
    python -m scripts.03_lr_range_test --config configs/resnet50_fold.yaml
    python -m scripts.03_lr_range_test --config configs/vits_fold.yaml --num-steps 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import ImageDataset, EvenSpreadBatchSampler, get_transforms
from src.losses import create_loss
from src.models import create_model
from src.splits import get_splits
from src.utils import load_config, get_device, get_device_config, set_seed

OUT_PATH = Path("results/lr_range_test.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Exponential LR range test")
    p.add_argument("--config", required=True)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--max-lr", type=float, default=1e-1)
    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--smooth", type=float, default=0.05,
                   help="EMA factor for the loss curve; raw per-batch loss is too noisy to read")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)

    device = get_device()
    device_config = get_device_config(device)

    df = pd.read_csv("data/data_manifest.csv")
    splits = get_splits(df, config["data"].get("split_mode", "group_cv"),
                        config["data"]["n_folds"], args.seed)
    train_idx, _ = splits[args.fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)

    dataset = ImageDataset(train_df, transform=get_transforms(config, "train"))
    sampler = EvenSpreadBatchSampler(train_df, batch_size=config["training"]["batch_size"])
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=device_config["num_workers"],
                        pin_memory=device_config["pin_memory"])

    model = create_model(config).to(device)
    if device_config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    loss_fn = create_loss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr,
                                  weight_decay=float(config["training"].get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=device_config["use_amp"])

    gamma = (args.max_lr / args.min_lr) ** (1 / args.num_steps)
    print(f"LR range test: {args.min_lr:.1e} -> {args.max_lr:.1e} over {args.num_steps} steps "
          f"(x{gamma:.4f} per step)")
    print(f"  config={args.config} | fold={args.fold} seed={args.seed} | "
          f"batch_size={config['training']['batch_size']}")

    records = []
    smoothed = None
    best_smoothed = float("inf")
    step = 0
    model.train()

    while step < args.num_steps:
        for batch in loader:
            if step >= args.num_steps:
                break
            lr = args.min_lr * (gamma ** step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            inputs = batch["image"].to(device)
            if device_config["channels_last"]:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device_config["use_amp"]):
                loss = loss_fn(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            raw = loss.item()
            smoothed = raw if smoothed is None else args.smooth * raw + (1 - args.smooth) * smoothed
            best_smoothed = min(best_smoothed, smoothed)
            records.append({"step": step, "lr": lr, "loss": raw, "smoothed": smoothed})

            if step % 25 == 0:
                print(f"  step {step:4d} | lr={lr:.2e} | loss={raw:.4f} | smoothed={smoothed:.4f}")

            # Diverged well past the minimum -- no information left in continuing.
            if smoothed > 4 * best_smoothed and step > 20:
                print(f"\n  Diverged at step {step} (lr={lr:.2e}); stopping early.")
                step = args.num_steps
                break
            step += 1

    res = pd.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, index=False)

    # Steepest descent = most negative gradient of smoothed loss w.r.t. log(lr).
    valid = res[res["lr"] > args.min_lr * 10].copy()
    if len(valid) > 10:
        grad = np.gradient(valid["smoothed"].to_numpy(), np.log10(valid["lr"].to_numpy()))
        steepest_lr = float(valid["lr"].to_numpy()[int(np.argmin(grad))])
        min_loss_lr = float(valid.loc[valid["smoothed"].idxmin(), "lr"])
        print("\n" + "=" * 60)
        print(f"  Steepest descent at lr = {steepest_lr:.2e}")
        print(f"  Minimum smoothed loss at lr = {min_loss_lr:.2e}")
        print(f"  Conventional pick: between {steepest_lr:.1e} and {min_loss_lr/10:.1e}")
        print("=" * 60)
    print(f"\nFull curve written to {OUT_PATH} — plot loss vs lr (log x) to read it yourself.")


if __name__ == "__main__":
    main()
