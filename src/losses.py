"""
src/losses.py — Loss registry. Weighted BCE only.
"""

import torch
import torch.nn as nn


def create_loss(config: dict) -> nn.Module:
    """Builds the configured loss.

    pos_weight multiplies the positive term. All method configs use 18.6,
    the training set's negative-to-positive ratio (2937/158). This is not
    the balancing value under Balanced-MixUp, which raises the effective
    positive rate to 12.6% where 6.94 would balance the loss; 18.6 gives
    positives 2.68x the loss mass of negatives.

    Reduction is always ``mean``: the loss returns one scalar per batch, which
    is what both the ordinary and the Balanced-MixUp training loops backward on.
    """
    loss_config = config["loss"]
    name = loss_config["name"].lower()
    if name != "bce":
        raise ValueError(f"Unknown loss type: {name}. Only 'bce' is supported.")

    pos_weight_val = loss_config.get("pos_weight", 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    print(f"  Loss: Creating '{name}' loss (pos_weight={pos_weight_val})")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
