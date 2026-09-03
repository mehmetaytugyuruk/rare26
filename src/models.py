"""
src/models.py — Model creation (timm backbones, full fine-tune).
"""

from pathlib import Path

import torch
import torch.nn as nn
import timm

_CLASSIFIER_ROOTS = ("fc", "head")  # timm's classifier attribute names


def _has_parameter_root(key: str, roots: tuple[str, ...]) -> bool:
    """Matches an exact state-dict root without swallowing names like fc_norm."""
    return any(key == root or key.startswith(f"{root}.") for root in roots)


def load_custom_pretrained(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    """Loads a custom (e.g. GastroNet) pretrained checkpoint into `model` in-place.

    Handles both DINOv1-family layouts: a flat backbone state_dict (both
    GastroNet checkpoints here), or one nested under a "teacher" key with
    "backbone." prefixes. Anything that doesn't match the target exactly is
    rejected rather than partially loaded.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw.get("teacher", raw) if isinstance(raw, dict) else raw

    if any(k.startswith("backbone.") for k in state_dict):
        state_dict = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}

    # Drop the new classifier's keys; the backbone is what we are loading.
    state_dict = {k: v for k, v in state_dict.items()
                  if not _has_parameter_root(k, _CLASSIFIER_ROOTS)}

    model_sd = model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in model_sd and v.shape == model_sd[k].shape}

    # Counting matches isn't enough: a ViT checkpoint from a different input
    # size matches every tensor but pos_embed, which alone is the difference
    # between pretrained and randomly initialised.
    conflicts = {k: (tuple(v.shape), tuple(model_sd[k].shape))
                 for k, v in state_dict.items()
                 if k in model_sd and v.shape != model_sd[k].shape}
    if conflicts:
        raise ValueError(
            f"{checkpoint_path.name} disagrees with the model on {len(conflicts)} tensor(s): "
            f"{conflicts}. Check model.name and data.img_size against the checkpoint."
        )

    # Also an architecture mismatch: a checkpoint tensor with no slot in this
    # model (e.g. register tokens) means the weights come from a different network.
    checkpoint_only = [k for k in state_dict if k not in model_sd]
    if checkpoint_only:
        preview = checkpoint_only[:10]
        suffix = " ..." if len(checkpoint_only) > len(preview) else ""
        raise ValueError(
            f"{checkpoint_path.name} contains checkpoint-only backbone tensor(s) "
            f"not present in the selected model: {preview}{suffix}. Refusing to "
            "silently discard an architecture mismatch."
        )

    # A partial load is not a valid pretrained run.
    backbone_slots = [k for k in model_sd
                      if not _has_parameter_root(k, _CLASSIFIER_ROOTS)]
    missing_backbone = [k for k in backbone_slots if k not in matched]
    if missing_backbone:
        preview = missing_backbone[:10]
        suffix = " ..." if len(missing_backbone) > len(preview) else ""
        raise ValueError(
            f"{checkpoint_path.name} only supplies {len(matched)}/{len(backbone_slots)} "
            f"backbone tensors; missing {preview}{suffix}. Refusing to train with "
            "a partially initialized backbone."
        )

    print(f"    Loaded {len(matched)}/{len(backbone_slots)} backbone tensors")

    model.load_state_dict(matched, strict=False)
    return model


def create_model(config: dict) -> nn.Module:
    """Creates the timm backbone specified in the configuration for full fine-tuning."""
    model_config = config["model"]
    pretrained_type = model_config.get("pretrained", "imagenet")

    model_name = model_config["name"]
    num_classes = model_config.get("num_classes", 1)
    img_size = config.get("data", {}).get("img_size")

    print(f"  Model: Creating timm model '{model_name}' (pretrained={pretrained_type})")

    # ViT position embeddings need img_size at construction; ResNet/ConvNeXt don't take this kwarg.
    create_kwargs = {"pretrained": pretrained_type == "imagenet", "num_classes": num_classes}
    if "vit" in model_name and img_size is not None:
        create_kwargs["img_size"] = img_size
    model = timm.create_model(model_name, **create_kwargs)

    if pretrained_type not in ("imagenet", "none", None):
        checkpoint_path = Path(pretrained_type)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parent.parent / checkpoint_path

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Custom pretrained checkpoint not found: {checkpoint_path}. "
                "Refusing to replace the requested pretrained run with random initialization."
            )

        print(f"    Loading custom weights from {checkpoint_path}")
        load_custom_pretrained(model, checkpoint_path)

    return model
