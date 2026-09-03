"""
src/utils.py — Config loading, seeding, checkpointing, validation guards.
"""

import json
import os
import random
import yaml
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Sets reproducibility seeds for random, numpy, and PyTorch.

    PYTHONHASHSEED is deliberately not set here. Hash randomisation is fixed
    when the interpreter starts, so assigning it at runtime would look like a
    guarantee without being one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str, base_path: str = "configs/base.yaml") -> dict:
    """Loads a config YAML and merges it onto base_path's defaults."""
    project_root = Path(__file__).resolve().parent.parent

    base_file = project_root / base_path
    if not base_file.exists():
        raise FileNotFoundError(f"Base config not found: {base_file}")

    with open(base_file, "r") as f:
        config = yaml.safe_load(f) or {}

    exp_file = Path(config_path)
    if not exp_file.is_absolute():
        exp_file = project_root / config_path

    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_file}")

    with open(exp_file, "r") as f:
        exp_config = yaml.safe_load(f) or {}

    def merge_dicts(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                merge_dicts(value, node)
            else:
                destination[key] = value
        return destination

    config = merge_dicts(exp_config, config)

    return config


def save_checkpoint(model: torch.nn.Module, path: Path, meta: dict = None):
    """Saves model weights, plus an optional metadata sidecar (same filename, .json).

    The .pth stays a plain state_dict for direct inference loading; config,
    epoch and validation metrics go in the sidecar.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    if meta is not None:
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device):
    """Loads model weights."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)


def setup_environment():
    """Sets process-wide backend flags. Runs on import of this module.

    The MPS fallback variable is read when an operator is dispatched, not when
    torch is imported, so setting it here still takes effect.
    """
    # Force CPU fallback on MPS for unsupported operators instead of crashing
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # TF32 for the fp32 ops AMP leaves alone -- safe since set_seed keeps
    # cudnn.benchmark off; TF32 trades precision, not determinism.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    """Auto-detects the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_config(device: torch.device) -> dict:
    """Returns loader/precision flags for the device.

    CUDA gets full-throughput settings. MPS and CPU share the conservative
    ones -- MPS has known issues with multiprocessing, AMP and pin_memory, so
    it is treated like CPU rather than given its own tuned path.
    """
    if device.type == "cuda":
        return {
            "num_workers": 4,
            "use_amp": True,
            "pin_memory": True,
            "channels_last": True,
        }
    return {
        "num_workers": 0,
        "use_amp": False,
        "pin_memory": False,
        "channels_last": False,
    }


setup_environment()
