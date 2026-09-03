"""Guards custom-checkpoint loading.

Every failure here is silent by default: a checkpoint that loads half the
backbone, or none of it, produces a model that trains and scores like a weak
one rather than a broken one.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.models import create_model, load_custom_pretrained


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.fc_norm = nn.LayerNorm(2)
        self.fc = nn.Linear(2, 1)


def _load(model, checkpoint):
    """Saves a checkpoint to a temp file and loads it into the model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "checkpoint.pth"
        torch.save(checkpoint, path)
        load_custom_pretrained(model, path)


class CustomPretrainedTests(unittest.TestCase):
    def _full_backbone(self, model):
        return {
            "stem.weight": torch.full_like(model.stem.weight, 3.0),
            "stem.bias": torch.full_like(model.stem.bias, 4.0),
            "fc_norm.weight": torch.full_like(model.fc_norm.weight, 5.0),
            "fc_norm.bias": torch.full_like(model.fc_norm.bias, 6.0),
        }

    def test_full_backbone_load_succeeds_and_classifier_is_ignored(self):
        model = TinyModel()
        original_fc = model.fc.weight.detach().clone()
        checkpoint = self._full_backbone(model) | {
            "fc.weight": torch.full_like(model.fc.weight, 9.0),
            "fc.bias": torch.full_like(model.fc.bias, 9.0),
        }

        _load(model, checkpoint)

        self.assertTrue(torch.equal(model.stem.weight, checkpoint["stem.weight"]))
        self.assertTrue(torch.equal(model.stem.bias, checkpoint["stem.bias"]))
        self.assertTrue(torch.equal(model.fc_norm.weight, checkpoint["fc_norm.weight"]))
        self.assertTrue(torch.equal(model.fc.weight, original_fc))

    def test_partial_backbone_load_is_rejected(self):
        model = TinyModel()
        with self.assertRaisesRegex(ValueError, "partially initialized backbone"):
            _load(model, {"stem.weight": torch.ones_like(model.stem.weight)})

    def test_shape_conflict_is_rejected(self):
        model = TinyModel()
        with self.assertRaisesRegex(ValueError, "disagrees with the model"):
            _load(model, {"stem.weight": torch.ones(3, 3),
                          "stem.bias": torch.ones_like(model.stem.bias)})

    def test_namespace_prefixed_checkpoint_is_rejected(self):
        """A DataParallel-style prefix matches nothing, which must raise rather
        than load an empty backbone."""
        model = TinyModel()
        with self.assertRaisesRegex(ValueError, "checkpoint-only backbone"):
            _load(model, {"module.stem.weight": torch.ones_like(model.stem.weight),
                          "module.stem.bias": torch.ones_like(model.stem.bias)})

    def test_checkpoint_only_backbone_tensor_is_rejected(self):
        model = TinyModel()
        checkpoint = self._full_backbone(model) | {"register_tokens": torch.ones(1, 4, 2)}

        with self.assertRaisesRegex(ValueError, "checkpoint-only backbone"):
            _load(model, checkpoint)

    def test_teacher_backbone_wrapper_still_loads(self):
        """DINOv1 checkpoints nest the backbone under a 'teacher' key, and keys
        outside that prefix are dropped."""
        model = TinyModel()
        original_fc = model.fc.weight.detach().clone()
        checkpoint = {"teacher": {
            "backbone.stem.weight": torch.full_like(model.stem.weight, 2.0),
            "backbone.stem.bias": torch.full_like(model.stem.bias, 3.0),
            "backbone.fc_norm.weight": torch.full_like(model.fc_norm.weight, 4.0),
            "backbone.fc_norm.bias": torch.full_like(model.fc_norm.bias, 5.0),
            "backbone.head.weight": torch.full_like(model.fc.weight, 9.0),
            "backbone.head.bias": torch.full_like(model.fc.bias, 9.0),
            "projection_head.last_layer": torch.ones(1),
        }}

        _load(model, checkpoint)

        self.assertTrue(torch.equal(model.stem.weight,
                                    checkpoint["teacher"]["backbone.stem.weight"]))
        self.assertTrue(torch.equal(model.fc.weight, original_fc))

    @patch("src.models.timm.create_model", return_value=TinyModel())
    def test_missing_custom_checkpoint_is_rejected(self, _create_model):
        """Falling back to random init would look like a bad training run."""
        config = {
            "model": {"name": "resnet50", "num_classes": 1, "pretrained": "missing.pth"},
            "data": {"img_size": 224},
        }

        with self.assertRaisesRegex(FileNotFoundError, "Refusing to replace"):
            create_model(config)


if __name__ == "__main__":
    unittest.main()
