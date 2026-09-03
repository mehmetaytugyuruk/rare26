import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from PIL import Image


prepare_manifest = import_module("scripts.00_prepare_manifest")
BASE_COLUMNS = prepare_manifest.BASE_COLUMNS
build_base_manifest = prepare_manifest.build_base_manifest


class PrepareManifestTests(unittest.TestCase):
    def test_builds_stable_rows_from_the_documented_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "data"
            specifications = [
                ("center_2", "neo", "b.png", (8, 5)),
                ("center_1", "ndbe", "z.png", (7, 4)),
                ("center_1", "ndbe", "a.png", (6, 3)),
                ("center_1", "neo", "c.png", (5, 2)),
                ("center_2", "ndbe", "d.png", (4, 2)),
            ]
            for center, class_name, filename, size in specifications:
                target = data_root / center / class_name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", size).save(target)

            manifest = build_base_manifest(data_root)

            self.assertEqual(list(manifest.columns), BASE_COLUMNS)
            self.assertEqual(manifest["path"].tolist(), sorted(manifest["path"]))
            self.assertEqual(
                manifest["path"].tolist(),
                [
                    "data/center_1/ndbe/a.png",
                    "data/center_1/ndbe/z.png",
                    "data/center_1/neo/c.png",
                    "data/center_2/ndbe/d.png",
                    "data/center_2/neo/b.png",
                ],
            )
            self.assertEqual(manifest["label"].tolist(), [0, 0, 1, 0, 1])
            self.assertEqual(manifest["format"].tolist(), ["PNG"] * 5)
            self.assertEqual(manifest.loc[0, ["width", "height"]].tolist(), [6, 3])

    def test_rejects_an_incomplete_dataset_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Expected dataset directory"):
                build_base_manifest(Path(tmp_dir) / "data")


if __name__ == "__main__":
    unittest.main()
