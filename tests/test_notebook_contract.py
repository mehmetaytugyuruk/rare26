"""Guards the Colab notebook's contracts as text.

Nothing here executes the notebook. These are the invariants that no local run
can catch -- a token reaching a command line, a checkout drifting off the
pinned commit, a guard moving after the Drive mount -- so they are asserted
against the notebook source instead.
"""

import json
import unittest
from pathlib import Path


class ColabNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).resolve().parents[1] / "notebook/colab_training.ipynb"
        cls.notebook = json.loads(notebook_path.read_text())
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_unconfigured_plan_stops_before_drive_mount(self):
        sha_guard = self.source.index("Set EXPECTED_COMMIT_SHA to the full 40-character")
        config_guard = self.source.index("Set CONFIGS_TO_RUN explicitly")
        drive_mount = self.source.index('drive.mount("/content/drive")')

        self.assertLess(sha_guard, drive_mount)
        self.assertLess(config_guard, drive_mount)

    def test_checkout_is_detached_and_verified_at_full_sha(self):
        self.assertIn('["git", "checkout", "--detach", EXPECTED_COMMIT_SHA]', self.source)
        self.assertIn('["git", "rev-parse", "HEAD"]', self.source)
        self.assertIn("actual_sha == EXPECTED_COMMIT_SHA", self.source)

    def test_configs_are_repo_relative_and_tracked_at_the_pinned_sha(self):
        self.assertIn("os.path.isabs(cfg_path)", self.source)
        self.assertIn("Config path escapes the audited checkout", self.source)
        self.assertIn(
            '["git", "ls-files", "--error-unmatch", "--", normalized_cfg]',
            self.source,
        )
        self.assertIn(
            '["git", "status", "--porcelain", "--untracked-files=all"]',
            self.source,
        )
        self.assertNotIn("--untracked-files=no", self.source)

    def test_public_clone_does_not_require_credentials(self):
        self.assertNotIn("GH_TOKEN", self.source)
        self.assertNotIn("google.colab import userdata", self.source)
        self.assertIn('["git", "clone", clean_url, LOCAL_REPO_DIR]', self.source)

    def test_private_manifest_is_generated_before_preflight(self):
        prepare = self.source.index('"scripts.00_prepare_manifest"')
        preflight = self.source.index(
            "preflight_experiment(cfg_path, limit_seeds=LIMIT_SEEDS)"
        )
        self.assertLess(prepare, preflight)

    def test_preflight_uses_the_production_entry_point(self):
        self.assertIn("from train import preflight_experiment", self.source)
        self.assertIn(
            "preflight_experiment(cfg_path, limit_seeds=LIMIT_SEEDS)",
            self.source,
        )
        self.assertNotIn("from src.trainer import Trainer", self.source)

    def test_each_config_checks_its_own_batch_size_without_dead_fallback(self):
        self.assertIn("batch_fit_by_config", self.source)
        self.assertIn("test_batch_size(cfg_path, design_batch_size)", self.source)
        self.assertNotIn("CONFIGS_TO_RUN[0]", self.source)
        self.assertNotIn("SAFE_BATCH_SIZE", self.source)
        self.assertNotIn("CANDIDATE_BATCH_SIZES", self.source)

    def test_batch_probe_reduces_the_loss_itself(self):
        """The probe calls create_loss() directly rather than going through the
        Trainer, so it must reduce any per-row loss before backward().
        """
        probe = self.source[self.source.index("def test_batch_size"):]
        probe = probe[:probe.index("batch_fit_by_config")]
        self.assertIn("loss = loss.mean()", probe)
        self.assertLess(probe.index("loss = loss.mean()"),
                        probe.index("scaler.scale(loss).backward()"))

    def test_none_limit_is_not_rendered_as_a_cli_value(self):
        self.assertNotIn("--limit-seeds {LIMIT_SEEDS}", self.source)
        self.assertIn("if LIMIT_SEEDS is not None", self.source)
        self.assertIn('command.extend(["--limit-seeds", str(LIMIT_SEEDS)])', self.source)
        self.assertIn("subprocess.run(command, check=True)", self.source)

    def test_no_results_file_is_read_or_synced(self):
        """The notebook reads results from stdout and checkpoint sidecars."""
        self.assertNotIn("experiments.csv", self.source)
        self.assertNotIn("record_type", self.source)


if __name__ == "__main__":
    unittest.main()
