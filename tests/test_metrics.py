"""Guards the scoring path itself.

If our deterministic FPR path drifts from the official bootstrap, every local
number in the project silently measures something other than the leaderboard.
"""

import unittest

import numpy as np

from src.metrics import fpr_at_tpr, fpr_to_lb_ppv, official_score, report


def _synthetic_at_fpr(target_fpr, n_neg=5000, n_pos=50, seed=42):
    """Builds scores whose FPR at 90% TPR is target_fpr by construction.

    The 100:1 negative:positive ratio matches what the official bootstrap
    resamples to. It also has to stay that size: at 20:1 the two agree only to
    ~3.5%, close enough to the 5% bar to make this flaky.
    """
    rng = np.random.RandomState(seed)
    y_true = np.array([0] * n_neg + [1] * n_pos)
    y_pos = rng.uniform(0.5, 1.0, n_pos)
    threshold = np.percentile(y_pos, 10)
    n_fps = int(target_fpr * n_neg)
    y_neg = np.concatenate([
        rng.uniform(0.0, threshold - 0.001, n_neg - n_fps),
        rng.uniform(threshold + 0.001, 1.0, n_fps),
    ])
    return y_true, np.concatenate([y_neg, y_pos])


class ScaleCalibrationTests(unittest.TestCase):
    def test_deterministic_ppv_tracks_the_official_bootstrap(self):
        """fpr_to_lb_ppv() must agree with evaluation_Grand-Challenge.py across
        the operating region the leaderboard occupies. A failure means local PPV
        is on a different scale from the ranking metric."""
        for target_fpr in (0.15, 0.25, 0.40):
            with self.subTest(target_fpr=target_fpr):
                y_true, y_score = _synthetic_at_fpr(target_fpr)
                deterministic = fpr_to_lb_ppv(fpr_at_tpr(y_true, y_score, 0.90))
                official = official_score(y_true, y_score)["PPV@90RECALL"]
                self.assertLess(
                    abs(deterministic - official) / official, 0.05,
                    f"deterministic PPV {deterministic:.4f} and official "
                    f"{official:.4f} diverge by more than 5%",
                )


class FprAtTprTests(unittest.TestCase):
    def test_separable_scores_give_low_fpr(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        self.assertLess(fpr_at_tpr(y, s, 0.90), 0.5)

    def test_random_scores_give_high_fpr(self):
        rng = np.random.RandomState(42)
        self.assertGreater(fpr_at_tpr(rng.randint(0, 2, 1000), rng.rand(1000), 0.90), 0.5)

    def test_report_matches_the_single_point_helper(self):
        """report() interpolates one shared ROC curve; fpr_at_tpr() builds its
        own. They must not drift apart."""
        rng = np.random.RandomState(0)
        y = np.array([0] * 900 + [1] * 100)
        s = np.concatenate([rng.normal(0, 1, 900), rng.normal(1.5, 1, 100)])
        r = report(y, s)
        for tpr, key in ((0.80, "fpr@80"), (0.85, "fpr@85"),
                         (0.90, "fpr@90"), (0.95, "fpr@95")):
            self.assertEqual(r[key], fpr_at_tpr(y, s, tpr))


if __name__ == "__main__":
    unittest.main()
