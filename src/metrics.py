"""
src/metrics.py — Evaluation metrics.

FPR@90TPR (low = good) is a deterministic local diagnostic. The official
PPV@90Recall score is computed separately with the challenge bootstrap.
"""

from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray,
               target_tpr: float = 0.90) -> float:
    """Returns FPR at target_tpr by interpolating the ROC curve. LOW = GOOD."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # np.interp requires x to be increasing; tpr from roc_curve is increasing
    return float(np.interp(target_tpr, tpr, fpr))


def fpr_to_lb_ppv(fpr: float, prevalence: float = 1 / 101) -> float:
    """Converts FPR@90TPR to expected leaderboard PPV at ~1% prevalence."""
    if fpr <= 0:
        return 1.0
    return 0.9 * prevalence / (0.9 * prevalence + fpr * (1 - prevalence))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Full diagnostic report: FPR@80/85/90/95, composite, AUROC, AUPRC, estimated LB PPV."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # One ROC curve for all four points; fpr_at_tpr() does the same interpolation.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr_80, fpr_85, fpr_90, fpr_95 = (
        float(np.interp(t, tpr, fpr)) for t in (0.80, 0.85, 0.90, 0.95)
    )

    return {
        "fpr@80": fpr_80,
        "fpr@85": fpr_85,
        "fpr@90": fpr_90,
        "fpr@95": fpr_95,
        "fpr_composite": float(np.mean([fpr_80, fpr_85, fpr_90, fpr_95])),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "lb_ppv_estimate": fpr_to_lb_ppv(fpr_90),
    }


# ---------------------------------------------------------------------------
# Official score wrapper
# ---------------------------------------------------------------------------

def official_score(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Calls the UNTOUCHED official evaluation_Grand-Challenge.py. Slow (1000 bootstrap iterations) -- never call inside the training loop."""
    # importlib because the filename has hyphens; loads the script unmodified.
    import importlib.util
    project_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "evaluation_grand_challenge",
        project_root / "evaluation_Grand-Challenge.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    return mod.bootstrap_metrics(y_true.tolist(), y_score.tolist())


# ---------------------------------------------------------------------------
# Center-normalized official score
# ---------------------------------------------------------------------------

def center_normalized_official_score(y_true: np.ndarray, y_score: np.ndarray,
                                      center: np.ndarray) -> dict:
    """Replaces each score with its rank (0..1) within its own centre, removing
    any between-centre offset while keeping each centre's internal ranking. The
    gap to official_score() is how much of the pooled score reads centre
    identity rather than the lesion.

    Report alongside official_score(), never instead of it -- lower raw-pooled
    but higher centre-normalized means less reliance on centre, which is the goal.
    """
    y_score = np.asarray(y_score, dtype=float)
    center = np.asarray(center)

    normalized = np.empty_like(y_score)
    for c in np.unique(center):
        mask = center == c
        normalized[mask] = rankdata(y_score[mask]) / mask.sum()

    return official_score(y_true, normalized)
