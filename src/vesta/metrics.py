"""Evaluation metrics, including the revised information-noise measure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """McNemar test with continuity correction (Dietterich 1998)."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum((~a_ok) & b_ok))
    n10 = int(np.sum(a_ok & (~b_ok)))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    # chi-square survival with 1 df: erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p": float(p)}


def mean_ci(values: list[float], z: float = 1.96) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mu = float(arr.mean())
    if len(arr) < 2:
        return {"mean": mu, "std": 0.0, "ci95": 0.0}
    std = float(arr.std(ddof=1))
    ci = z * std / math.sqrt(len(arr))
    return {"mean": mu, "std": std, "ci95": float(ci)}


@dataclass
class NoiseReport:
    in_old: float
    in_new: float
    compression: float
    n_delivered: float
    n_raw: float


def information_noise(
    n_raw: int,
    delivered_tokens: list[str],
    relevant: set[str],
) -> NoiseReport:
    """Length-aware noise.

    Old (reviewer H2): n_irrelevant / n_raw — short answers look good even if mostly junk.
    New: n_irrelevant / n_delivered — precision of the delivered payload (Sun et al., 2019).
    """
    n_del = max(len(delivered_tokens), 1)
    n_irrel = sum(1 for tok in delivered_tokens if tok not in relevant)
    return NoiseReport(
        in_old=n_irrel / max(n_raw, 1),
        in_new=n_irrel / n_del,
        compression=n_del / max(n_raw, 1),
        n_delivered=n_del,
        n_raw=n_raw,
    )


def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    if returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(periods))


def cohen_kappa(a: list[str] | np.ndarray, b: list[str] | np.ndarray) -> dict[str, float]:
    """Unweighted Cohen's κ. Agreement of two nominal labellings."""
    a = np.asarray(list(a))
    b = np.asarray(list(b))
    if len(a) == 0 or len(a) != len(b):
        return {"kappa": 0.0, "po": 0.0, "pe": 1.0, "n": 0}
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    n = len(a)
    po = float((a == b).mean())
    pe = 0.0
    for lab in labels:
        pe += (a == lab).mean() * (b == lab).mean()
    if pe >= 1.0 - 1e-12:
        kappa = 1.0 if po >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {"kappa": float(kappa), "po": po, "pe": float(pe), "n": int(n)}


def fleiss_kappa(raters: list[list[str]]) -> dict[str, float]:
    """Fleiss' κ for ≥2 complete raters on the same items."""
    cols = [np.asarray(r) for r in raters]
    n = len(cols[0])
    m = len(cols)
    labels = sorted(set().union(*[set(c.tolist()) for c in cols]))
    table = np.zeros((n, len(labels)))
    idx = {lab: i for i, lab in enumerate(labels)}
    for c in cols:
        for i, lab in enumerate(c):
            table[i, idx[lab]] += 1
    p_j = table.sum(axis=0) / (n * m)
    pe = float((p_j ** 2).sum())
    p_i = ((table * table).sum(axis=1) - m) / (m * (m - 1)) if m > 1 else np.zeros(n)
    po = float(p_i.mean()) if m > 1 else 1.0
    if pe >= 1.0 - 1e-12:
        kappa = 1.0 if po >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {"kappa": float(kappa), "po": po, "pe": pe, "n": n, "raters": m}
