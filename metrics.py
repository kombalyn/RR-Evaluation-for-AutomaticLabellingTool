"""Hibamérők és bootstrap konfidencia intervallum a MAE-re."""

from __future__ import annotations

import numpy as np


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def root_mean_square_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bootstrap_mae_ci(
    absolute_errors: np.ndarray,
    n_boot: int = 1000,
    seed: int | None = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Abszolút hibák újramintázásával a MAE eloszlása; percentilis CI.
    Vissza: (lower, upper) a (alpha/2, 1-alpha/2) percentilisek az átlagra.
    """
    e = np.asarray(absolute_errors, dtype=np.float64).ravel()
    n = e.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(e, size=n, replace=True)
        means[i] = float(np.mean(sample))
    lo = float(np.percentile(means, 100 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return lo, hi
