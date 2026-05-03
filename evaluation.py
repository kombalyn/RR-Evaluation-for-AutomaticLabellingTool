"""Teljes RR értékelési folyamat: szűrés, ablakolás, becslés vs. referencia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RREvalConfig
from filters import bandpass_sos
from metrics import bootstrap_mae_ci, mean_absolute_error, root_mean_square_error
from rr_estimation import estimate_rr_window


@dataclass(frozen=True)
class RREvaluationResult:
    """Egy futás eredménye — táblázat / ábra exportálható."""

    estimated_rpm: np.ndarray
    reference_rpm: np.ndarray
    absolute_errors: np.ndarray
    mae: float
    rmse: float
    coverage: float
    n_windows_valid: int
    n_windows_total: int
    mae_ci_lower: float
    mae_ci_upper: float


def evaluate_signal_vs_reference(
    signal: np.ndarray,
    reference: np.ndarray,
    cfg: RREvalConfig | None = None,
) -> RREvaluationResult:
    """
    signal, reference: azonos hosszú 1D sorozatok (mintavételezés: cfg.fps).
    Referencia RR egy ablakon: a referencia jel átlaga az adott időintervallumon
    (a Colab verzióval megegyező definíció).
    """
    cfg = cfg or RREvalConfig()
    sig = np.asarray(signal, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if sig.shape != ref.shape:
        raise ValueError("signal és reference mérete nem egyezik.")

    filtered = bandpass_sos(
        sig,
        fs=cfg.fps,
        low_hz=cfg.bandpass_low_hz,
        high_hz=cfg.bandpass_high_hz,
        order=cfg.bandpass_order,
    )

    window = int(round(cfg.window_seconds * cfg.fps))
    if window < 32:
        raise ValueError("Az ablak túl rövid; növeld a window_seconds értéket.")

    est_list: list[float] = []
    ref_list: list[float] = []

    n_total = 0
    for start in range(0, len(filtered) - window, window):
        seg = filtered[start : start + window]
        ref_seg = ref[start : start + window]
        if seg.size != window:
            continue
        n_total += 1
        r_est = estimate_rr_window(seg, cfg)
        r_ref = float(np.mean(ref_seg))
        est_list.append(r_est)
        ref_list.append(r_ref)

    est = np.asarray(est_list, dtype=np.float64)
    refv = np.asarray(ref_list, dtype=np.float64)

    valid = np.isfinite(est) & np.isfinite(refv)
    est_v = est[valid]
    ref_v = refv[valid]
    abs_err = np.abs(est_v - ref_v)

    mae = mean_absolute_error(est_v, ref_v) if est_v.size else float("nan")
    rmse = root_mean_square_error(est_v, ref_v) if est_v.size else float("nan")
    coverage = float(est_v.size / max(n_total, 1))

    lo, hi = bootstrap_mae_ci(abs_err, n_boot=cfg.bootstrap_n, seed=cfg.bootstrap_seed)

    return RREvaluationResult(
        estimated_rpm=est_v,
        reference_rpm=ref_v,
        absolute_errors=abs_err,
        mae=mae,
        rmse=rmse,
        coverage=coverage,
        n_windows_valid=int(est_v.size),
        n_windows_total=int(n_total),
        mae_ci_lower=lo,
        mae_ci_upper=hi,
    )
