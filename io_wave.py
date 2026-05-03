"""Légzési hullám / RR táblák betöltése (CSV vagy szóköz / tab)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wave_matrix(
    path: str | Path,
    delimiter: str = ",",
    signal_col: int = 0,
    reference_col: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vissza: (signal_1d, reference_1d) — azonos hossz.
    reference_col=-1: utolsó oszlop a referencia (RR vagy referencia eszköz jele).
    """
    p = Path(path)
    data = np.loadtxt(p, delimiter=delimiter, dtype=np.float64)
    if data.ndim == 1:
        raise ValueError(f"Legalább 2 oszlop kell (jel + referencia): {p}")
    n_cols = data.shape[1]
    ref_idx = n_cols - 1 if reference_col < 0 else reference_col
    if not (0 <= signal_col < n_cols and 0 <= ref_idx < n_cols):
        raise ValueError(f"Érvénytelen oszlop index: signal={signal_col}, ref={ref_idx}, n_cols={n_cols}")
    signal = np.asarray(data[:, signal_col], dtype=np.float64).ravel()
    reference = np.asarray(data[:, ref_idx], dtype=np.float64).ravel()
    if signal.shape[0] != reference.shape[0]:
        raise ValueError("A jel és a referencia hossza nem egyezik.")
    return signal, reference
