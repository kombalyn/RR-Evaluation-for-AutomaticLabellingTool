"""Sávszűrő jelfeldolgozás (zéró fáziseltolás: SOS + filtfilt)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_sos(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 6,
) -> np.ndarray:
    """Butterworth sávszűrő, előre–hátra szűrés a fáziseltolás csökkentésére."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < max(3 * order, 16):
        return x.copy()
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 1.0 - 1e-6)
    if low >= high:
        raise ValueError(f"Érvénytelen sáv: low={low_hz}, high={high_hz}, fs={fs}")
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float64)
