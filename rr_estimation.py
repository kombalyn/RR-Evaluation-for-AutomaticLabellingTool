"""RR (légzésszám, BPM) becslés ablakon belül: csúcs + FFT tartalék."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from config import RREvalConfig


def robust_normalize(sig: np.ndarray) -> np.ndarray:
    """MAD-alapú skálázás (robosztus outlierekhez)."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    x = x - np.median(x)
    mad = np.median(np.abs(x))
    if mad > 0:
        x = x / (1.4826 * mad)
    else:
        x = x / (np.std(x) + 1e-9)
    return x


def rr_from_peaks(sig: np.ndarray, cfg: RREvalConfig) -> float:
    """Inter-csúcs időköz mediánja → BPM."""
    fs = cfg.fps
    min_distance = max(1, int(fs * cfg.peak_min_distance_sec))
    height = cfg.peak_height_std_factor * float(np.std(sig))
    peaks, _ = find_peaks(sig, distance=min_distance, height=height)
    if peaks.size < cfg.peak_min_count:
        return float("nan")
    diffs = np.diff(peaks.astype(np.float64))
    median_period_s = float(np.median(diffs)) / fs
    if median_period_s <= 0:
        return float("nan")
    return 60.0 / median_period_s


def rr_from_fft(sig: np.ndarray, cfg: RREvalConfig) -> float:
    """Domináns frekvencia a megadott Hz sávban → BPM."""
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / cfg.fps)
    spec = np.abs(np.fft.rfft(sig))
    mask = (freqs >= cfg.fft_freq_low_hz) & (freqs <= cfg.fft_freq_high_hz)
    if not np.any(mask):
        return float("nan")
    k = int(np.argmax(spec[mask]))
    peak_hz = float(freqs[mask][k])
    return peak_hz * 60.0


def estimate_rr_window(sig: np.ndarray, cfg: RREvalConfig) -> float:
    """Normalizálás → csúcs-alapú RR; ha érvénytelen, FFT tartalék."""
    z = robust_normalize(sig)
    rr_p = rr_from_peaks(z, cfg)
    if np.isnan(rr_p) or rr_p < cfg.min_rpm or rr_p > cfg.max_rpm:
        rr_f = rr_from_fft(z, cfg)
        return rr_f
    return rr_p
