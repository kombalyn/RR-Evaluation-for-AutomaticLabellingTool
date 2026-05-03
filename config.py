"""RR értékelés — hiperparaméterek (cikk / pipeline szerint hangolható)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RREvalConfig:
    """Légzési jel feldolgozása és RR becslés."""

    fps: float = 20.0
    bandpass_low_hz: float = 0.3
    bandpass_high_hz: float = 2.0
    bandpass_order: int = 6
    # Ablakolás: referencia RR átlaga az ablakon (perc alapú)
    window_seconds: float = 20.0
    # RR fiziológiás tartomány (BPM)
    min_rpm: float = 15.0
    max_rpm: float = 120.0
    # Peak-alapú becslés
    peak_min_distance_sec: float = 0.5  # max ~120 BPM
    peak_height_std_factor: float = 0.3
    peak_min_count: int = 3  # medián periódushoz
    # FFT tartalék (Hz)
    fft_freq_low_hz: float = 0.3
    fft_freq_high_hz: float = 2.0
    # Bootstrap a MAE-re (abszolút hibák)
    bootstrap_n: int = 1000
    bootstrap_seed: int | None = 42
