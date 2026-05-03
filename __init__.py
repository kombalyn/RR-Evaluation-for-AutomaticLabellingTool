"""RR (légzésszám) becslés összehasonlítása referencia jellel."""

from .config import RREvalConfig
from .evaluation import RREvaluationResult, evaluate_signal_vs_reference
from .io_wave import load_wave_matrix

__all__ = [
    "RREvalConfig",
    "RREvaluationResult",
    "evaluate_signal_vs_reference",
    "load_wave_matrix",
]
