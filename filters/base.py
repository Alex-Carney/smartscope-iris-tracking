# ASCII only
# smartscope_aruco/filters/base.py
from __future__ import annotations
from typing import Optional, Tuple
import copy
import numpy as np

class BaseFilter:
    """
    Streaming scalar filter for real-time use (causal).
    """

    def process(self, x: float) -> Optional[float]:
        """Push one sample, return filtered value (or None until ready)."""
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    # ---------- visualization & latency ----------
    def impulse_response(self, max_lags: int = 200, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lags, weights) where lags = [0, -1, -2, ...].
        For IIR filters, return a truncated tail (by `tol` or `max_lags`).
        """
        raise NotImplementedError

    def effective_delay_samples(self) -> float:
        """
        Center-of-mass of impulse response (in samples, non-negative).
        Computed from impulse_response() by default.
        """
        lags, w = self.impulse_response()
        if w.size == 0:
            return 0.0
        # lags are 0,-1,-2,... ; convert to nonnegative delays d = -lags
        d = -lags
        return float(np.sum(d * w))

    def copy(self) -> "BaseFilter":
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return self.__class__.__name__
