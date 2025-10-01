# ASCII only
# smartscope_aruco/filters/boxcar.py
from collections import deque
from typing import Optional, Tuple
import numpy as np
from base import BaseFilter

class Boxcar(BaseFilter):
    """
    Causal moving average over the last N samples (uniform FIR).
    y[n] = (1/N) * sum_{k=0..N-1} x[n-k]
    """

    def __init__(self, N: int):
        if N < 1:
            raise ValueError("Boxcar N must be >= 1")
        self.N = int(N)
        self.buf = deque()
        self.sum = 0.0

    def process(self, x: float) -> Optional[float]:
        self.buf.append(float(x))
        self.sum += float(x)
        if len(self.buf) > self.N:
            self.sum -= self.buf.popleft()
        if len(self.buf) < self.N:
            return None
        return self.sum / self.N

    def reset(self) -> None:
        self.buf.clear()
        self.sum = 0.0

    def impulse_response(self, max_lags: int = 200, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        w = np.full(self.N, 1.0 / self.N, dtype=float)   # k=0..N-1
        lags = -np.arange(0, self.N, dtype=int)          # [0,-1,-2,...]
        return lags, w

    def __str__(self) -> str:
        return f"Boxcar N={self.N}"
