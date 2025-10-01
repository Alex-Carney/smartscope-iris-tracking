# ASCII only
# filters/ema_cascade.py
from typing import Optional, Tuple
import numpy as np
from .base import BaseFilter

class CascadedEMA(BaseFilter):
    """
    N EMAs in series. Magnitude roll-off ~ 6*N dB/oct.
    Delay (center-of-mass) ≈ N * (1-α)/α samples.
    """
    def __init__(self, alpha: float, stages: int = 2):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1]")
        if stages < 1:
            raise ValueError("stages must be >= 1")
        self.alpha = float(alpha)
        self.N = int(stages)
        self._y = [None] * self.N

    def process(self, x: float) -> Optional[float]:
        a = self.alpha
        y = float(x)
        for i in range(self.N):
            if self._y[i] is None:
                self._y[i] = y
            else:
                self._y[i] = a * y + (1.0 - a) * self._y[i]
            y = self._y[i]
        return y

    def reset(self) -> None:
        for i in range(self.N):
            self._y[i] = None

    def impulse_response(self, max_lags: int = 200, tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        # Closed form of N-fold EMA: negative-binomial weights
        a = self.alpha
        r = 1.0 - a
        # choose K so tail weight < tol
        # tail ~ const * r^k -> k ~ log(tol)/log(r)
        K = max(1, min(int(np.ceil(np.log(max(tol, 1e-12)) / np.log(max(r, 1e-12)))) + self.N, max_lags))
        k = np.arange(0, K, dtype=int)
        # w[k] = a^N * C(k+N-1, N-1) * r^k
        from math import comb
        nb = np.array([comb(kj + self.N - 1, self.N - 1) for kj in k], dtype=float)
        w = (a ** self.N) * nb * (r ** k)
        w /= np.sum(w)  # unity DC gain numerically
        lags = -k
        return lags, w

    def effective_delay_samples(self) -> float:
        return self.N * (1.0 - self.alpha) / self.alpha

    def __str__(self) -> str:
        return f"EMAx{self.N} α={self.alpha:g}"

def alpha_for_target_delay(target_delay_samples: float, stages: int) -> float:
    """Pick α so cascaded EMA has given center-of-mass delay (samples)."""
    d = max(1e-12, float(target_delay_samples))
    N = max(1, int(stages))
    return N / (N + d)
