# ASCII only
# smartscope_aruco/filters/ema.py
from typing import Optional, Tuple
import numpy as np
from filters.base import BaseFilter

class EMA(BaseFilter):
    """
    Exponential moving average (causal IIR):
      y[n] = α x[n] + (1-α) y[n-1], 0<α<=1.
    Impulse response weights:
      h[0]=α, h[-1]=α(1-α), h[-2]=α(1-α)^2, ...
    """

    def __init__(self, alpha: float):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("EMA alpha must be in (0,1]")
        self.alpha = float(alpha)
        self._y = None  # type: Optional[float]

    def process(self, x: float) -> Optional[float]:
        x = float(x)
        if self._y is None:
            self._y = x
        else:
            a = self.alpha
            self._y = a * x + (1.0 - a) * self._y
        return self._y

    def reset(self) -> None:
        self._y = None

    def impulse_response(self, max_lags: int = 200, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        a = self.alpha
        r = (1.0 - a)
        # truncate where tail weight gets very small
        K = max(1, min(int(np.ceil(np.log(max(tol, 1e-12)) / np.log(max(r, 1e-12)))) + 1, max_lags))
        k = np.arange(0, K, dtype=int)
        w = (a * (r ** k)).astype(float)
        lags = -k  # [0,-1,-2,...]
        return lags, w

    def __str__(self) -> str:
        return f"EMA α={self.alpha:g}"
