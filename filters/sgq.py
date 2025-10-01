# ASCII only
# smartscope_aruco/filters/sg_causal.py
from collections import deque
from typing import Optional, Tuple
import numpy as np
from filters.base import BaseFilter

def _causal_savgol_weights(window: int, polyorder: int) -> np.ndarray:
    """
    Compute FIR weights h for a causal Savitzky–Golay smoother:
    Fit a polynomial of degree p to the last W samples at times t=0,-1,-2,...,
    return estimated value at t=0 as a linear comb. of samples.
      y[n] = sum_{k=0..W-1} h[k] * x[n-k]
    This is LTI, strictly causal, and introduces a constant delay equal to
    the center-of-mass of h (not (W-1)/2 unless symmetric).
    """
    if window < 3 or polyorder >= window:
        raise ValueError("window must be >=3 and polyorder < window")
    # times for samples x[n-k]
    t = -np.arange(0, window, dtype=float)  # [0,-1,-2,...,-W+1]
    # design matrix A (W x (p+1)): columns [t^0, t^1, ..., t^p]
    A = np.vstack([t**j for j in range(polyorder + 1)]).T
    # We want y(0) = a0, where a = (A^T A)^{-1} A^T x
    # So y = e0^T a = e0^T (A^T A)^{-1} A^T x = h^T x with:
    e0 = np.zeros((polyorder + 1, 1)); e0[0, 0] = 1.0
    AtA = A.T @ A
    h = A @ np.linalg.pinv(AtA) @ e0  # (W x 1)
    h = h[:, 0]
    # Numerical tidy: enforce unity DC gain
    h = h / np.sum(h)
    return h.astype(float)

class CausalSavGol(BaseFilter):
    """
    Causal Savitzky–Golay polynomial smoother (right-edge fit).
    Static FIR: y[n] = sum_{k=0..W-1} h[k] x[n-k], with h from LS fit.
    """
    def __init__(self, window: int, polyorder: int = 2):
        if window % 1 != 0 or window < 3:
            raise ValueError("window must be integer >= 3")
        if polyorder < 0:
            raise ValueError("polyorder must be >= 0")
        if polyorder >= window:
            raise ValueError("polyorder must be < window")
        self.W = int(window)
        self.P = int(polyorder)
        self.h = _causal_savgol_weights(self.W, self.P)  # length W, for lags 0..W-1
        self.buf = deque(maxlen=self.W)  # store recent samples (newest at left)

    def process(self, x: float) -> Optional[float]:
        self.buf.appendleft(float(x))  # buf[0]=x[n], buf[1]=x[n-1], ...
        if len(self.buf) < self.W:
            return None
        # align h[k] with buf[k]
        b = np.fromiter(self.buf, dtype=float, count=self.W)
        return float(np.dot(self.h, b))

    def reset(self) -> None:
        self.buf.clear()

    def impulse_response(self, max_lags: int = 200, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        lags = -np.arange(0, self.W, dtype=int)
        return lags, self.h.copy()

    def __str__(self) -> str:
        return f"SG(causal) p={self.P}, w={self.W}"
