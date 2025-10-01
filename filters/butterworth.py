# ASCII only
# filters/biquad.py
from typing import Optional, Tuple, Literal
import numpy as np
from scipy.signal import iirfilter, sosfilt  # design once; streaming uses custom SOS
from .base import BaseFilter

class SOSSection:
    __slots__ = ("b0","b1","b2","a1","a2","x1","x2","y1","y2")
    def __init__(self, b0,b1,b2,a1,a2):
        self.b0, self.b1, self.b2, self.a1, self.a2 = map(float, (b0,b1,b2,a1,a2))
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
    def step(self, x: float) -> float:
        y = self.b0*x + self.b1*self.x1 + self.b2*self.x2 - self.a1*self.y1 - self.a2*self.y2
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y
    def reset(self):
        self.x1=self.x2=self.y1=self.y2=0.0

class BiquadLowpass(BaseFilter):
    """
    Causal IIR low-pass as a cascade of biquads (SOS).
    type: 'butter' | 'cheby1' | 'cheby2' | 'ellip'
    order: even recommended (2,4,6,...)
    fc_hz: cutoff (–3 dB for Butterworth)
    """
    def __init__(self,
                 fs_hz: float,
                 fc_hz: float,
                 order: int = 4,
                 ftype: Literal["butter","cheby1","cheby2","ellip"]="butter",
                 rp: float = 1.0,   # passband ripple (dB) for cheby1/ellip
                 rs: float = 40.0   # stopband attenuation (dB) for cheby2/ellip
                 ):
        if fs_hz <= 0 or fc_hz <= 0 or fc_hz >= fs_hz/2:
            raise ValueError("invalid fs/fc")
        self.fs = float(fs_hz)
        self.fc = float(fc_hz)
        self.order = int(order)
        self.ftype = ftype
        kwargs = dict(output="sos", fs=self.fs)
        if ftype in ("cheby1","ellip"): kwargs["rp"] = rp
        if ftype in ("cheby2","ellip"): kwargs["rs"] = rs
        sos = iirfilter(self.order, self.fc, btype="low", ftype=self.ftype, **kwargs)
        # store as sections
        self.sos = [SOSSection(s[0],s[1],s[2],s[4],s[5]) for s in sos]

    def process(self, x: float) -> Optional[float]:
        y = float(x)
        for s in self.sos:
            y = s.step(y)
        return y

    def reset(self) -> None:
        for s in self.sos:
            s.reset()

    def impulse_response(self, max_lags: int = 512, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        # simulate impulse through sections
        K = max_lags
        w = np.zeros(K, dtype=float)
        for n in range(K):
            x = 1.0 if n == 0 else 0.0
            y = x
            for s in self.sos:
                y = s.step(y)
            w[n] = y
        # reset state because we touched it
        self.reset()
        # truncate tiny tail
        if np.abs(w).sum() > 0:
            c = np.cumsum(np.abs(w[::-1]))
            T = K - np.argmax(c > tol * c[0]) - 1
            w = w[:max(8, T)]
        lags = -np.arange(0, w.size, dtype=int)
        # normalize DC gain to 1 (should already be)
        if w.sum() != 0:
            w = w / w.sum()
        return lags, w

    def __str__(self) -> str:
        return f"{self.ftype.capitalize()} LP o={self.order}, fc={self.fc:g}Hz"
