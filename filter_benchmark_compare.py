# ASCII only
# smartscope_aruco/filter_benchmark_compare.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import copy

from .filters.base import BaseFilter


class FilterBenchmarkCompare:
    """
    Feed raw (x_mm, y_mm) to this during your run. It will run two filters
    in parallel (independent instances for X and Y), and at finish() it will
    render a 2x3 grid:

      Row 1: DC-removed time traces (RAW, Filter A, Filter B)
      Row 2: Welch PSDs mm^2/Hz (RAW, Filter A, Filter B)

    Column titles include the filter name and its nominal delay in milliseconds.
    """

    def __init__(
        self,
        filter_a: BaseFilter,
        filter_b: BaseFilter,
        fps_hint: Optional[float],
        out_path: str = "filter_compare.png",
        title: str = "Filter Comparison",
    ):
        self.fps = float(fps_hint) if fps_hint else None
        self.out_path = out_path
        self.title = title

        # Clone per-axis so states are independent
        self.fx_a = copy.deepcopy(filter_a)
        self.fy_a = copy.deepcopy(filter_a)
        self.fx_b = copy.deepcopy(filter_b)
        self.fy_b = copy.deepcopy(filter_b)

        # Raw + filtered traces
        self.x_raw, self.y_raw = [], []
        self.xa, self.ya = [], []
        self.xb, self.yb = [], []

    # ---- feed one raw sample ----
    def add(self, x_mm: float, y_mm: float) -> None:
        x = float(x_mm); y = float(y_mm)
        self.x_raw.append(x); self.y_raw.append(y)

        ya = self.fx_a.process(x); yya = self.fy_a.process(y)
        if ya is not None and yya is not None:
            self.xa.append(ya); self.ya.append(yya)

        yb = self.fx_b.process(x); yyb = self.fy_b.process(y)
        if yb is not None and yyb is not None:
            self.xb.append(yb); self.yb.append(yyb)

    # ---- helpers ----
    @staticmethod
    def _dc(arr: np.ndarray) -> np.ndarray:
        return arr - float(np.median(arr)) if arr.size else arr

    def _welch(self, dev: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.fps is None or dev.size < 32:
            return None, None
        n = dev.size
        nperseg = min(max(256, 2 ** int(np.floor(np.log2(n // 4)))), n)
        f, Pxx = welch(
            dev,
            fs=self.fps,
            window=get_window("hann", nperseg),
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            return_onesided=True,
        )
        return f, Pxx  # mm^2/Hz

    def _delay_ms(self, filt: BaseFilter) -> Optional[float]:
        if self.fps is None:
            return None
        return 1000.0 * filt.effective_delay_samples() / self.fps

    # ---- render ----
    def finish(self) -> Optional[str]:
        xr = np.asarray(self.x_raw, float); yr = np.asarray(self.y_raw, float)
        xa = np.asarray(self.xa, float);    ya = np.asarray(self.ya, float)
        xb = np.asarray(self.xb, float);    yb = np.asarray(self.yb, float)

        if xr.size < 2:
            print("FilterBenchmarkCompare: not enough samples.")
            return None

        # Prepare deviations (median-subtracted)
        xr_d = self._dc(xr); yr_d = self._dc(yr)
        xa_d = self._dc(xa); ya_d = self._dc(ya)
        xb_d = self._dc(xb); yb_d = self._dc(yb)

        # Figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(self.title)

        # Titles with delay
        da_ms = self._delay_ms(self.fx_a)
        db_ms = self._delay_ms(self.fx_b)
        ttl_raw = "RAW (delay = 0 ms)"
        ttl_a   = f"{str(self.fx_a)} (delay ≈ {da_ms:.2f} ms)" if da_ms is not None else str(self.fx_a)
        ttl_b   = f"{str(self.fx_b)} (delay ≈ {db_ms:.2f} ms)" if db_ms is not None else str(self.fx_b)

        # ---- Row 1: time traces (X & Y overlays) ----
        panels = [
            (axes[0, 0], xr_d, yr_d, ttl_raw),
            (axes[0, 1], xa_d, ya_d, ttl_a),
            (axes[0, 2], xb_d, yb_d, ttl_b),
        ]
        for ax, dx, dy, title in panels:
            if dx.size:
                ax.plot(np.arange(dx.size), dx, linewidth=0.8, label="X dev [mm]")
            if dy.size:
                ax.plot(np.arange(dy.size), dy, linewidth=0.8, label="Y dev [mm]")
            ax.set_title(title)
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Deviation [mm]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)

        # ---- Row 2: Welch PSDs ----
        panels = [
            (axes[1, 0], xr_d, yr_d, ttl_raw),
            (axes[1, 1], xa_d, ya_d, ttl_a),
            (axes[1, 2], xb_d, yb_d, ttl_b),
        ]
        for ax, dx, dy, title in panels:
            fx, Px = self._welch(dx) if dx.size else (None, None)
            fy, Py = self._welch(dy) if dy.size else (None, None)
            if fx is not None and Px is not None and np.any(fx > 0):
                m = fx > 0
                ax.loglog(fx[m], Px[m], label="X PSD [mm²/Hz]")
            if fy is not None and Py is not None and np.any(fy > 0):
                m = fy > 0
                ax.loglog(fy[m], Py[m], label="Y PSD [mm²/Hz]")
            ax.set_title("Welch PSD (Hann, density)")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [mm²/Hz]")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)

        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"FilterBenchmarkCompare: saved figure to {self.out_path}")
        return self.out_path
