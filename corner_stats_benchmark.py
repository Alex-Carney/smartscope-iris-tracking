# ASCII only
# corner_stats_benchmark.py
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

class CornerStatsBenchmark:
    """
    Collect per-frame corner positions (X,Y for 4 corners), compute stats and whiteness,
    and render a compact figure.

    We compute per-stream:
      - median, std, max |dev|, 99.9% |dev|
      - Welch PSD (detrended), log-log slope as "whiteness" (|slope| ~ 0 => white)
      - lag-1 autocorrelation |rho1| (0 => white)

    add(corners_px): corners in PIXELS, shape (4,2) float32
    We convert to mm using frame-local scale derived from edges each frame.
    """

    def __init__(
        self,
        aruco_w_mm: float,
        aruco_h_mm: float,
        fps_hint: float,
        out_path: str = "corner_stats.png",
        title: str = "Corner Statistics",
    ):
        self.aruco_w_mm = float(aruco_w_mm)
        self.aruco_h_mm = float(aruco_h_mm)
        self.fs = float(fps_hint)
        self.out_path = out_path
        self.title = title

        # 4 corners × 2 axes -> store as list of arrays
        self.series: List[List[float]] = [[]
            for _ in range(8)]  # order: c0x,c0y,c1x,c1y,c2x,c2y,c3x,c3y

    @staticmethod
    def _mm_scale_from_corners(corners_px: np.ndarray, aruco_w_mm: float, aruco_h_mm: float) -> Tuple[float, float]:
        c = corners_px.astype(np.float64).reshape(4, 2)
        px_w = float(np.linalg.norm(c[1] - c[0]))
        px_h = float(np.linalg.norm(c[2] - c[1]))
        if px_w <= 0.0 or px_h <= 0.0:
            return 0.0, 0.0
        return aruco_w_mm / px_w, aruco_h_mm / px_h

    def add(self, corners_px: np.ndarray) -> None:
        if corners_px is None or len(corners_px) != 4:
            return
        sx, sy = self._mm_scale_from_corners(corners_px, self.aruco_w_mm, self.aruco_h_mm)
        if sx <= 0.0 or sy <= 0.0:
            return
        for i in range(4):
            x_mm = float(corners_px[i, 0]) * sx
            y_mm = float(corners_px[i, 1]) * sy
            self.series[2*i + 0].append(x_mm)
            self.series[2*i + 1].append(y_mm)

    @staticmethod
    def _stats(a: np.ndarray):
        if a.size == 0:
            return dict(median=np.nan, std=np.nan, p999=np.nan, dmax=np.nan)
        med = np.median(a)
        dev = np.abs(a - med)
        return dict(
            median=float(med),
            std=float(np.std(a)),
            p999=float(np.percentile(dev, 99.9)),
            dmax=float(np.max(dev)),
        )

    @staticmethod
    def _lag1_autocorr(a: np.ndarray):
        if a.size < 3:
            return np.nan
        x = a - np.mean(a)
        num = np.dot(x[:-1], x[1:])
        den = np.dot(x, x)
        if den == 0.0:
            return np.nan
        return float(num / den)

    def _welch_slope(self, a: np.ndarray):
        """
        Welch PSD slope on log10 S vs log10 f over a midband (exclude lowest/highest).
        Returns slope (ideal white: ~0) and f-range used.
        """
        if a.size < 16 or self.fs <= 0:
            return np.nan, (np.nan, np.nan)
        x = a - np.median(a)
        # Welch
        nperseg = max(16, min(1024, 1 << int(np.floor(np.log2(max(16, a.size // 4))))))
        f, Pxx = welch(x, fs=self.fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant", scaling="density")
        # keep finite, positive
        m = np.isfinite(Pxx) & np.isfinite(f) & (f > 0)
        f, Pxx = f[m], Pxx[m]
        if f.size < 8:
            return np.nan, (np.nan, np.nan)
        # Midband: 10%..60% of usable range
        lo = int(0.10 * f.size)
        hi = int(0.60 * f.size)
        if hi <= lo + 2:
            lo, hi = 0, f.size
        f_fit = f[lo:hi]
        s_fit = Pxx[lo:hi]
        if f_fit.size < 8:
            return np.nan, (np.nan, np.nan)
        X = np.log10(f_fit)
        Y = np.log10(s_fit + 1e-30)
        # linear fit
        A = np.vstack([X, np.ones_like(X)]).T
        sol, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        slope = float(sol[0])
        return slope, (float(f_fit[0]), float(f_fit[-1]))

    def finish(self) -> Optional[str]:
        # Convert to ndarray
        arrs = [np.asarray(s, dtype=float) for s in self.series]
        if all(a.size == 0 for a in arrs):
            print("CornerStatsBenchmark: no data")
            return None

        labels = [
            "C0 X", "C0 Y", "C1 X", "C1 Y", "C2 X", "C2 Y", "C3 X", "C3 Y"
        ]
        stats = [self._stats(a) for a in arrs]
        rho1 = [self._lag1_autocorr(a) for a in arrs]
        slopes = [self._welch_slope(a)[0] for a in arrs]

        # ---- Figure: 2 rows × 4 cols: X on row1, Y on row2 (each corner) ----
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False, sharey=False)
        t = None
        for corner in range(4):
            # X
            ax = axes[0, corner]
            x = arrs[2*corner + 0]
            if x.size > 0:
                if t is None: t = np.arange(x.size)
                xdev = x - np.median(x)
                ax.plot(t[:xdev.size], xdev, linewidth=1.0)
                s = stats[2*corner + 0]; r = rho1[2*corner + 0]; sl = slopes[2*corner + 0]
                ax.set_title(f"C{corner} X\nstd={s['std']:.4g}  p99.9|d|={s['p999']:.4g}\n|ρ₁|={abs(r):.3f}  |slope|={abs(sl):.3f}", fontsize=10)
                ax.set_ylabel("X dev [mm]" if corner == 0 else "")
                ax.grid(True, alpha=0.3)
            # Y
            ay = axes[1, corner]
            y = arrs[2*corner + 1]
            if y.size > 0:
                if t is None: t = np.arange(y.size)
                ydev = y - np.median(y)
                ay.plot(t[:ydev.size], ydev, linewidth=1.0)
                s = stats[2*corner + 1]; r = rho1[2*corner + 1]; sl = slopes[2*corner + 1]
                ay.set_title(f"C{corner} Y\nstd={s['std']:.4g}  p99.9|d|={s['p999']:.4g}\n|ρ₁|={abs(r):.3f}  |slope|={abs(sl):.3f}", fontsize=10)
                ay.set_ylabel("Y dev [mm]" if corner == 0 else "")
                ay.set_xlabel("sample idx")
                ay.grid(True, alpha=0.3)

        fig.suptitle(self.title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"CornerStatsBenchmark: saved {self.out_path}")

        # ---- Console summary ----
        print("\n--- Corner per-axis stats ---")
        for i, (lab, st, r, sl) in enumerate(zip(labels, stats, rho1, slopes)):
            print(f"{lab:5s} | median={st['median']:.6g}  std={st['std']:.6g}  p99.9|d|={st['p999']:.6g}  "
                  f"dmax={st['dmax']:.6g}  |ρ₁|={abs(r):.4f}  |slope|={abs(sl):.4f}")

        return self.out_path
