# ASCII only
# corner_stats_benchmark.py
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

class CornerStatsBenchmark:
    """
    Collect per-frame corner positions (X,Y for 4 corners), compute stats,
    whiteness, and an online 8x8 covariance matrix in mm for the vector:
        z = [C0x, C1x, C2x, C3x, C0y, C1y, C2y, C3y]

    add(corners_px): corners in PIXELS, shape (4,2) float32
    We convert to mm using a per-frame scale derived from the detected edges.
    """

    def __init__(
        self,
        aruco_w_mm: float,
        aruco_h_mm: float,
        fps_hint: float,
        out_path: str = "corner_stats.png",
        title: str = "Corner Statistics",
        cov_npy_path: str = "corner_cov.npy",
        cov_fig_path: str = "corner_cov_heatmap.png",
    ):
        self.aruco_w_mm = float(aruco_w_mm)
        self.aruco_h_mm = float(aruco_h_mm)
        self.fs = float(fps_hint)
        self.out_path = out_path
        self.title = title
        self.cov_npy_path = cov_npy_path
        self.cov_fig_path = cov_fig_path

        # For plots: 4 corners × 2 axes, interleaved [c0x,c0y,c1x,c1y,c2x,c2y,c3x,c3y]
        self.series: List[List[float]] = [[] for _ in range(8)]

        # Online covariance (Welford) for z = [x0,x1,x2,x3,y0,y1,y2,y3] in mm
        self._n = 0
        self._mean = np.zeros(8, dtype=np.float64)
        self._M2   = np.zeros((8, 8), dtype=np.float64)

    @staticmethod
    def _mm_scale_from_corners(corners_px: np.ndarray, aruco_w_mm: float, aruco_h_mm: float) -> Tuple[float, float]:
        c = corners_px.astype(np.float64).reshape(4, 2)
        px_w = float(np.linalg.norm(c[1] - c[0]))
        px_h = float(np.linalg.norm(c[2] - c[1]))
        if px_w <= 0.0 or px_h <= 0.0:
            return 0.0, 0.0
        return aruco_w_mm / px_w, aruco_h_mm / px_h

    def _cov_add(self, z: np.ndarray) -> None:
        """Welford multivariate update for the 8-dim vector z."""
        self._n += 1
        delta  = z - self._mean
        self._mean += delta / self._n
        delta2 = z - self._mean
        self._M2 += np.outer(delta, delta2)

    def covariance(self) -> Optional[np.ndarray]:
        if self._n < 2:
            return None
        return self._M2 / (self._n - 1)

    def count(self) -> int:
        return self._n

    def add(self, corners_px: np.ndarray) -> None:
        if corners_px is None or len(corners_px) != 4:
            return
        sx, sy = self._mm_scale_from_corners(corners_px, self.aruco_w_mm, self.aruco_h_mm)
        if sx <= 0.0 or sy <= 0.0:
            return

        # For plots (interleaved)
        for i in range(4):
            x_mm = float(corners_px[i, 0]) * sx
            y_mm = float(corners_px[i, 1]) * sy
            self.series[2*i + 0].append(x_mm)
            self.series[2*i + 1].append(y_mm)

        # For covariance: z = [x0,x1,x2,x3,y0,y1,y2,y3] in mm
        cx = corners_px[:, 0].astype(np.float64) * sx
        cy = corners_px[:, 1].astype(np.float64) * sy
        z = np.empty(8, dtype=np.float64)
        z[0:4] = cx
        z[4:8] = cy
        self._cov_add(z)

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
        """Welch PSD slope on log10 S vs log10 f over a midband. White ~ 0."""
        if a.size < 16 or self.fs <= 0:
            return np.nan, (np.nan, np.nan)
        x = a - np.median(a)
        nperseg = max(16, min(1024, 1 << int(np.floor(np.log2(max(16, a.size // 4))))))
        f, Pxx = welch(x, fs=self.fs, nperseg=nperseg, noverlap=nperseg//2,
                       detrend="constant", scaling="density")
        m = np.isfinite(Pxx) & np.isfinite(f) & (f > 0)
        f, Pxx = f[m], Pxx[m]
        if f.size < 8:
            return np.nan, (np.nan, np.nan)
        lo = int(0.10 * f.size)
        hi = int(0.60 * f.size)
        if hi <= lo + 2:
            lo, hi = 0, f.size
        f_fit = f[lo:hi]; s_fit = Pxx[lo:hi]
        if f_fit.size < 8:
            return np.nan, (np.nan, np.nan)
        X = np.log10(f_fit); Y = np.log10(s_fit + 1e-30)
        A = np.vstack([X, np.ones_like(X)]).T
        sol, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        slope = float(sol[0])
        return slope, (float(f_fit[0]), float(f_fit[-1]))

    def _save_cov_heatmap(self, R: np.ndarray) -> None:
        labels_R = ["C0x","C1x","C2x","C3x","C0y","C1y","C2y","C3y"]
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        im = ax.imshow(R, cmap="coolwarm", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Covariance [mm^2]")
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(labels_R, rotation=45, ha="right")
        ax.set_yticklabels(labels_R)
        ax.set_title("Corner measurement covariance R (mm^2)")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(self.cov_fig_path, dpi=150)
        plt.close(fig)
        print(f"CornerStatsBenchmark: saved covariance heatmap -> {self.cov_fig_path}")

    def finish(self) -> Optional[str]:
        # Convert to ndarray for plots
        arrs = [np.asarray(s, dtype=float) for s in self.series]
        if all(a.size == 0 for a in arrs):
            print("CornerStatsBenchmark: no data")
            return None

        labels = ["C0 X","C0 Y","C1 X","C1 Y","C2 X","C2 Y","C3 X","C3 Y"]
        stats  = [self._stats(a) for a in arrs]
        rho1   = [self._lag1_autocorr(a) for a in arrs]
        slopes = [self._welch_slope(a)[0] for a in arrs]

        # ---- Figure: 2 rows × 4 cols time traces ----
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False, sharey=False)
        t = None
        for corner in range(4):
            ax = axes[0, corner]
            x = arrs[2*corner + 0]
            if x.size > 0:
                if t is None: t = np.arange(x.size)
                xdev = x - np.median(x)
                ax.plot(t[:xdev.size], xdev, linewidth=1.0)
                s = stats[2*corner + 0]; r = rho1[2*corner + 0]; sl = slopes[2*corner + 0]
                ax.set_title(f"C{corner} X\nstd={s['std']:.4g}  p99.9|d|={s['p999']:.4g}\n|rho1|={abs(r):.3f}  |slope|={abs(sl):.3f}", fontsize=10)
                ax.set_ylabel("X dev [mm]" if corner == 0 else "")
                ax.grid(True, alpha=0.3)

            ay = axes[1, corner]
            y = arrs[2*corner + 1]
            if y.size > 0:
                if t is None: t = np.arange(y.size)
                ydev = y - np.median(y)
                ay.plot(t[:ydev.size], ydev, linewidth=1.0)
                s = stats[2*corner + 1]; r = rho1[2*corner + 1]; sl = slopes[2*corner + 1]
                ay.set_title(f"C{corner} Y\nstd={s['std']:.4g}  p99.9|d|={s['p999']:.4g}\n|rho1|={abs(r):.3f}  |slope|={abs(sl):.3f}", fontsize=10)
                ay.set_ylabel("Y dev [mm]" if corner == 0 else "")
                ay.set_xlabel("sample idx")
                ay.grid(True, alpha=0.3)

        fig.suptitle(self.title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"CornerStatsBenchmark: saved {self.out_path}")

        # ---- Covariance outputs ----
        R = self.covariance()
        if R is None:
            print("CornerStatsBenchmark: covariance not available (n < 2)")
        else:
            np.save(self.cov_npy_path, R)
            print(f"CornerStatsBenchmark: saved 8x8 covariance -> {self.cov_npy_path}")
            # quick numeric summary
            stds = np.sqrt(np.diag(R))
            std_str = ", ".join(f"{v:.4g}" for v in stds)
            print(f"CornerStatsBenchmark: n={self._n}, diag stds (mm) = [{std_str}]")
            # heatmap figure
            self._save_cov_heatmap(R)

        # ---- Console summary per stream ----
        print("\n--- Corner per-axis stats ---")
        for (lab, st, r, sl) in zip(labels, stats, rho1, slopes):
            print(f"{lab:5s} | median={st['median']:.6g}  std={st['std']:.6g}  "
                  f"p99.9|d|={st['p999']:.6g}  dmax={st['dmax']:.6g}  "
                  f"|rho1|={abs(r):.4f}  |slope|={abs(sl):.4f}")

        return self.out_path
