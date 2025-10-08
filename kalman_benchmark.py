# ASCII only
# kalman_benchmark.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

@dataclass
class KalmanBenchmarkConfig:
    out_path: str = "kalman_benchmark.png"
    fps_hint: float = 60.0
    title: str = "Kalman vs Raw"

class KalmanBenchmark:
    """
    Collect raw center (x,y) and KF center (x,y), then render:
      - time traces (detrended by median) with horizontal "noise floor" lines
      - Welch PSDs for raw and KF

    Usage:
      kb = KalmanBenchmark(KalmanBenchmarkConfig(...))
      kb.add(raw_x, raw_y, kf_x, kf_y)  # each frame
      kb.finish()
    """
    def __init__(self, cfg: KalmanBenchmarkConfig):
        self.cfg = cfg
        self._rx: List[float] = []
        self._ry: List[float] = []
        self._fx: List[float] = []
        self._fy: List[float] = []

    def add(self, raw_x_mm: float, raw_y_mm: float, kf_x_mm: float, kf_y_mm: float) -> None:
        self._rx.append(float(raw_x_mm))
        self._ry.append(float(raw_y_mm))
        self._fx.append(float(kf_x_mm))
        self._fy.append(float(kf_y_mm))

    @staticmethod
    def _noise_floor(a: np.ndarray) -> float:
        if a.size == 0:
            return np.nan
        med = np.median(a)
        dev = np.abs(a - med)
        return float(np.percentile(dev, 99.9))

    @staticmethod
    def _welch(fs: float, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if a.size < 16 or fs <= 0: return np.array([]), np.array([])
        x = a - np.median(a)
        # power-of-two segment roughly a quarter of the record, bounded
        nps = max(16, min(1024, 1 << int(np.floor(np.log2(max(16, a.size // 4))))))
        f, Pxx = welch(x, fs=fs, nperseg=nps, noverlap=nps//2, detrend="constant", scaling="density")
        m = np.isfinite(f) & np.isfinite(Pxx) & (f > 0)
        return f[m], Pxx[m]

    def finish(self) -> Optional[str]:
        rx = np.asarray(self._rx, dtype=float)
        ry = np.asarray(self._ry, dtype=float)
        fx = np.asarray(self._fx, dtype=float)
        fy = np.asarray(self._fy, dtype=float)

        if rx.size == 0:
            print("KalmanBenchmark: no samples")
            return None

        t = np.arange(rx.size)
        fs = float(self.cfg.fps_hint)

        # Noise floors (99.9% |dev|)
        nf_rx = self._noise_floor(rx)
        nf_ry = self._noise_floor(ry)
        nf_fx = self._noise_floor(fx)
        nf_fy = self._noise_floor(fy)

        # PSDs
        frx, Prx = self._welch(fs, rx)
        fry, Pry = self._welch(fs, ry)
        ffx, Pfx = self._welch(fs, fx)
        ffy, Pfy = self._welch(fs, fy)

        # Figure: 2 rows (X,Y) × 2 cols (Time trace, Welch PSD)
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
        fig.suptitle(self.cfg.title, fontsize=14)

        # --- Row 0: X ---
        ax_x_time = axes[0, 0]
        ax_x_psd  = axes[0, 1]

        xdev_raw = rx - np.median(rx)
        xdev_kf  = fx - np.median(fx)
        ax_x_time.plot(t, xdev_raw, label=f"RAW  (NF={nf_rx:.4g} mm)", linewidth=1.1)
        ax_x_time.plot(t, xdev_kf,  label=f"KF   (NF={nf_fx:.4g} mm)", linewidth=1.1)
        ax_x_time.axhline(+nf_rx, linestyle="--", linewidth=0.9, alpha=0.6)
        ax_x_time.axhline(-nf_rx, linestyle="--", linewidth=0.9, alpha=0.6)
        ax_x_time.set_ylabel("X dev [mm]")
        ax_x_time.set_xlabel("sample idx")
        ax_x_time.grid(True, alpha=0.3)
        ax_x_time.legend(loc="upper right", fontsize=9, frameon=True)

        if frx.size > 0 and ffx.size > 0:
            ax_x_psd.loglog(frx, Prx, label="RAW", linewidth=1.1)
            ax_x_psd.loglog(ffx, Pfx, label="KF",  linewidth=1.1)
            ax_x_psd.set_ylabel("PSD X [mm^2/Hz]")
            ax_x_psd.set_xlabel("Frequency [Hz]")
            ax_x_psd.grid(True, which="both", alpha=0.3)
            ax_x_psd.legend(loc="upper right", fontsize=9, frameon=True)

        # --- Row 1: Y ---
        ax_y_time = axes[1, 0]
        ax_y_psd  = axes[1, 1]

        ydev_raw = ry - np.median(ry)
        ydev_kf  = fy - np.median(fy)
        ax_y_time.plot(t, ydev_raw, label=f"RAW  (NF={nf_ry:.4g} mm)", linewidth=1.1)
        ax_y_time.plot(t, ydev_kf,  label=f"KF   (NF={nf_fy:.4g} mm)", linewidth=1.1)
        ax_y_time.axhline(+nf_ry, linestyle="--", linewidth=0.9, alpha=0.6)
        ax_y_time.axhline(-nf_ry, linestyle="--", linewidth=0.9, alpha=0.6)
        ax_y_time.set_ylabel("Y dev [mm]")
        ax_y_time.set_xlabel("sample idx")
        ax_y_time.grid(True, alpha=0.3)
        ax_y_time.legend(loc="upper right", fontsize=9, frameon=True)

        if fry.size > 0 and ffy.size > 0:
            ax_y_psd.loglog(fry, Pry, label="RAW", linewidth=1.1)
            ax_y_psd.loglog(ffy, Pfy, label="KF",  linewidth=1.1)
            ax_y_psd.set_ylabel("PSD Y [mm^2/Hz]")
            ax_y_psd.set_xlabel("Frequency [Hz]")
            ax_y_psd.grid(True, which="both", alpha=0.3)
            ax_y_psd.legend(loc="upper right", fontsize=9, frameon=True)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(self.cfg.out_path, dpi=150)
        plt.close(fig)
        print(f"KalmanBenchmark: saved {self.cfg.out_path}")

        # Console summary
        print("\n--- Kalman vs RAW noise floors (99.9% |dev|) ---")
        print(f"X: RAW={nf_rx:.6g} mm   KF={nf_fx:.6g} mm   (ratio={nf_fx/max(1e-18,nf_rx):.3f})")
        print(f"Y: RAW={nf_ry:.6g} mm   KF={nf_fy:.6g} mm   (ratio={nf_fy/max(1e-18,nf_ry):.3f})")

        return self.cfg.out_path
