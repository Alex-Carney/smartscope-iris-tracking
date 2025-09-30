# ASCII only
# smartscope_aruco/noise_benchmark.py
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import allantools as at


class NoiseBenchmark:
    """
    Tracks positions and, on finish(), saves a figure with:
      (1) DC-removed time trace vs sample index
      (2) Amplitude spectrum (single-sided) via rFFT (requires fps_hint)
      (3) Overlapping Allan deviation (OADEV) via allantools (if installed)

    Notes:
      - We DC-remove using the median (robust). Switch to mean if you prefer.
      - FFT assumes (approximately) uniform sampling at fps_hint (Hz).
      - OADEV treats the position series as 'phase' data.
    """

    def __init__(
        self,
        out_path: str = "noise_benchmark.png",
        fps_hint: Optional[float] = None,
        use_median_dc: bool = True,
        title: str = "Noise Benchmark"
    ):
        self.out_path = out_path
        self.fps_hint = fps_hint
        self.use_median_dc = use_median_dc
        self.title = title
        self._xs: List[float] = []
        self._ys: List[float] = []

    def add(self, x_mm: float, y_mm: float) -> None:
        self._xs.append(float(x_mm))
        self._ys.append(float(y_mm))

    def _dc_remove(self, arr: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.use_median_dc:
            dc = float(np.median(arr))
        else:
            dc = float(np.mean(arr))
        return arr - dc, dc

    def _compute_fft(self, dev: np.ndarray, fs: float):
        """
        Returns (freqs_Hz, amp_mm) for the single-sided spectrum of the detrended series.
        """
        n = dev.size
        if n < 8:
            return None, None
        # Hann window to reduce leakage
        win = np.hanning(n)
        devw = dev * win
        # rfft single-sided
        spec = np.fft.rfft(devw)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        # Simple amplitude scaling: normalize by window sum to get approx amplitude in mm
        # Multiply by 2 for single-sided (except DC and Nyquist bins).
        # This is a heuristic scaling; for strict PSD/ASD use Welch instead.
        scale = (2.0 / np.sum(win))
        amp = np.abs(spec) * scale
        if n % 2 == 0:
            # even length includes Nyquist bin at the end; don't double that one
            if amp.size > 1:
                amp[1:-1] = amp[1:-1]
        else:
            if amp.size > 1:
                amp[1:] = amp[1:]
        return freqs, amp

    def _compute_allan(self, dev: np.ndarray, fs: float):
        """
        Overlapping Allan deviation (OADEV) using allantools.
        Treat positions as 'phase' data. Returns (taus_s, oadev).
        """
        if at is None or dev.size < 16 or fs <= 0:
            return None, None
        n = dev.size
        tau0 = 1.0 / fs
        # Reasonable tau grid: from tau0 to ~N/2 samples
        hi_samp = max(2, n // 2)
        if hi_samp <= 2:
            return None, None
        taus = np.logspace(np.log10(tau0), np.log10(hi_samp * tau0), num=30)
        try:
            taus_out, oadev, oadev_err, ns = at.oadev(dev, rate=fs, data_type="phase", taus=taus)
            return np.asarray(taus_out, dtype=float), np.asarray(oadev, dtype=float)
        except Exception:
            return None, None

    def finish(self) -> Optional[str]:
        if len(self._xs) < 2:
            print("NoiseBenchmark: not enough samples to analyze.")
            return None

        x = np.asarray(self._xs, dtype=float)
        y = np.asarray(self._ys, dtype=float)

        # DC removal (use median for robustness)
        x_dev, x_dc = self._dc_remove(x)
        y_dev, y_dc = self._dc_remove(y)

        # Build figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
        fig.suptitle(self.title)

        # 1) Time trace (index vs deviation)
        ax = axes[0]
        idx = np.arange(x_dev.size)
        ax.plot(idx, x_dev, label="X dev [mm]")
        ax.plot(idx, y_dev, label="Y dev [mm]")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Deviation [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("DC-removed time trace (median-subtracted)")

        # 2) FFT (amplitude spectrum)
        ax = axes[1]
        fs = float(self.fps_hint) if self.fps_hint else None
        if fs and fs > 0.0:
            fx, ax_amp = self._compute_fft(x_dev, fs)
            fy, ay_amp = self._compute_fft(y_dev, fs)
            if fx is not None and ax_amp is not None:
                # Avoid log(0) at DC in log plots: skip f=0
                m = fx > 0
                ax.plot(fx[m], ax_amp[m], label="|X(f)| [mm]")
            if fy is not None and ay_amp is not None:
                m = fy > 0
                ax.plot(fy[m], ay_amp[m], label="|Y(f)| [mm]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Amplitude [mm]")
            ax.set_title("Amplitude spectrum (Hann windowed, single-sided)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "FFT skipped (no fps_hint)", ha="center", va="center")
            ax.axis("off")

        # 3) Allan deviation (OADEV)
        ax = axes[2]
        if fs and fs > 0.0:
            tx, axadev = self._compute_allan(x_dev, fs)
            ty, ayadev = self._compute_allan(y_dev, fs)
            plotted = False
            if tx is not None and axadev is not None:
                ax.loglog(tx, axadev, label="X OADEV [mm]")
                plotted = True
            if ty is not None and ayadev is not None:
                ax.loglog(ty, ayadev, label="Y OADEV [mm]")
                plotted = True
            if plotted:
                ax.set_xlabel("τ [s]")
                ax.set_ylabel("Allan deviation [mm]")
                ax.set_title("Overlapping Allan deviation (allantools)")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(loc="upper right", fontsize=9)
            else:
                ax.text(0.5, 0.5, "Allan deviation skipped (need allantools + enough samples)", ha="center", va="center")
                ax.axis("off")
        else:
            ax.text(0.5, 0.5, "Allan deviation skipped (no fps_hint)", ha="center", va="center")
            ax.axis("off")

        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"NoiseBenchmark: saved plot to {self.out_path}")
        return self.out_path
