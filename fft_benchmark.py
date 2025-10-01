# ASCII only
# smartscope_aruco/noise_benchmark.py
from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import allantools as at
from scipy.signal import welch, get_window


class NoiseBenchmark:
    """
    Tracks positions and, on finish(), saves a figure with:
      (1) DC-removed time trace vs sample index
      (2) Welch PSD (mm^2/Hz) of X,Y (needs fps_hint)
      (3) Overlapping Allan deviation via allantools + fits:
          - Free log–log slope fit: log10(sigma)=m*log10(tau)+b  (report m, R^2)
          - Constrained tau^-1/2 overlay: sigma ≈ K * tau^(-1/2)  (solve K)

    No ASD is computed or reported.
    """

    def __init__(
        self,
        out_path: str = "noise_benchmark.png",
        fps_hint: Optional[float] = None,
        use_median_dc: bool = True,
        title: str = "Noise Benchmark",
        allan_fit_portion: float = 0.4,   # fit first 40% of Allan points
        allan_variant: str = "mdev",
        allan_data_type: str = "freq",
        min_fit_points: int = 8,
        welch_nperseg: Optional[int] = None,
        welch_overlap: float = 0.5
    ):
        self.out_path = out_path
        self.fps_hint = fps_hint
        self.use_median_dc = use_median_dc
        self.title = title
        self.allan_fit_portion = float(np.clip(allan_fit_portion, 0.1, 0.9))
        self.min_fit_points = int(min_fit_points)
        self.welch_nperseg = welch_nperseg
        self.welch_overlap = float(np.clip(welch_overlap, 0.0, 0.95))
        self.allan_variant = allan_variant.lower()  # "oadev" or "mdev"
        self.allan_data_type = allan_data_type.lower()  # "phase" or "freq"

        self._xs: List[float] = []
        self._ys: List[float] = []
        self.results: Dict[str, Dict[str, float]] = {}

    # ---------------- core accumulation ----------------

    def add(self, x_mm: float, y_mm: float) -> None:
        self._xs.append(float(x_mm))
        self._ys.append(float(y_mm))

    @staticmethod
    def _dc_remove(arr: np.ndarray, use_median: bool) -> Tuple[np.ndarray, float]:
        dc = float(np.median(arr)) if use_median else float(np.mean(arr))
        return arr - dc, dc

    # ---------------- PSD (Welch) ----------------

    def _compute_welch(self, dev: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Welch PSD in units of mm^2/Hz.
        """
        n = dev.size
        if n < 16 or fs <= 0:
            return None, None

        # Choose nperseg if not provided: ~1/8 of series, at least 256, power of two
        if self.welch_nperseg is None:
            guess = max(256, int(n // 8))
            # round to nearest power of two not exceeding n
            pow2 = 2 ** int(np.floor(np.log2(max(16, guess))))
            nperseg = min(pow2, n)
        else:
            nperseg = min(int(self.welch_nperseg), n)

        noverlap = int(self.welch_overlap * nperseg)
        window = get_window("hann", nperseg, fftbins=True)
        f, Pxx = welch(
            dev,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=True,
            scaling="density"  # -> mm^2/Hz
        )
        return f, Pxx

    # ---------------- Allan + fits ----------------

    def _compute_allan(self, dev: np.ndarray, fs: float):
        """
        Allan-family deviation using allantools.
        - self.allan_variant: "oadev" (classic overlapping) or "mdev" (modified Allan)
        - self.allan_data_type: "phase" or "freq"
        Returns (taus_s, sigma).
        """
        if at is None or dev.size < 16 or fs <= 0:
            return None, None

        tau0 = 1.0 / fs
        hi_samp = max(2, dev.size // 2)
        taus = np.logspace(np.log10(tau0), np.log10(hi_samp * tau0), num=30)

        try:
            if self.allan_data_type == "freq":
                data = np.diff(dev) * fs  # convert displacement to “frequency-like” samples
                rate = fs  # 1 / tau0 of the derived series
                if self.allan_variant == "mdev":
                    taus_out, sig, err, ns = at.mdev(data, rate=rate, data_type="freq", taus=taus)
                else:
                    taus_out, sig, err, ns = at.oadev(data, rate=rate, data_type="freq", taus=taus)
            else:
                # phase/position data directly
                if self.allan_variant == "mdev":
                    taus_out, sig, err, ns = at.mdev(dev, rate=fs, data_type="phase", taus=taus)
                else:
                    taus_out, sig, err, ns = at.oadev(dev, rate=fs, data_type="phase", taus=taus)
            return np.asarray(taus_out, float), np.asarray(sig, float)
        except Exception:
            return None, None

    @staticmethod
    def _linfit_loglog(taus: np.ndarray, sigmas: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit log10(sigma) = m * log10(tau) + b. Returns (m, b, R^2).
        """
        x = np.log10(taus)
        y = np.log10(sigmas)
        if x.size < 2:
            return np.nan, np.nan, np.nan
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(m), float(b), r2

    @staticmethod
    def _fit_k_tau_inv_sqrt(taus: np.ndarray, sigmas: np.ndarray) -> Optional[float]:
        """
        Solve least-squares for K in sigma ≈ K * tau^(-1/2).
        """
        if taus.size == 0 or sigmas.size == 0:
            return None
        w = taus ** (-1.5)
        den = float(np.sum(w * w))
        if den == 0.0:
            return None
        num = float(np.sum(sigmas * w))
        return num / den

    def _analyze_axis(self, dev: np.ndarray, fs: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # Welch PSD
        f_psd, Pxx = self._compute_welch(dev, fs) if fs else (None, None)

        # Allan
        taus, sig = self._compute_allan(dev, fs) if fs else (None, None)

        fit = {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "K_tau_inv_sqrt": np.nan
        }

        if taus is not None and sig is not None:
            n = taus.size
            k = max(self.min_fit_points, int(np.ceil(self.allan_fit_portion * n)))
            k = min(k, n)
            taus_fit = taus[:k]
            sig_fit = sig[:k]

            # Free slope fit
            m, b, r2 = self._linfit_loglog(taus_fit, sig_fit)
            fit["slope"], fit["intercept"], fit["r2"] = m, b, r2

            # Constrained tau^-1/2 overlay
            K = self._fit_k_tau_inv_sqrt(taus_fit, sig_fit)
            if K is not None:
                fit["K_tau_inv_sqrt"] = float(K)

        out["_psd_f"] = f_psd
        out["_psd_Pxx"] = Pxx
        out["_allan_taus"] = taus
        out["_allan_sigma"] = sig
        out.update(fit)
        return out

    # ---------------- Finish: plotting + console ----------------

    def finish(self) -> Optional[str]:
        if len(self._xs) < 2:
            print("NoiseBenchmark: not enough samples to analyze.")
            return None

        x = np.asarray(self._xs, dtype=float)
        y = np.asarray(self._ys, dtype=float)
        x_dev, _ = self._dc_remove(x, self.use_median_dc)
        y_dev, _ = self._dc_remove(y, self.use_median_dc)

        fs = float(self.fps_hint) if self.fps_hint else None
        self.results["X"] = self._analyze_axis(x_dev, fs) if fs else {}
        self.results["Y"] = self._analyze_axis(y_dev, fs) if fs else {}

        # ---------- Plot ----------
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
        fig.suptitle(self.title)

        # 1) Time trace
        ax = axes[0]
        idx = np.arange(x_dev.size)
        ax.plot(idx, x_dev, label="X dev [mm]")
        ax.plot(idx, y_dev, label="Y dev [mm]")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Deviation [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("DC-removed time trace (median-subtracted)")

        # 2) Welch PSD
        ax = axes[1]
        if fs and self.results["X"].get("_psd_f") is not None:
            for lab in ("X", "Y"):
                f = self.results[lab]["_psd_f"]
                Pxx = self.results[lab]["_psd_Pxx"]
                if f is None or Pxx is None:
                    continue
                # skip f=0 for log axes
                m = f > 0
                ax.loglog(f[m], Pxx[m], label=f"{lab} Welch PSD [mm²/Hz]")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [mm²/Hz]")
            ax.set_title("Welch PSD (Hann, density)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "PSD skipped (no fps_hint)", ha="center", va="center")
            ax.axis("off")

        # 3) Allan with fits
        ax = axes[2]
        plotted = False
        for lab in ("X", "Y"):
            taus = self.results.get(lab, {}).get("_allan_taus")
            sig = self.results.get(lab, {}).get("_allan_sigma")
            if taus is None or sig is None:
                continue
            ax.loglog(taus, sig, "o-", linewidth=1.5, markersize=4, label=f"{lab} OADEV [mm]")
            # Overlay free-slope fit over chosen region
            m = self.results[lab].get("slope")
            b = self.results[lab].get("intercept")
            r2 = self.results[lab].get("r2")
            if np.isfinite(m) and np.isfinite(b):
                k = max(self.min_fit_points, int(np.ceil(self.allan_fit_portion * taus.size)))
                k = min(k, taus.size)
                t_fit = taus[:k]
                y_fit = 10.0 ** (b + m * np.log10(t_fit))
                ax.loglog(t_fit, y_fit, "--", linewidth=2.0, label=f"{lab} fit: m={m:.3f}, R²={r2:.3f}")
            # Overlay constrained tau^-1/2
            K = self.results[lab].get("K_tau_inv_sqrt")
            if K is not None and np.isfinite(K):
                k = max(self.min_fit_points, int(np.ceil(self.allan_fit_portion * taus.size)))
                k = min(k, taus.size)
                t_fit = taus[:k]
                y_tau = K * (t_fit ** (-0.5))
                ax.loglog(t_fit, y_tau, ":", linewidth=1.8, label=f"{lab} ~ K·τ^(-1/2)")
            plotted = True

        if plotted:
            ax.set_xlabel("τ [s]")
            ax.set_ylabel("Allan deviation [mm]")
            kind = "MDEV" if self.allan_variant == "mdev" else "OADEV"
            ax.set_title(f"{kind} with log–log slope fit and τ^{{-1/2}} overlay")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Allan deviation skipped (need allantools + enough samples)", ha="center", va="center")
            ax.axis("off")

        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"NoiseBenchmark: saved plot to {self.out_path}")

        # ---------- Console summary ----------
        print("\n--- Allan slope (log–log) and τ^{-1/2} scale ---")
        for lab in ("X", "Y"):
            r = self.results.get(lab, {})
            m = r.get("slope")
            r2 = r.get("r2")
            K = r.get("K_tau_inv_sqrt")
            if m is None or np.isnan(m):
                print(f"{lab}: no Allan fit.")
                continue
            print(f"{lab}: slope m = {m:.3f} (ideal -0.5), R^2 = {r2:.3f}")
            if K is not None and np.isfinite(K):
                print(f"{lab}: fitted K for τ^(-1/2): {K:.6g} mm")

        return self.out_path
