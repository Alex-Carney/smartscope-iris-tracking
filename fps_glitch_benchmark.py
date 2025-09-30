# ASCII only
# smartscope_aruco/fps_glitch_benchmark.py
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math


class FPSGlitchBenchmark:
    """
    Tracks per-frame timestamps and positions to diagnose duplicate-frame glitches.

    - "Camera FPS": 1 / dt between *all* detected marker frames.
    - "Effective FPS": only advances when (x, y) changes; repeats are treated as no-frame.

    Duplicate detection:
      mode="exact": (x_mm, y_mm) must match exactly
      mode="radial": sqrt(dx^2 + dy^2) <= tol_mm

    Produces a 2-panel figure and prints a summary.
    """

    def __init__(
        self,
        out_path: str = "fps_glitch_benchmark.png",
        mode: str = "exact",              # "exact" or "radial"
        tol_mm: float = 1e-8,             # used only if mode == "radial"
        min_run_to_mark: int = 2          # shade glitch bands of >= this length
    ):
        self.out_path = out_path
        self.mode = mode
        self.tol_mm = float(tol_mm)
        self.min_run_to_mark = int(min_run_to_mark)

        # Per-frame series
        self.ts: List[float] = []         # perf_counter timestamps for frames with a marker
        self.xs: List[float] = []
        self.ys: List[float] = []

        # Effective series (unique-only)
        self.unique_ts: List[float] = []
        self.unique_idx: List[int] = []   # index into full sequence where a unique update occurred

        # Bookkeeping for runs of duplicates
        self._prev_xy: Optional[Tuple[float, float]] = None
        self._prev_unique_ts: Optional[float] = None
        self._current_run_start: Optional[int] = None
        self._current_run_len: int = 0
        self.glitch_bands: List[Tuple[int, int]] = []  # (start_idx, end_idx) inclusive

    # ------------------------ feed data ------------------------

    def _is_duplicate(self, x: float, y: float) -> bool:
        if self._prev_xy is None:
            return False
        if self.mode == "exact":
            return (x == self._prev_xy[0]) and (y == self._prev_xy[1])
        # radial tolerance
        dx = x - self._prev_xy[0]
        dy = y - self._prev_xy[1]
        return (dx * dx + dy * dy) <= (self.tol_mm * self.tol_mm)

    def add(self, t_now: float, x_mm: float, y_mm: float) -> None:
        """
        Call once per detected marker frame (BEFORE noise gating).
        """
        self.ts.append(float(t_now))
        self.xs.append(float(x_mm))
        self.ys.append(float(y_mm))

        dup = self._is_duplicate(x_mm, y_mm)

        # Maintain duplicate run bands
        idx = len(self.ts) - 1
        if dup:
            if self._current_run_start is None:
                self._current_run_start = idx - 1  # include the first equal pair's predecessor
                self._current_run_len = 2
            else:
                self._current_run_len += 1
        else:
            if self._current_run_start is not None and self._current_run_len >= self.min_run_to_mark:
                self.glitch_bands.append((self._current_run_start, idx - 1))
            self._current_run_start = None
            self._current_run_len = 0

        # Manage effective (unique) timestamps
        if (not dup) or (len(self.unique_ts) == 0):
            self.unique_ts.append(float(t_now))
            self.unique_idx.append(idx)
            self._prev_unique_ts = float(t_now)

        # Update previous xy
        self._prev_xy = (float(x_mm), float(y_mm))

    # ------------------------ finalize / plotting ------------------------

    def finish(self) -> Optional[str]:
        n = len(self.ts)
        if n < 2:
            print("FPSGlitchBenchmark: not enough frames.")
            return None

        # Close any open duplicate band
        if self._current_run_start is not None and self._current_run_len >= self.min_run_to_mark:
            self.glitch_bands.append((self._current_run_start, n - 1))

        t = np.asarray(self.ts, dtype=float)
        # Camera FPS (instantaneous between all frames)
        dt = np.diff(t)
        cam_fps = np.where(dt > 0, 1.0 / dt, np.nan)
        cam_idx = np.arange(1, n)

        # Effective FPS (between unique frames only)
        ut = np.asarray(self.unique_ts, dtype=float)
        uidx = np.asarray(self.unique_idx, dtype=int)
        if ut.size >= 2:
            udt = np.diff(ut)
            eff_fps = np.where(udt > 0, 1.0 / udt, np.nan)
            eff_idx = uidx[1:]
        else:
            eff_fps = np.array([], dtype=float)
            eff_idx = np.array([], dtype=int)

        # ---- Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

        # Panel 1: Camera FPS
        ax1 = axes[0]
        ax1.plot(cam_idx, cam_fps, linewidth=1.0, label="Camera FPS (all frames)")
        if cam_fps.size > 0:
            ax1.hlines(np.nanmedian(cam_fps), cam_idx[0], cam_idx[-1], linestyles="--", label="Median")
        ax1.set_xlabel("Frame index")
        ax1.set_ylabel("FPS [1/s]")
        ax1.set_title("Camera FPS vs. frame index")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)

        # Shade detected glitch bands
        for a, b in self.glitch_bands:
            ax1.axvspan(a, b, color="tab:red", alpha=0.12, zorder=0)

        # Panel 2: Effective FPS
        ax2 = axes[1]
        if eff_fps.size > 0:
            ax2.plot(eff_idx, eff_fps, linewidth=1.2, label="Effective FPS (unique updates)")
            ax2.hlines(np.nanmedian(eff_fps), eff_idx[0], eff_idx[-1], linestyles="--", label="Median")
        else:
            ax2.text(0.5, 0.5, "No unique updates found", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("Frame index (of accepted unique update)")
        ax2.set_ylabel("FPS [1/s]")
        ax2.set_title("Effective FPS (duplicates ignored)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

        # Same glitch shading on panel 2 for visual alignment
        for a, b in self.glitch_bands:
            ax2.axvspan(a, b, color="tab:red", alpha=0.12, zorder=0)

        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"FPSGlitchBenchmark: saved plot to {self.out_path}")

        # ---- Console summary
        dup_flags = self._duplicate_flags()
        n_dup = int(np.sum(dup_flags))
        first_glitch_idx = int(np.argmax(dup_flags)) + 1 if n_dup > 0 else -1
        t0 = t[0]
        t_first = t[first_glitch_idx] - t0 if first_glitch_idx >= 1 else None

        print("\n--- FPS Glitch Summary ---")
        print(f"Total frames: {n}")
        print(f"Duplicate frames: {n_dup} ({(100.0*n_dup/max(1,n)):.2f}%)")
        if first_glitch_idx >= 1:
            print(f"First duplicate at index {first_glitch_idx} (t = {t_first:.3f} s from start)")
        else:
            print("No duplicates detected.")
        if self.glitch_bands:
            worst = max((b - a + 1) for a, b in self.glitch_bands)
            print(f"Glitch bands (>= {self.min_run_to_mark} in a row): {len(self.glitch_bands)}, worst run = {worst} frames")
        if cam_fps.size > 0:
            print(f"Camera FPS: median={np.nanmedian(cam_fps):.2f}, mean={np.nanmean(cam_fps):.2f}")
        if eff_fps.size > 0:
            print(f"Effective FPS: median={np.nanmedian(eff_fps):.2f}, mean={np.nanmean(eff_fps):.2f}")
        else:
            print("Effective FPS: N/A (no unique transitions)")

        return self.out_path

    def _duplicate_flags(self) -> np.ndarray:
        """Boolean array of length N-1; True if frame i equals i-1."""
        n = len(self.xs)
        if n < 2:
            return np.zeros(0, dtype=bool)
        flags = np.zeros(n - 1, dtype=bool)
        for i in range(1, n):
            if self.mode == "exact":
                flags[i - 1] = (self.xs[i] == self.xs[i - 1]) and (self.ys[i] == self.ys[i - 1])
            else:
                dx = self.xs[i] - self.xs[i - 1]
                dy = self.ys[i] - self.ys[i - 1]
                flags[i - 1] = (dx * dx + dy * dy) <= (self.tol_mm * self.tol_mm)
        return flags
