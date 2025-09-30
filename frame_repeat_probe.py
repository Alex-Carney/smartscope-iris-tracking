# ASCII only
# smartscope_aruco/frame_repeat_probe.py
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import zlib
import cv2
import time


class FrameRepeatProbe:
    """
    Detects actual duplicate frames (identical JPEG bytes) and near-duplicates (tiny image delta).
    Records:
      - CRC32 of each JPEG (raw bytes)
      - Mean absolute difference (MAD) of downsampled grayscale vs previous

    Produces a 2-panel plot and prints summary stats and first duplicate point.
    """

    def __init__(
        self,
        out_path: str = "frame_repeat_probe.png",
        diff_downsample: Tuple[int, int] = (160, 120),
        diff_thresh_mm: float = 0.0,  # not used (we don't know mm/pixel here)
        diff_thresh_gray: float = 0.5 # MAD threshold in gray levels (0..255) to call "stale"
    ):
        self.out_path = out_path
        self.ds = diff_downsample
        self.diff_thresh = float(diff_thresh_gray)

        self.ts: List[float] = []
        self.crc: List[int] = []
        self.mad: List[float] = []

        self._prev_gray_small: Optional[np.ndarray] = None
        self._last_crc: Optional[int] = None

        # duplicate bands by CRC
        self.crc_bands: List[Tuple[int, int]] = []
        self._run_start: Optional[int] = None
        self._run_len: int = 0

    def add_jpeg(self, t_now: float, jpeg_bytes: bytes):
        c = zlib.crc32(jpeg_bytes)
        self.ts.append(float(t_now))
        self.crc.append(int(c))

        # duplicate-run bookkeeping
        i = len(self.crc) - 1
        if self._last_crc is not None and c == self._last_crc:
            if self._run_start is None:
                self._run_start = i - 1
                self._run_len = 2
            else:
                self._run_len += 1
        else:
            if self._run_start is not None and self._run_len >= 2:
                self.crc_bands.append((self._run_start, i - 1))
            self._run_start = None
            self._run_len = 0
        self._last_crc = c

    def add_gray(self, gray: np.ndarray):
        # fast downsample + MAD to previous
        small = cv2.resize(gray, self.ds, interpolation=cv2.INTER_AREA)
        if self._prev_gray_small is None:
            self.mad.append(np.nan)
        else:
            diff = np.abs(small.astype(np.int16) - self._prev_gray_small.astype(np.int16))
            self.mad.append(float(np.mean(diff)))
        self._prev_gray_small = small

    def finish(self) -> Optional[str]:
        n = len(self.ts)
        if n < 2:
            print("FrameRepeatProbe: not enough frames.")
            return None
        if self._run_start is not None and self._run_len >= 2:
            self.crc_bands.append((self._run_start, n - 1))

        t = np.asarray(self.ts, float)
        dt = np.diff(t)
        fps = np.where(dt > 0, 1.0 / dt, np.nan)
        idx = np.arange(1, n)

        mad = np.asarray(self.mad, float)
        mad_idx = np.arange(mad.size)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

        # Panel 1: instantaneous FPS (all frames), shade CRC duplicate bands
        ax1 = axes[0]
        ax1.plot(idx, fps, linewidth=1.0, label="Instant FPS")
        ax1.hlines(np.nanmedian(fps), idx[0], idx[-1], linestyles="--", label="Median")
        for a, b in self.crc_bands:
            ax1.axvspan(a, b, color="tab:red", alpha=0.15, zorder=0)
        ax1.set_xlabel("Frame index")
        ax1.set_ylabel("FPS [1/s]")
        ax1.set_title("Instantaneous FPS with identical-JPEG bands (red)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)

        # Panel 2: MAD of grayscale (downsampled), near-zero => stale content
        ax2 = axes[1]
        ax2.semilogy(mad_idx, np.clip(mad, 1e-6, None), linewidth=1.0, label="MAD(gray)")
        ax2.hlines(self.diff_thresh, mad_idx[0], mad_idx[-1], linestyles="--", label=f"Threshold {self.diff_thresh:.2f}")
        for a, b in self.crc_bands:
            ax2.axvspan(a, b, color="tab:red", alpha=0.15, zorder=0)
        ax2.set_xlabel("Frame index")
        ax2.set_ylabel("Mean abs diff (gray levels)")
        ax2.set_title("Frame-to-frame MAD (downsampled)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"FrameRepeatProbe: saved plot to {self.out_path}")

        # Console summary
        dup_crc = 0
        for a, b in self.crc_bands:
            dup_crc += (b - a)  # counts repeats beyond first in the band
        frac = 100.0 * dup_crc / max(1, n)
        first_dup = self.crc_bands[0][0] if self.crc_bands else -1
        print("\n--- FrameRepeatProbe summary ---")
        print(f"Frames: {n}, identical-JPEG repeats (beyond first): {dup_crc} ({frac:.2f}%)")
        if first_dup >= 0:
            print(f"First identical-JPEG run begins at frame {first_dup}")
        nz = mad[1:][~np.isnan(mad[1:])]
        if nz.size:
            print(f"MAD(gray) median={np.median(nz):.3f}, 1st pct={np.percentile(nz,1):.3f}")
        return self.out_path
