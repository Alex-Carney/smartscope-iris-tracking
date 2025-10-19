# ASCII only
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, get_window

from config import AppConfig
from ffmpeg_stream import FFMPEGMJPEGStream
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker, MarkerMeasurement, TRACKED_MARKERS
from time_accounting import TimeAccounting
from basic_benchmark import BasicBenchmark


# ---- Tunables ----
SKIP_FIRST_N_DETECTIONS = 200
NOISE_SIGMA_MULTIPLIER = 2.5
TRACE_PLOT_PATH = "static_multi_traces.png"
FRAME_ACCOUNTING_PATH = "frame_accounting_multi.png"
COMMON_MODE_PLOT_PATH = "dynamic_cmr_compare.png"
COMMON_MODE_THRESHOLD_MM_PER_S = 1.0


@dataclass
class MarkerSeries:
    times_s: List[float] = field(default_factory=list)
    time_keys: List[int] = field(default_factory=list)
    xs_mm: List[float] = field(default_factory=list)
    ys_mm: List[float] = field(default_factory=list)
    _index_map: Dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def append(self, t_s: float, center_mm: Tuple[float, float]) -> None:
        t_val = float(t_s)
        key = int(round(t_val * 1_000_000))
        self.times_s.append(t_val)
        self.time_keys.append(key)
        self.xs_mm.append(float(center_mm[0]))
        self.ys_mm.append(float(center_mm[1]))
        self._index_map[key] = len(self.times_s) - 1

    def _sync_index_map(self) -> None:
        if len(self._index_map) != len(self.time_keys):
            self._index_map = {key: idx for idx, key in enumerate(self.time_keys)}

    def index_for_key(self, key: int) -> int | None:
        self._sync_index_map()
        return self._index_map.get(key)

    def velocities(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(self.times_s, dtype=np.float64)
        xs = np.asarray(self.xs_mm, dtype=np.float64)
        ys = np.asarray(self.ys_mm, dtype=np.float64)
        n = times.size
        vx = np.zeros(n, dtype=np.float64)
        vy = np.zeros(n, dtype=np.float64)
        if n < 2:
            return times, vx, vy
        dt = np.diff(times)
        dx = np.diff(xs)
        dy = np.diff(ys)
        valid = dt > 0
        vx_vals = np.zeros_like(dx)
        vy_vals = np.zeros_like(dy)
        vx_vals[valid] = dx[valid] / dt[valid]
        vy_vals[valid] = dy[valid] / dt[valid]
        vx[1:] = vx_vals
        vy[1:] = vy_vals
        return times, vx, vy


@dataclass
class PairwiseStats:
    timestamps_s: List[float] = field(default_factory=list)
    main_x_mm: List[float] = field(default_factory=list)
    main_y_mm: List[float] = field(default_factory=list)
    static_x_mm: List[float] = field(default_factory=list)
    static_y_mm: List[float] = field(default_factory=list)

    def add(
        self,
        t_s: float,
        main_center_mm: Tuple[float, float],
        static_center_mm: Tuple[float, float],
    ) -> None:
        self.timestamps_s.append(float(t_s))
        self.main_x_mm.append(float(main_center_mm[0]))
        self.main_y_mm.append(float(main_center_mm[1]))
        self.static_x_mm.append(float(static_center_mm[0]))
        self.static_y_mm.append(float(static_center_mm[1]))

    def _as_arrays(self):
        main_x = np.asarray(self.main_x_mm, dtype=np.float64)
        main_y = np.asarray(self.main_y_mm, dtype=np.float64)
        static_x = np.asarray(self.static_x_mm, dtype=np.float64)
        static_y = np.asarray(self.static_y_mm, dtype=np.float64)
        return main_x, main_y, static_x, static_y

    def correlation_summary(self, noise_sigma: float) -> Dict[str, float]:
        summary: Dict[str, float] = {"samples": float(len(self.timestamps_s))}
        if len(self.timestamps_s) < 3:
            summary.update({
                "overall_corr_x": float("nan"),
                "overall_corr_y": float("nan"),
                "noise_corr_x": float("nan"),
                "noise_corr_y": float("nan"),
                "large_corr_x": float("nan"),
                "large_corr_y": float("nan"),
                "noise_fraction": float("nan"),
                "large_fraction": float("nan"),
            })
            return summary

        main_x, main_y, static_x, static_y = self._as_arrays()
        main_x_c = main_x - np.median(main_x)
        main_y_c = main_y - np.median(main_y)
        static_x_c = static_x - np.median(static_x)
        static_y_c = static_y - np.median(static_y)

        thr_main_x = _robust_threshold(main_x_c, noise_sigma)
        thr_main_y = _robust_threshold(main_y_c, noise_sigma)
        thr_static_x = _robust_threshold(static_x_c, noise_sigma)
        thr_static_y = _robust_threshold(static_y_c, noise_sigma)

        noise_mask = (
            (np.abs(main_x_c) <= thr_main_x) &
            (np.abs(static_x_c) <= thr_static_x) &
            (np.abs(main_y_c) <= thr_main_y) &
            (np.abs(static_y_c) <= thr_static_y)
        )
        large_mask = ~noise_mask

        summary.update({
            "noise_threshold_main_x": float(thr_main_x),
            "noise_threshold_main_y": float(thr_main_y),
            "noise_threshold_static_x": float(thr_static_x),
            "noise_threshold_static_y": float(thr_static_y),
            "noise_fraction": float(np.mean(noise_mask)) if noise_mask.size else float("nan"),
            "large_fraction": float(np.mean(large_mask)) if large_mask.size else float("nan"),
            "overall_corr_x": _safe_corr(main_x_c, static_x_c),
            "overall_corr_y": _safe_corr(main_y_c, static_y_c),
            "noise_corr_x": _safe_corr_masked(main_x_c, static_x_c, noise_mask),
            "noise_corr_y": _safe_corr_masked(main_y_c, static_y_c, noise_mask),
            "large_corr_x": _safe_corr_masked(main_x_c, static_x_c, large_mask),
            "large_corr_y": _safe_corr_masked(main_y_c, static_y_c, large_mask),
        })
        return summary


class MultiMarkerRecorder:
    def __init__(self, main_id: int):
        self.main_id = int(main_id)
        self.first_ts: float | None = None
        self.marker_series: Dict[int, MarkerSeries] = {}
        self.pairwise: Dict[int, PairwiseStats] = {}
        self.samples: List[Tuple[float, int, Dict[int, Tuple[float, float]]]] = []

    def add(self, timestamp: float, detections: Dict[int, MarkerMeasurement]) -> None:
        if not detections:
            return
        if self.first_ts is None:
            self.first_ts = float(timestamp)
        t_rel = float(timestamp - self.first_ts)
        key = int(round(t_rel * 1_000_000))

        for marker_id, measurement in detections.items():
            center_mm = measurement[0]
            marker_key = int(marker_id)
            series = self.marker_series.setdefault(marker_key, MarkerSeries())
            series.append(t_rel, center_mm)

        if self.main_id not in detections:
            return

        main_center = detections[self.main_id][0]
        for marker_id, measurement in detections.items():
            if marker_id == self.main_id:
                continue
            pair = self.pairwise.setdefault(int(marker_id), PairwiseStats())
            pair.add(t_rel, main_center, measurement[0])

        sample_positions = {
            int(marker_id): (float(measurement[0][0]), float(measurement[0][1]))
            for marker_id, measurement in detections.items()
        }
        self.samples.append((t_rel, key, sample_positions))

    def compute_common_mode_result(self, threshold_mm_per_s: float) -> "CommonModeResult | None":
        dynamic_series = self.marker_series.get(self.main_id)
        if dynamic_series is None or not dynamic_series.times_s:
            return None

        times_dyn = np.asarray(dynamic_series.times_s, dtype=np.float64)
        _, dyn_vx, dyn_vy = dynamic_series.velocities()

        static_ids = [
            marker_id for marker_id, role in TRACKED_MARKERS.items()
            if marker_id != self.main_id and str(role).lower() != "dynamic"
        ]
        static_cache: Dict[int, Tuple[MarkerSeries, np.ndarray, np.ndarray]] = {}
        for marker_id in static_ids:
            series = self.marker_series.get(marker_id)
            if series is None or not series.times_s:
                continue
            _, vx, vy = series.velocities()
            static_cache[marker_id] = (series, vx, vy)

        if not self.samples:
            return None

        times_out: List[float] = []
        raw_x: List[float] = []
        raw_y: List[float] = []
        cmr_x: List[float] = []
        cmr_y: List[float] = []
        raw_vx_list: List[float] = []
        raw_vy_list: List[float] = []
        cmr_vx_list: List[float] = []
        cmr_vy_list: List[float] = []

        for t_rel, key, positions in self.samples:
            if self.main_id not in positions:
                continue
            dyn_idx = dynamic_series.index_for_key(key)
            if dyn_idx is None or dyn_idx >= dyn_vx.size:
                continue

            dyn_pos = positions[self.main_id]
            dyn_vx_val = float(dyn_vx[dyn_idx])
            dyn_vy_val = float(dyn_vy[dyn_idx])

            static_vxs: List[float] = []
            static_vys: List[float] = []
            for marker_id, (series, vx_arr, vy_arr) in static_cache.items():
                if marker_id not in positions:
                    continue
                s_idx = series.index_for_key(key)
                if s_idx is None or s_idx >= vx_arr.size:
                    continue
                static_vxs.append(float(vx_arr[s_idx]))
                static_vys.append(float(vy_arr[s_idx]))

            if static_vxs:
                mean_vx = float(np.mean(static_vxs))
                mean_vy = float(np.mean(static_vys))
                speed = float(np.hypot(mean_vx, mean_vy))
                if speed < threshold_mm_per_s:
                    mean_vx = 0.0
                    mean_vy = 0.0
            else:
                mean_vx = 0.0
                mean_vy = 0.0

            corrected_vx = dyn_vx_val - mean_vx
            corrected_vy = dyn_vy_val - mean_vy

            times_out.append(t_rel)
            raw_x.append(float(dyn_pos[0]))
            raw_y.append(float(dyn_pos[1]))
            raw_vx_list.append(dyn_vx_val)
            raw_vy_list.append(dyn_vy_val)
            cmr_vx_list.append(corrected_vx)
            cmr_vy_list.append(corrected_vy)

            if not cmr_x:
                cmr_x.append(float(dyn_pos[0]))
                cmr_y.append(float(dyn_pos[1]))
            else:
                dt = max(t_rel - times_out[-2], 0.0)
                cmr_x.append(cmr_x[-1] + corrected_vx * dt)
                cmr_y.append(cmr_y[-1] + corrected_vy * dt)

        if not times_out:
            return None

        return CommonModeResult(
            times_s=np.asarray(times_out, dtype=np.float64),
            raw_x_mm=np.asarray(raw_x, dtype=np.float64),
            raw_y_mm=np.asarray(raw_y, dtype=np.float64),
            cmr_x_mm=np.asarray(cmr_x, dtype=np.float64),
            cmr_y_mm=np.asarray(cmr_y, dtype=np.float64),
            raw_vx_mms=np.asarray(raw_vx_list, dtype=np.float64),
            raw_vy_mms=np.asarray(raw_vy_list, dtype=np.float64),
            cmr_vx_mms=np.asarray(cmr_vx_list, dtype=np.float64),
            cmr_vy_mms=np.asarray(cmr_vy_list, dtype=np.float64),
            threshold_mm_per_s=float(threshold_mm_per_s),
        )

    def sample_counts(self) -> Dict[int, int]:
        return {marker_id: len(series.times_s) for marker_id, series in self.marker_series.items()}

    def pair_counts(self) -> Dict[int, int]:
        return {marker_id: len(stats.timestamps_s) for marker_id, stats in self.pairwise.items()}

    def compute_correlations(self, noise_sigma: float) -> Dict[int, Dict[str, float]]:
        return {
            marker_id: stats.correlation_summary(noise_sigma)
            for marker_id, stats in self.pairwise.items()
        }

    def save_overlay_plot(self, out_path: str) -> str | None:
        if not self.marker_series:
            print("No marker data collected; skipping overlay plot.")
            return None

        fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
        ax_x_time, ax_y_time = axes[0, 0], axes[0, 1]
        ax_x_vel, ax_y_vel = axes[1, 0], axes[1, 1]
        ax_x_fft, ax_y_fft = axes[2, 0], axes[2, 1]

        sorted_ids = sorted(self.marker_series.keys())
        cmap = plt.get_cmap("tab10", max(1, len(sorted_ids)))

        for idx, marker_id in enumerate(sorted_ids):
            series = self.marker_series[marker_id]
            if not series.times_s:
                continue

            times = np.asarray(series.times_s, dtype=np.float64)
            xs = np.asarray(series.xs_mm, dtype=np.float64)
            ys = np.asarray(series.ys_mm, dtype=np.float64)
            xs_centered = xs - float(np.median(xs))
            ys_centered = ys - float(np.median(ys))
            _, vx, vy = series.velocities()

            color = cmap(idx)
            label = f"ID {marker_id}"

            ax_x_time.plot(times, xs_centered, label=label, color=color, linewidth=0.9)
            ax_y_time.plot(times, ys_centered, label=label, color=color, linewidth=0.9)
            ax_x_vel.plot(times, vx, label=label, color=color, linewidth=0.9)
            ax_y_vel.plot(times, vy, label=label, color=color, linewidth=0.9)

            fx, Px = _welch_psd(times, xs_centered)
            if fx is not None and Px is not None:
                mask = fx > 0
                if np.any(mask):
                    ax_x_fft.loglog(fx[mask], Px[mask], label=label, color=color, linewidth=1.0)

            fy, Py = _welch_psd(times, ys_centered)
            if fy is not None and Py is not None:
                mask = fy > 0
                if np.any(mask):
                    ax_y_fft.loglog(fy[mask], Py[mask], label=label, color=color, linewidth=1.0)

        ax_x_time.set_ylabel("X center [mm]")
        ax_y_time.set_ylabel("Y center [mm]")
        ax_x_time.set_xlabel("Time [s]")
        ax_y_time.set_xlabel("Time [s]")
        ax_x_time.set_title("X positions vs time (median-centered)")
        ax_y_time.set_title("Y positions vs time (median-centered)")
        ax_x_time.grid(True, alpha=0.3)
        ax_y_time.grid(True, alpha=0.3)
        ax_x_time.legend(loc="upper right", fontsize=9, ncol=2)
        ax_y_time.legend(loc="upper right", fontsize=9, ncol=2)

        ax_x_vel.set_title("X velocity vs time")
        ax_y_vel.set_title("Y velocity vs time")
        ax_x_vel.set_xlabel("Time [s]")
        ax_y_vel.set_xlabel("Time [s]")
        ax_x_vel.set_ylabel("Velocity [mm/s]")
        ax_y_vel.set_ylabel("Velocity [mm/s]")
        ax_x_vel.grid(True, alpha=0.3)
        ax_y_vel.grid(True, alpha=0.3)
        ax_x_vel.legend(loc="upper right", fontsize=9, ncol=2)
        ax_y_vel.legend(loc="upper right", fontsize=9, ncol=2)

        ax_x_fft.set_title("X Welch PSD (Hann window)")
        ax_y_fft.set_title("Y Welch PSD (Hann window)")
        ax_x_fft.set_xlabel("Frequency [Hz]")
        ax_y_fft.set_xlabel("Frequency [Hz]")
        ax_x_fft.set_ylabel("PSD [mm^2/Hz]")
        ax_y_fft.set_ylabel("PSD [mm^2/Hz]")
        ax_x_fft.grid(True, which="both", alpha=0.3)
        ax_y_fft.grid(True, which="both", alpha=0.3)
        ax_x_fft.legend(loc="upper right", fontsize=9, ncol=2)
        ax_y_fft.legend(loc="upper right", fontsize=9, ncol=2)

        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved overlay plot to {out_path}")
        return out_path

@dataclass
class CommonModeResult:
    times_s: np.ndarray
    raw_x_mm: np.ndarray
    raw_y_mm: np.ndarray
    cmr_x_mm: np.ndarray
    cmr_y_mm: np.ndarray
    raw_vx_mms: np.ndarray
    raw_vy_mms: np.ndarray
    cmr_vx_mms: np.ndarray
    cmr_vy_mms: np.ndarray
    threshold_mm_per_s: float


def _robust_threshold(centered: np.ndarray, sigma: float) -> float:
    if centered.size == 0:
        return float(0.0)
    mad = np.median(np.abs(centered - np.median(centered)))
    if mad > 0:
        return float(sigma * 1.4826 * mad)
    std = float(np.std(centered))
    return float(sigma * std)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_corr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    if mask.sum() < 3:
        return float("nan")
    return _safe_corr(a[mask], b[mask])


def _estimate_fps(times: np.ndarray) -> float | None:
    if times.size < 2:
        return None
    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    median_dt = float(np.median(dt))
    if median_dt <= 0.0 or not np.isfinite(median_dt):
        return None
    return 1.0 / median_dt


def _welch_psd(times: np.ndarray, deviations: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if deviations.size < 32:
        return None, None
    fps = _estimate_fps(times)
    if fps is None:
        return None, None

    n = deviations.size
    nperseg = min(max(256, 2 ** int(np.floor(np.log2(max(1, n // 4))))), n)
    if nperseg <= 16:
        return None, None

    window = get_window("hann", nperseg)
    f, Pxx = welch(
        deviations,
        fs=fps,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, Pxx


def save_common_mode_plot(result: CommonModeResult, out_path: str) -> str | None:
    if result is None or result.times_s.size < 2:
        print("Common-mode plot skipped (insufficient samples).")
        return None

    times = result.times_s
    raw_x = result.raw_x_mm
    raw_y = result.raw_y_mm
    cmr_x = result.cmr_x_mm
    cmr_y = result.cmr_y_mm

    raw_x_c = raw_x - np.median(raw_x)
    raw_y_c = raw_y - np.median(raw_y)
    cmr_x_c = cmr_x - np.median(cmr_x)
    cmr_y_c = cmr_y - np.median(cmr_y)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_xt, ax_yt = axes[0, 0], axes[0, 1]
    ax_xf, ax_yf = axes[1, 0], axes[1, 1]

    ax_xt.plot(times, raw_x_c, label="Raw centered X", linewidth=0.9, color="tab:blue")
    ax_xt.plot(times, cmr_x_c, label="CMR centered X", linewidth=0.9, color="tab:orange")
    ax_xt.set_title("Dynamic marker X (median-centered)")
    ax_xt.set_xlabel("Time [s]")
    ax_xt.set_ylabel("Deviation [mm]")
    ax_xt.grid(True, alpha=0.3)
    ax_xt.legend(loc="upper right", fontsize=9)

    ax_yt.plot(times, raw_y_c, label="Raw centered Y", linewidth=0.9, color="tab:blue")
    ax_yt.plot(times, cmr_y_c, label="CMR centered Y", linewidth=0.9, color="tab:orange")
    ax_yt.set_title("Dynamic marker Y (median-centered)")
    ax_yt.set_xlabel("Time [s]")
    ax_yt.set_ylabel("Deviation [mm]")
    ax_yt.grid(True, alpha=0.3)
    ax_yt.legend(loc="upper right", fontsize=9)

    fx_raw, Px_raw = _welch_psd(times, raw_x_c)
    fx_cmr, Px_cmr = _welch_psd(times, cmr_x_c)
    if fx_raw is not None and Px_raw is not None and np.any(fx_raw > 0):
        mask = fx_raw > 0
        ax_xf.loglog(fx_raw[mask], Px_raw[mask], label="Raw X", color="tab:blue")
    if fx_cmr is not None and Px_cmr is not None and np.any(fx_cmr > 0):
        mask = fx_cmr > 0
        ax_xf.loglog(fx_cmr[mask], Px_cmr[mask], label="CMR X", color="tab:orange")
    ax_xf.set_title("Welch PSD X (Hann window)")
    ax_xf.set_xlabel("Frequency [Hz]")
    ax_xf.set_ylabel("PSD [mm^2/Hz]")
    ax_xf.grid(True, which="both", alpha=0.3)
    ax_xf.legend(loc="upper right", fontsize=9)

    fy_raw, Py_raw = _welch_psd(times, raw_y_c)
    fy_cmr, Py_cmr = _welch_psd(times, cmr_y_c)
    if fy_raw is not None and Py_raw is not None and np.any(fy_raw > 0):
        mask = fy_raw > 0
        ax_yf.loglog(fy_raw[mask], Py_raw[mask], label="Raw Y", color="tab:blue")
    if fy_cmr is not None and Py_cmr is not None and np.any(fy_cmr > 0):
        mask = fy_cmr > 0
        ax_yf.loglog(fy_cmr[mask], Py_cmr[mask], label="CMR Y", color="tab:orange")
    ax_yf.set_title("Welch PSD Y (Hann window)")
    ax_yf.set_xlabel("Frequency [Hz]")
    ax_yf.set_ylabel("PSD [mm^2/Hz]")
    ax_yf.grid(True, which="both", alpha=0.3)
    ax_yf.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Dynamic vs Common-Mode Rejected (threshold = {result.threshold_mm_per_s:.3f} mm/s)"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved common-mode comparison plot to {out_path}")
    return out_path


async def run(app: AppConfig):
    cam = app.camera
    und = app.undistort
    arc = app.aruco
    jpg = app.jpeg
    run_cfg = app.run

    stream = FFMPEGMJPEGStream(cam.device_name, cam.width, cam.height, cam.fps)
    decoder = JPEGDecoder(jpg.libjpeg_turbo_path)
    timer = TimeAccounting(
        out_path=FRAME_ACCOUNTING_PATH,
        title="Frame Accounting (multi-marker static correlation)",
    )
    tracker = ArucoTracker(
        arc.dictionary,
        arc.aruco_id,
        arc.aruco_w_mm,
        arc.aruco_h_mm,
        frame_size_px=(cam.width, cam.height),
        isotropic_scale=False,
        time_accounting=timer,
        roi_first=True,
    )

    K, D = und.as_np()
    undistorter = Undistorter(K, D, (cam.width, cam.height))

    bench = BasicBenchmark()
    recorder = MultiMarkerRecorder(main_id=arc.aruco_id)

    await stream.start()
    first_frame_saved = False
    skipped = 0
    samples = 0
    announced_skip = False

    try:
        while samples < run_cfg.max_samples:
            jpg_bytes = await stream.read_jpeg()
            if jpg_bytes is None:
                print("[WARN] stream.read_jpeg() returned None; stopping.")
                break

            timestamp = time.perf_counter()
            timer.start_frame()

            frame = decoder.decode_bgr(jpg_bytes)
            timer.mark("jpeg_decode")

            if und.enable_frame_undistort:
                frame = undistorter.remap(frame)
                timer.mark("undistort frame")

            if not first_frame_saved and run_cfg.save_first_frame:
                cv2.imwrite(run_cfg.save_first_frame_path, frame)
                print(f"Saved first frame to {run_cfg.save_first_frame_path}")
                first_frame_saved = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timer.mark("to_gray")

            undistort_points_fn = undistorter.undistort_points if und.enable_corner_undistort else None
            detections = tracker.detect_multiple_mm(gray, undistort_points_fn, timer=timer)

            if arc.aruco_id not in detections:
                timer.mark("multi:main_missing")
                timer.end_frame()
                continue

            if skipped < SKIP_FIRST_N_DETECTIONS:
                skipped += 1
                if not announced_skip:
                    print(f"[INFO] Dropping first {SKIP_FIRST_N_DETECTIONS} detections to bypass transient.")
                    announced_skip = True
                if skipped % 50 == 0 or skipped == SKIP_FIRST_N_DETECTIONS:
                    print(f"[INFO] Dropped {skipped}/{SKIP_FIRST_N_DETECTIONS}")
                timer.mark("multi:skip")
                timer.end_frame()
                continue

            bench.mark_processed()
            bench.mark_with_marker()
            bench.tick_fps()

            center_mm = detections[arc.aruco_id][0]
            bench.add_position(center_mm[0], center_mm[1])
            recorder.add(timestamp, detections)

            samples += 1
            timer.mark("multi:record")
            timer.end_frame()

        print("Multi-marker benchmark loop ended.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        await stream.stop()
        timer.finish()
        recorder.save_overlay_plot(TRACE_PLOT_PATH)

        cmr_result = recorder.compute_common_mode_result(COMMON_MODE_THRESHOLD_MM_PER_S)
        if cmr_result is not None:
            save_common_mode_plot(cmr_result, COMMON_MODE_PLOT_PATH)
            raw_x_std = float(np.std(cmr_result.raw_x_mm - np.median(cmr_result.raw_x_mm)))
            raw_y_std = float(np.std(cmr_result.raw_y_mm - np.median(cmr_result.raw_y_mm)))
            cmr_x_std = float(np.std(cmr_result.cmr_x_mm - np.median(cmr_result.cmr_x_mm)))
            cmr_y_std = float(np.std(cmr_result.cmr_y_mm - np.median(cmr_result.cmr_y_mm)))
            raw_v_rms = float(np.sqrt(np.mean(cmr_result.raw_vx_mms**2 + cmr_result.raw_vy_mms**2)))
            cmr_v_rms = float(np.sqrt(np.mean(cmr_result.cmr_vx_mms**2 + cmr_result.cmr_vy_mms**2)))
            print("\n--- Common-mode rejection summary ---")
            print(f"Threshold: {cmr_result.threshold_mm_per_s:.3f} mm/s")
            print(f"Std dev (raw vs CMR) X: {raw_x_std:.6f} -> {cmr_x_std:.6f} mm")
            print(f"Std dev (raw vs CMR) Y: {raw_y_std:.6f} -> {cmr_y_std:.6f} mm")
            print(f"Velocity RMS (raw vs CMR): {raw_v_rms:.6f} -> {cmr_v_rms:.6f} mm/s")
        else:
            print("\nCommon-mode rejection summary unavailable (insufficient overlapping data).")

        correlations = recorder.compute_correlations(NOISE_SIGMA_MULTIPLIER)
        counts = recorder.sample_counts()
        pair_counts = recorder.pair_counts()
        bench.print_summary("(multi-marker main)")

        if counts:
            print("\n--- Marker detection counts ---")
            for marker_id in sorted(counts.keys()):
                total = counts[marker_id]
                paired = pair_counts.get(marker_id, 0)
                suffix = " (main)" if marker_id == arc.aruco_id else ""
                print(f"ID {marker_id:>4d}{suffix}: {total} detections, {paired} paired with main")
        else:
            print("\nNo marker detections captured.")

        if correlations:
            print("\n--- Correlation vs main marker ---")
            for marker_id in sorted(correlations.keys()):
                stats = correlations[marker_id]
                print(f"ID {marker_id:>4d}: samples={int(stats['samples'])}, "
                      f"overall corr (X/Y)=({stats['overall_corr_x']:.4f}, {stats['overall_corr_y']:.4f}), "
                      f"noise corr (X/Y)=({stats['noise_corr_x']:.4f}, {stats['noise_corr_y']:.4f}), "
                      f"large corr (X/Y)=({stats['large_corr_x']:.4f}, {stats['large_corr_y']:.4f}), "
                      f"noise fraction={stats['noise_fraction']:.3f}")
        else:
            print("\nNo static markers overlapped with the main marker; nothing to correlate.")


if __name__ == "__main__":
    asyncio.run(run(AppConfig()))
