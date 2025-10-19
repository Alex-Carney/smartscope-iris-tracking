# ASCII only
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from ffmpeg_stream import FFMPEGMJPEGStream
from filter_benchmark_compare import FilterBenchmarkCompare
from filters.boxcar import Boxcar
from filters.ema_cascade import CascadedEMA
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker
from basic_benchmark import BasicBenchmark
from nats_publisher import NatsPublisher
from fft_benchmark import NoiseBenchmark
from fps_glitch_benchmark import FPSGlitchBenchmark
from frame_repeat_probe import FrameRepeatProbe
from time_accounting import TimeAccounting
from corner_stats_benchmark import CornerStatsBenchmark

# Kalman
from kf_corner import CornerKalman, CornerKFConfig
from kalman_benchmark import KalmanBenchmark, KalmanBenchmarkConfig

from config import AppConfig


async def run(app: AppConfig):
    cam = app.camera
    und = app.undistort
    arc = app.aruco
    jpg = app.jpeg
    run_cfg = app.run

    # -------- Filters to compare (benchmark only) --------
    filter_A = Boxcar(N=27)
    filter_B = CascadedEMA(alpha=0.2, stages=3)

    PUBLISH_MODE = "raw"        # "raw" | "filter_a" | "filter_b" | "kf"
    FALLBACK_RAW_UNTIL_READY = True

    # NEW: drop initial detections instead of sleeping
    SKIP_FIRST_N_DETECTIONS = 200  # <-- tune after you inspect the transient

    pub_fx = pub_fy = None
    if PUBLISH_MODE == "filter_a":
        pub_fx, pub_fy = filter_A.copy(), filter_A.copy()
    elif PUBLISH_MODE == "filter_b":
        pub_fx, pub_fy = filter_B.copy(), filter_B.copy()

    filters = FilterBenchmarkCompare(
        filter_a=filter_A,
        filter_b=filter_B,
        fps_hint=cam.fps,
        out_path="filter_compare.png",
        title=f"Filter comparison {str(filter_A)} vs {str(filter_B)}"
    )

    # --- STREAM/DECODE/PIPELINE ---
    stream = FFMPEGMJPEGStream(cam.device_name, cam.width, cam.height, cam.fps)
    decoder = JPEGDecoder(jpg.libjpeg_turbo_path)
    timer = TimeAccounting()
    tracker = ArucoTracker(
        arc.dictionary, arc.aruco_id, arc.aruco_w_mm, arc.aruco_h_mm,
        frame_size_px=(cam.width, cam.height), isotropic_scale=False
    )

    glitch = FPSGlitchBenchmark(out_path="fps_glitch_benchmark.png")
    repeat_probe = FrameRepeatProbe(out_path="frame_repeat_probe.png")

    K, D = und.as_np()
    undistorter = Undistorter(K, D, (cam.width, cam.height))

    bench = BasicBenchmark()

    # Benchmarks (raw center + raw corners)
    noise_bench = NoiseBenchmark(out_path="noise_benchmark.png", fps_hint=cam.fps)
    corner_bench = CornerStatsBenchmark(
        aruco_w_mm=arc.aruco_w_mm, aruco_h_mm=arc.aruco_h_mm,
        fps_hint=cam.fps, out_path="corner_stats.png",
        title="Corner per-axis noise"
    )

    # --- KALMAN: optional compare vs raw (for plot only) ---
    try:
        npy_path = Path(__file__).with_name("static_corner_cov.npy")
        R_meas = np.load(npy_path)  # 8x8 mm^2
        if R_meas.shape != (8, 8):
            print(f"static_corner_cov.npy has shape {R_meas.shape}, expected (8,8); ignoring.")
            R_meas = None
    except FileNotFoundError:
        print("R not measured yet (static_corner_cov.npy missing). Using default R.")
        R_meas = None
    except Exception as e:
        print(f"Failed to load static_corner_cov.npy: {e}. Using default R.")
        R_meas = None

    Q_VALUE = 1e3
    kf = CornerKalman(CornerKFConfig(fps=cam.fps, q_process=Q_VALUE, R=R_meas))
    kal_bench = KalmanBenchmark(KalmanBenchmarkConfig(
        out_path="kalman_benchmark.png", fps_hint=cam.fps, title="Kalman vs RAW (center)"
    ))

    pub = NatsPublisher(app.nats.servers, app.nats.subject, app.nats.enable)
    await pub.connect()
    await stream.start()
    first_frame_saved = False

    samples_collected = 0          # count **valid detections** kept for benchmarking
    skipped_detections = 0         # count initial detections we drop on purpose
    announced_skip = False

    try:
        while samples_collected < run_cfg.max_samples:
            jpg_bytes = await stream.read_jpeg()
            if jpg_bytes is None:
                print("\n[WARN] stream.read_jpeg() returned None; stopping.\n")
                break

            # Decode first (common to both paths)
            frame = decoder.decode_bgr(jpg_bytes)
            now = time.perf_counter()
            repeat_probe.add_jpeg(now, jpg_bytes)

            if und.enable_frame_undistort:
                frame = undistorter.remap(frame)

            if not first_frame_saved and run_cfg.save_first_frame:
                cv2.imwrite(run_cfg.save_first_frame_path, frame)
                print(f"Saved first frame to {run_cfg.save_first_frame_path}")
                first_frame_saved = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            repeat_probe.add_gray(gray)

            # Try to detect (we want ROI/subpix to warm even during skip)
            und_points_fn = undistorter.undistort_points if und.enable_corner_undistort else None
            result = tracker.detect_mm(gray, und_points_fn)
            if result is None:
                # No detection: do not count for skip or samples; just continue
                continue

            # -------------------------------
            # SKIP PHASE: drop first N detections from ALL stats/plots/publish
            # -------------------------------
            if skipped_detections < SKIP_FIRST_N_DETECTIONS:
                skipped_detections += 1
                if not announced_skip:
                    print(f"[INFO] Dropping first {SKIP_FIRST_N_DETECTIONS} detections to bypass camera transient...")
                    announced_skip = True
                # Optionally print progress every 50
                if skipped_detections % 50 == 0 or skipped_detections == SKIP_FIRST_N_DETECTIONS:
                    print(f"[INFO] Dropped {skipped_detections}/{SKIP_FIRST_N_DETECTIONS}")
                # Do NOT start/mark/end timer; do NOT touch any benchmarks; do NOT publish.
                continue

            # -------------------------------
            # NORMAL BENCHMARK/PUBLISH PATH
            # -------------------------------
            timer.start_frame()
            bench.mark_processed()

            # We’ve already decoded & grayscale’d above; just unpack and proceed.
            (mm, corners_px, fov_mm) = result
            x_mm, y_mm = mm

            # Benchmarks
            timer.mark("jpeg_decode")          # light approximation (already decoded)
            timer.mark("undistort frame")      # ditto; for rough attribution
            bench.mark_with_marker()

            # Stats/plots
            filters.add(x_mm, y_mm)
            glitch.add(now, x_mm, y_mm)
            corner_bench.add(corners_px)
            bench.tick_fps()
            noise_bench.add(x_mm, y_mm)

            # KF for comparison plot
            c = corners_px.astype(np.float64).reshape(4, 2)
            px_w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
            px_h = 0.5 * (np.linalg.norm(c[2] - c[1]) + np.linalg.norm(c[3] - c[0]))
            if px_w > 0 and px_h > 0:
                sx = arc.aruco_w_mm / px_w
                sy = arc.aruco_h_mm / px_h
                z_mm = np.empty(8, dtype=np.float64)
                for i in range(4):
                    z_mm[2*i + 0] = c[i, 0] * sx
                    z_mm[2*i + 1] = c[i, 1] * sy
                kf.step(z_mm)
                kx, ky = kf.get_center_mm()
                kal_bench.add(x_mm, y_mm, kx, ky)
            else:
                kx, ky = x_mm, y_mm

            # Optional publish
            if PUBLISH_MODE == "raw":
                out_x, out_y = x_mm, y_mm
            elif PUBLISH_MODE == "filter_a":
                fx, fy = pub_fx.process(x_mm), pub_fy.process(y_mm)
                if fx is None or fy is None:
                    out_x, out_y = (x_mm, y_mm) if FALLBACK_RAW_UNTIL_READY else (None, None)
                else:
                    out_x, out_y = fx, fy
            elif PUBLISH_MODE == "filter_b":
                fx, fy = pub_fx.process(x_mm), pub_fy.process(y_mm)
                if fx is None or fy is None:
                    out_x, out_y = (x_mm, y_mm) if FALLBACK_RAW_UNTIL_READY else (None, None)
                else:
                    out_x, out_y = fx, fy
            else:  # "kf"
                out_x, out_y = (kx, ky)

            if out_x is not None and out_y is not None:
                await pub.publish_xy(out_x, out_y, angle_deg=0.0)
                if run_cfg.print_coords:
                    print(f"Published: x={out_x:.3f} mm, y={out_y:.3f} mm")

            samples_collected += 1
            timer.mark("End")
            timer.end_frame()

        print("Main loop ended.")

    except KeyboardInterrupt:
        pass
    finally:
        await pub.close()
        await stream.stop()
        glitch.finish()
        timer.finish()
        filters.finish()
        repeat_probe.finish()
        noise_bench.finish()
        corner_bench.finish()
        kal_bench.finish()
        bench.print_summary("(detections)")


if __name__ == "__main__":
    asyncio.run(run(AppConfig()))
