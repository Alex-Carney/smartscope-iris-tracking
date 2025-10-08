# ASCII only
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from ffmpeg_stream import FFMPEGMJPEGStream
from filter_benchmark_compare import FilterBenchmarkCompare
from filters.boxcar import Boxcar
from filters.ema import EMA
from filters.sgq import CausalSavGol
from filters.butterworth import BiquadLowpass
from filters.ema_cascade import CascadedEMA
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker
from noise_gate import NoiseGate
from basic_benchmark import BasicBenchmark
from nats_publisher import NatsPublisher
from fft_benchmark import NoiseBenchmark
from fps_glitch_benchmark import FPSGlitchBenchmark
from frame_repeat_probe import FrameRepeatProbe
from time_accounting import TimeAccounting
from noise_adaptive_filter import NoiseAdaptiveDualFloor2D
from corner_stats_benchmark import CornerStatsBenchmark

# NEW: Kalman
from kf_corner import CornerKalman, CornerKFConfig
from kalman_benchmark import KalmanBenchmark, KalmanBenchmarkConfig

from config import AppConfig


async def run(app: AppConfig):
    cam = app.camera
    und = app.undistort
    arc = app.aruco
    jpg = app.jpeg
    ngc = app.noise
    run_cfg = app.run

    # --------------------------------------------
    # FILTERS: define two to compare (unchanged)
    # --------------------------------------------
    filter_A = Boxcar(N=27)
    filter_B = CascadedEMA(alpha=0.2, stages=3)

    # --------------------------------------------
    # SIMPLE SWITCH: what do we send to NATS?
    #   "raw"       -> publish raw positions
    #   "filter_a"  -> publish filter_A output
    #   "filter_b"  -> publish filter_B output
    #   "kf"        -> publish Kalman-filtered center
    # --------------------------------------------
    PUBLISH_MODE = "kf"                # "raw" | "filter_a" | "filter_b" | "kf"
    FALLBACK_RAW_UNTIL_READY = True

    # Dedicated per-axis instances for the publish path (do not reuse comparator's)
    pub_fx = pub_fy = None
    if PUBLISH_MODE == "filter_a":
        pub_fx, pub_fy = filter_A.copy(), filter_A.copy()
    elif PUBLISH_MODE == "filter_b":
        pub_fx, pub_fy = filter_B.copy(), filter_B.copy()

    # Comparator (makes its own internal copies)
    filters = FilterBenchmarkCompare(
        filter_a=filter_A,
        filter_b=filter_B,
        fps_hint=cam.fps,
        out_path="filter_compare_boxcar9_vs_ema025.png",
        title=f"Filter comparison {str(filter_A)} vs {str(filter_B)}"
    )

    # --- STREAM/DECODE/PIPELINE ---
    stream = FFMPEGMJPEGStream(cam.device_name, cam.width, cam.height, cam.fps)
    decoder = JPEGDecoder(jpg.libjpeg_turbo_path)
    timer = TimeAccounting()
    tracker = ArucoTracker(arc.dictionary, arc.aruco_id, arc.aruco_w_mm, arc.aruco_h_mm)
    glitch = FPSGlitchBenchmark(out_path="fps_glitch_benchmark.png")
    repeat_probe = FrameRepeatProbe(out_path="frame_repeat_probe.png")

    K, D = und.as_np()
    undistorter = Undistorter(K, D, (cam.width, cam.height))

    noise_gate = NoiseGate(ngc.enable, ngc.use_radial, ngc.floor_mm, ngc.floor_x_mm, ngc.floor_y_mm)
    bench = BasicBenchmark()

    # Noise benchmark (raw center)
    noise_bench = NoiseBenchmark(out_path="noise_benchmark.png", fps_hint=cam.fps)

    # Per-corner stats (raw corners)
    corner_bench = CornerStatsBenchmark(
        aruco_w_mm=arc.aruco_w_mm,
        aruco_h_mm=arc.aruco_h_mm,
        fps_hint=cam.fps,
        out_path="corner_stats.png",
        title="Corner per-axis noise"
    )

    # --- KALMAN: load R if available, create filter + benchmark ---
    try:
        npy_path = Path(__file__).with_name("corner_cov.npy")
        R_meas = np.load(npy_path)  # expected 8x8 mm^2
        if R_meas.shape != (8, 8):
            print(f"corner_cov.npy has shape {R_meas.shape}, expected (8,8); ignoring.")
            R_meas = None
    except FileNotFoundError:
        print("R was not measured yet (corner_cov.npy not found). Using default R.")
        R_meas = None
    except Exception as e:
        print(f"Failed to load corner_cov.npy: {e}. Using default R.")
        R_meas = None

    Q_accel_value = 1e5  # increase by decades to hug motion harder

    kf = CornerKalman(CornerKFConfig(
        fps=cam.fps,
        q_accel=Q_accel_value,
        R=R_meas
    ))

    kal_bench = KalmanBenchmark(KalmanBenchmarkConfig(
        out_path="kalman_benchmark.png",
        fps_hint=cam.fps,
        title="Kalman vs RAW (center)"
    ))

    pub = NatsPublisher(app.nats.servers, app.nats.subject, app.nats.enable)
    await pub.connect()
    await stream.start()
    first_frame_saved = False

    # ---- sub-noise filter used by the adaptive logic ----
    subnoise_fx = Boxcar(N=15)
    subnoise_fy = Boxcar(N=15)

    # Pick Layer-2 floor (T2). Example: 10x lower than T1.
    T2_SCALE = 0.10
    floor2_mm   = max(1e-12, ngc.floor_mm   * T2_SCALE)
    floor2_x_mm = max(1e-12, ngc.floor_x_mm * T2_SCALE)
    floor2_y_mm = max(1e-12, ngc.floor_y_mm * T2_SCALE)

    naf = NoiseAdaptiveDualFloor2D(
        subnoise_fx, subnoise_fy,
        use_radial=ngc.use_radial,
        floor1_mm=ngc.floor_mm, floor1_x_mm=ngc.floor_x_mm, floor1_y_mm=ngc.floor_y_mm,
        floor2_mm=floor2_mm,   floor2_x_mm=floor2_x_mm,   floor2_y_mm=floor2_y_mm,
        drop_during_warmup=True,
    )

    last_published = None

    try:
        while len(bench.positions) < run_cfg.max_samples:
            jpg_bytes = await stream.read_jpeg()
            if jpg_bytes is None:
                print('\n\n\n MAJOR GLITCH OCCURED. BREAKING --- stream read returned None --- \n\n\n')
                break

            timer.start_frame()
            bench.mark_processed()

            frame = decoder.decode_bgr(jpg_bytes)
            timer.mark("jpeg_decode")
            now = time.perf_counter()
            repeat_probe.add_jpeg(now, jpg_bytes)

            if und.enable_frame_undistort:
                frame = undistorter.remap(frame)
            timer.mark("undistort frame")

            if not first_frame_saved and run_cfg.save_first_frame:
                cv2.imwrite(run_cfg.save_first_frame_path, frame)
                print(f"Saved first frame to {run_cfg.save_first_frame_path}")
                first_frame_saved = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            repeat_probe.add_gray(gray)

            und_points_fn = undistorter.undistort_points if und.enable_corner_undistort else None
            result = tracker.detect_mm(gray, und_points_fn)
            if result is None:
                continue  # IMPORTANT: no predict-only step per your request

            timer.mark("aruco detect")
            bench.mark_with_marker()

            # Unpack detection
            (center_mm, corners_px) = result
            x_mm, y_mm = center_mm

            # Benchmarks
            now = time.perf_counter()
            filters.add(x_mm, y_mm)
            glitch.add(now, x_mm, y_mm)
            corner_bench.add(corners_px)
            bench.tick_fps()

            # Feed raw center to the noise benchmark (once per frame)
            noise_bench.add(x_mm, y_mm)

            # --------- KALMAN: measurement z (8,) in mm from corners and step ---------
            c = corners_px.astype(np.float64).reshape(4, 2)
            px_w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
            px_h = 0.5 * (np.linalg.norm(c[2] - c[1]) + np.linalg.norm(c[3] - c[0]))
            if px_w <= 0 or px_h <= 0:
                # degenerate; skip KF this frame (but keep rest of pipeline)
                continue

            sx = arc.aruco_w_mm / px_w
            sy = arc.aruco_h_mm / px_h

            z_mm = np.empty(8, dtype=np.float64)
            for i in range(4):
                z_mm[2*i + 0] = c[i, 0] * sx
                z_mm[2*i + 1] = c[i, 1] * sy

            pos8, vel8 = kf.step(z_mm)
            kx, ky = kf.get_center_mm()
            kal_bench.add(x_mm, y_mm, kx, ky)
            # --------------------------------------------------------------------

            # -------------------------------
            # SELECT WHAT TO PUBLISH
            # -------------------------------
            if PUBLISH_MODE == "kf":
                # Publish Kalman-filtered center directly
                out_x, out_y = kx, ky
                await pub.publish_xy(out_x, out_y, angle_deg=0.0)
                last_published = (out_x, out_y)
                bench.add_position(x_mm, y_mm)  # keep raw in stats
            else:
                # Existing adaptive-path for raw / filter_a / filter_b modes
                publish, out_x, out_y, mode = naf.process(x_mm, y_mm, last_published)

                if publish:
                    await pub.publish_xy(out_x, out_y, angle_deg=0.0)
                    last_published = (out_x, out_y)
                    bench.add_position(x_mm, y_mm)  # count as kept only when we publish
                else:
                    if pub_fx is None:
                        out_x, out_y = x_mm, y_mm
                    else:
                        fx = pub_fx.process(x_mm)
                        fy = pub_fy.process(y_mm)
                        if fx is not None and fy is not None:
                            out_x, out_y = fx, fy
                        else:
                            if FALLBACK_RAW_UNTIL_READY:
                                out_x, out_y = x_mm, y_mm
                            else:
                                timer.mark("End")
                                timer.end_frame()
                                continue

                    await pub.publish_xy(out_x, out_y, angle_deg=0.0)
                    last_published = (out_x, out_y)

            timer.mark("End")
            timer.end_frame()

        print("Main loop ended.")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise
    finally:
        await pub.close()
        await stream.stop()
        glitch.finish()
        timer.finish()
        filters.finish()
        repeat_probe.finish()
        noise_bench.finish()
        corner_bench.finish()
        kal_bench.finish()  # KF vs RAW figure
        print(naf.summary())
        bench.print_summary("(published samples)")


if __name__ == "__main__":
    asyncio.run(run(AppConfig()))
