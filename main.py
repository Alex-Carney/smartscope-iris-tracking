import asyncio
import time

import cv2
import numpy as np
from ffmpeg_stream import FFMPEGMJPEGStream
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker
from noise_gate import NoiseGate
from basic_benchmark import BasicBenchmark
from nats_publisher import NatsPublisher
from fft_benchmark import NoiseBenchmark
from fps_glitch_benchmark import FPSGlitchBenchmark
from config import AppConfig


async def run(app: AppConfig):
    cam = app.camera
    und = app.undistort
    arc = app.aruco
    jpg = app.jpeg
    ngc = app.noise
    run_cfg = app.run

    stream = FFMPEGMJPEGStream(cam.device_name, cam.width, cam.height, cam.fps)
    decoder = JPEGDecoder(jpg.libjpeg_turbo_path)
    tracker = ArucoTracker(arc.dictionary, arc.aruco_id, arc.aruco_w_mm, arc.aruco_h_mm)
    glitch = FPSGlitchBenchmark(out_path="fps_glitch_benchmark.png")

    K, D = und.as_np()
    undistorter = Undistorter(K, D, (cam.width, cam.height))

    noise_gate = NoiseGate(ngc.enable, ngc.use_radial, ngc.floor_mm, ngc.floor_x_mm, ngc.floor_y_mm)
    bench = BasicBenchmark()

    # NEW noise benchmark instance (use camera FPS as the Allan/FFT sampling rate)
    noise_bench = NoiseBenchmark(out_path="noise_benchmark.png", fps_hint=cam.fps)

    pub = NatsPublisher(app.nats.servers, app.nats.subject, app.nats.enable)
    await pub.connect()

    await stream.start()

    first_frame_saved = False

    try:
        while len(bench.positions) < run_cfg.max_samples:
            jpg_bytes = await stream.read_jpeg()
            if jpg_bytes is None:
                break
            bench.mark_processed()
            frame = decoder.decode_bgr(jpg_bytes)

            if und.enable_frame_undistort:
                frame = undistorter.remap(frame)

            if not first_frame_saved and run_cfg.save_first_frame:
                cv2.imwrite(run_cfg.save_first_frame_path, frame)
                print(f"Saved first frame to {run_cfg.save_first_frame_path}")
                first_frame_saved = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            und_points_fn = undistorter.undistort_points if und.enable_corner_undistort else None
            mm = tracker.detect_mm(gray, und_points_fn)
            if mm is None:
                continue
            bench.mark_with_marker()
            x_mm, y_mm = mm
            now = time.perf_counter()
            glitch.add(now, x_mm, y_mm)
            bench.tick_fps()

            if not noise_gate.should_send(x_mm, y_mm):
                bench.mark_skipped()
                continue

            # Keep sample, print, publish
            bench.add_position(x_mm, y_mm)
            print(f"{x_mm:.5f},{y_mm:.5f}")
            await pub.publish_xy(x_mm, y_mm, angle_deg=0.0)

            # Feed noise benchmark
            noise_bench.add(x_mm, y_mm)

    except KeyboardInterrupt:
        pass
    finally:
        await pub.close()
        await stream.stop()  # drain subprocess first to avoid Proactor warnings
        # Save the noise benchmark figure
        glitch.finish()
        noise_bench.finish()
        bench.print_summary("(kept samples)")

if __name__ == "__main__":
    asyncio.run(run(AppConfig()))