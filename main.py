import asyncio
import cv2
import numpy as np

from ffmpeg_stream import FFMPEGMJPEGStream
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker
from noise_gate import NoiseGate
from basic_benchmark import BasicBenchmark
from nats_publisher import NatsPublisher

from config import AppConfig


async def run(app: AppConfig):
    camera_cfg = app.camera
    undistort_cfg = app.undistort
    aruco_cfg = app.aruco
    jpg_cfg = app.jpeg
    noise_cfg = app.noise
    run_cfg = app.run

    stream = FFMPEGMJPEGStream(camera_cfg.device_name, camera_cfg.width, camera_cfg.height, camera_cfg.fps)
    decoder = JPEGDecoder(jpg_cfg.libjpeg_turbo_path)
    tracker = ArucoTracker(aruco_cfg.dictionary, aruco_cfg.aruco_id, aruco_cfg.aruco_w_mm, aruco_cfg.aruco_h_mm)

    K, D = undistort_cfg.as_np()
    undistorter = Undistorter(K, D, (camera_cfg.width, camera_cfg.height))

    noise_gate = NoiseGate(noise_cfg.enable, noise_cfg.use_radial, noise_cfg.floor_mm, noise_cfg.floor_x_mm, noise_cfg.floor_y_mm)
    bench = BasicBenchmark()

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

            if undistort_cfg.enable_frame_undistort:
                frame = undistorter.remap(frame)

            if not first_frame_saved and run_cfg.save_first_frame:
                cv2.imwrite(run_cfg.save_first_frame_path, frame)
                print(f"Saved first frame to {run_cfg.save_first_frame_path}")
                first_frame_saved = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            und_points_fn = undistorter.undistort_points if undistort_cfg.enable_corner_undistort else None
            mm = tracker.detect_mm(gray, und_points_fn)
            if mm is None:
                continue
            bench.mark_with_marker()
            x_mm, y_mm = mm
            bench.tick_fps()

            if not noise_gate.should_send(x_mm, y_mm):
                bench.mark_skipped()
                continue

            # Keep sample, print, and publish
            bench.add_position(x_mm, y_mm)
            print(f"{x_mm:.5f},{y_mm:.5f}")
            await pub.publish_xy(x_mm, y_mm, angle_deg=0.0)

    except KeyboardInterrupt:
        pass
    finally:
        await pub.close()
        await stream.stop()
        bench.print_summary("(kept samples)")

if __name__ == "__main__":
    asyncio.run(run(AppConfig()))