# ASCII only
# main.py  — minimal raw publisher
import asyncio
import time

import cv2
from ffmpeg_stream import FFMPEGMJPEGStream
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker
from nats_publisher import NatsPublisher
from config import AppConfig


async def run(app: AppConfig):
    cam = app.camera
    und = app.undistort
    arc = app.aruco
    jpg = app.jpeg
    run_cfg = app.run

    # Pipeline objects
    stream = FFMPEGMJPEGStream(cam.device_name, cam.width, cam.height, cam.fps)
    decoder = JPEGDecoder(jpg.libjpeg_turbo_path)
    tracker = ArucoTracker(arc.dictionary, arc.aruco_id, arc.aruco_w_mm, arc.aruco_h_mm)

    K, D = und.as_np()
    undistorter = Undistorter(K, D, (cam.width, cam.height))  # used only if enabled

    pub = NatsPublisher(app.nats.servers, app.nats.subject, app.nats.enable)
    await pub.connect()
    await stream.start()

    samples = 0
    try:
        while run_cfg.max_samples <= 0 or samples < run_cfg.max_samples:
            jpg_bytes = await stream.read_jpeg()
            if jpg_bytes is None:
                break

            frame = decoder.decode_bgr(jpg_bytes)
            if und.enable_frame_undistort:
                frame = undistorter.remap(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            und_points_fn = undistorter.undistort_points if und.enable_corner_undistort else None
            mm = tracker.detect_mm(gray, und_points_fn)
            if mm is None:
                continue

            x_mm, y_mm = mm
            await pub.publish_xy(x_mm, y_mm, angle_deg=0.0)
            samples += 1

    except KeyboardInterrupt:
        pass
    finally:
        # clean shutdown (avoid Proactor warnings)
        await pub.close()
        await stream.stop()


if __name__ == "__main__":
    asyncio.run(run(AppConfig()))
