# ASCII only
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from ffmpeg_stream import FFMPEGMJPEGStream
from filters.boxcar import Boxcar
from jpeg_decoder import JPEGDecoder
from undistort import Undistorter
from aruco_tracker import ArucoTracker
from nats_publisher import NatsPublisher

# Optional Kalman on the center derived from corners
from kf_corner import CornerKalman, CornerKFConfig
from config import AppConfig


def _under_floor(dx: float, dy: float, *, radial: bool, f_mm: float, fx_mm: float, fy_mm: float) -> bool:
    if radial:
        return (dx*dx + dy*dy) ** 0.5 <= f_mm
    else:
        return (abs(dx) <= fx_mm) and (abs(dy) <= fy_mm)


async def run(app: AppConfig):
    cam = app.camera
    und = app.undistort
    arc = app.aruco
    jpg = app.jpeg
    ngc = app.noise
    run_cfg = app.run

    # ---------------- Product toggles ----------------
    USE_KF = False           # apply Kalman before filtering when under T1
    FILTER = Boxcar(N=6)   # filter applied in the noise region
    FALLBACK_RAW_UNTIL_READY = True  # during filter warmup in noise region, drop (recommended)

    # Floors
    use_radial = ngc.use_radial
    # T1 = raw floor (from config)
    T1_mm, T1x_mm, T1y_mm = ngc.floor_mm, ngc.floor_x_mm, ngc.floor_y_mm
    T1_mm = 50
    T1x_mm = 50
    T1y_mm = 50
    T2_VAL = .1

    # T2 = post-filter floor (scale T1 or set explicitly)
    T2_mm  = T2_VAL
    T2x_mm = T2_VAL
    T2y_mm = T2_VAL

    # ---------- Pipeline ----------
    stream = FFMPEGMJPEGStream(cam.device_name, cam.width, cam.height, cam.fps)
    decoder = JPEGDecoder(jpg.libjpeg_turbo_path)
    tracker = ArucoTracker(
        arc.dictionary, arc.aruco_id, arc.aruco_w_mm, arc.aruco_h_mm,
        frame_size_px=(cam.width, cam.height), isotropic_scale=False
    )
    K, D = und.as_np()
    undistorter = Undistorter(K, D, (cam.width, cam.height))

    # NATS
    pub = NatsPublisher(app.nats.servers, app.nats.subject, app.nats.enable)
    await pub.connect()
    await stream.start()

    # Per-axis filter instances (independent of any benchmark code)
    fx, fy = FILTER.copy(), FILTER.copy()

    # Optional Kalman
    if USE_KF:
        # Load R if measured; otherwise default is fine
        try:
            npy_path = Path(__file__).with_name("static_corner_cov.npy")
            R_meas = np.load(npy_path)
            if R_meas.shape != (8, 8):
                print(f"static_corner_cov.npy has shape {R_meas.shape}, expected (8,8); ignoring.")
                R_meas = None
        except Exception:
            R_meas = None
        kf = CornerKalman(CornerKFConfig(fps=cam.fps, q_process=1e3, R=R_meas))
    else:
        kf = None

    last_pub = None  # (x,y) of last value we actually published

    try:
        while True:
            jpg_bytes = await stream.read_jpeg()
            if jpg_bytes is None:
                print("[WARN] stream.read_jpeg() returned None; stopping.")
                break

            frame = decoder.decode_bgr(jpg_bytes)
            if und.enable_frame_undistort:
                frame = undistorter.remap(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            und_points_fn = undistorter.undistort_points if und.enable_corner_undistort else None
            result = tracker.detect_mm(gray, und_points_fn)
            if result is None:
                continue

            (mm, corners_px, _fov_mm) = result
            x_raw, y_raw = mm

            # First publish: always RAW; reset filters
            if last_pub is None:
                fx.reset(); fy.reset()
                await pub.publish_xy(x_raw, y_raw, angle_deg=0.0)
                last_pub = (x_raw, y_raw)
                continue

            # Δ vs last published (T1 decision)
            dx = x_raw - last_pub[0]
            dy = y_raw - last_pub[1]

            if not _under_floor(dx, dy, radial=use_radial, f_mm=T1_mm, fx_mm=T1x_mm, fy_mm=T1y_mm):
                # Real motion -> publish RAW, reset filters so they don't smear
                fx.reset(); fy.reset()
                await pub.publish_xy(x_raw, y_raw, angle_deg=0.0)
                last_pub = (x_raw, y_raw)
                continue

            # In noise region -> KF (optional) -> filter
            xin, yin = x_raw, y_raw
            if USE_KF:
                c = corners_px.astype(np.float64).reshape(4, 2)
                px_w = 0.5*(np.linalg.norm(c[1]-c[0]) + np.linalg.norm(c[2]-c[3]))
                px_h = 0.5*(np.linalg.norm(c[2]-c[1]) + np.linalg.norm(c[3]-c[0]))
                if px_w > 0 and px_h > 0:
                    sx = arc.aruco_w_mm / px_w
                    sy = arc.aruco_h_mm / px_h
                    z = np.empty(8, dtype=np.float64)
                    for i in range(4):
                        z[2*i+0] = c[i, 0] * sx
                        z[2*i+1] = c[i, 1] * sy
                    kf.step(z)
                    xin, yin = kf.get_center_mm()

            xv = fx.process(xin)
            yv = fy.process(yin)
            if xv is None or yv is None:
                if FALLBACK_RAW_UNTIL_READY:
                    await pub.publish_xy(x_raw, y_raw, angle_deg=0.0)
                    last_pub = (x_raw, y_raw)
                # else drop during warmup
                continue

            # Post-filter delta vs last published (T2 decision)
            dxf = float(xv) - last_pub[0]
            dyf = float(yv) - last_pub[1]
            if _under_floor(dxf, dyf, radial=use_radial, f_mm=T2_mm, fx_mm=T2x_mm, fy_mm=T2y_mm):
                # below floor -> drop
                continue

            # Filtered movement above T2 -> publish filtered
            await pub.publish_xy(float(xv), float(yv), angle_deg=0.0)
            last_pub = (float(xv), float(yv))

    except KeyboardInterrupt:
        pass
    finally:
        await pub.close()
        await stream.stop()

if __name__ == "__main__":
    asyncio.run(run(AppConfig()))
