from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class CameraConfig:
    # device_name: str = "Global Shutter Camera"
    device_name: str = "4K AF Camera"
    width: int = 3840
    height: int = 2160
    fps: int = 30
    # ffmpeg uses dshow on Windows; adjust if you change platforms

@dataclass
class ArucoConfig:
    aruco_id: int = 0 
    aruco_w_mm: float = 19.05  # 0.75 inches
    aruco_h_mm: float = 19.05
    dictionary: str = "DICT_4X4_50"  # OpenCV predefined dictionary name

@dataclass
class UndistortConfig:
    enable_frame_undistort: bool = False # remap each frame
    enable_corner_undistort: bool = False # undistort corners before subpix
    camera_matrix: List[List[float]] = field(default_factory=lambda: [
    [9.41642211e+03, 0.0, 9.64483756e+02],
    [0.0, 9.40228615e+03, 6.08511402e+02],
    [0.0, 0.0, 1.0]
    ])
    dist_coeffs: List[float] = field(default_factory=lambda: [
    4.64572643e+00, -2.68199626e+02, -1.41158321e-02, -2.14888005e-02, -2.27739842e+00
    ])
    def as_np(self) -> Tuple[np.ndarray, np.ndarray]:
        K = np.array(self.camera_matrix, dtype=np.float64)
        D = np.array(self.dist_coeffs, dtype=np.float64)
        return K, D


@dataclass
class JPEGConfig:
    libjpeg_turbo_path: str = r"c:\\libjpeg-turbo-gcc64\\bin\\libturbojpeg.dll"

@dataclass
class NATSConfig:
    servers: Tuple[str, ...] = ("nats://129.170.212.19:32201",)
    subject: str = "camera.stream.panda"
    enable: bool = True

@dataclass
class NoiseGateConfig:
    enable: bool = True
    use_radial: bool = False
    floor_mm: float = 50.0
    floor_x_mm: float = 50.0
    floor_y_mm: float = 50.0

@dataclass
class RunConfig:
    print_coords: bool = False
    max_samples: int = 1000
    save_first_frame: bool = True
    save_first_frame_path: str = "frame_undistorted.png"

@dataclass
class AppConfig:
    # IMPORTANT: use default_factory for nested dataclasses (they’re mutable)
    camera: CameraConfig = field(default_factory=CameraConfig)
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    undistort: UndistortConfig = field(default_factory=UndistortConfig)
    jpeg: JPEGConfig = field(default_factory=JPEGConfig)
    nats: NATSConfig = field(default_factory=NATSConfig)
    noise: NoiseGateConfig = field(default_factory=NoiseGateConfig)
    run: RunConfig = field(default_factory=RunConfig)