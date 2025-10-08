# ASCII only
# kf_corner.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

CENTER_STAGE_X = 50
CENTER_STAGE_Y = 50


def _I(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float64)

@dataclass
class CornerKFConfig:
    fps: float                 # camera rate (Hz)
    q_process: float = 1.0     # process noise scale (Q = q_process * I)
    # Measurement covariance 8x8 (mm^2) for [C0x,C0y,C1x,C1y,C2x,C2y,C3x,C3y]
    R: Optional[np.ndarray] = None
    # Initial covariance (variances) for positions (first 8) and velocities (last 8)
    p0_pos: float = 1e3
    p0_vel: float = 1e3

class CornerKalman:
    """
    State x (16):
        [ C0x,C0y, C1x,C1y, C2x,C2y, C3x,C3y,  V0x,V0y, V1x,V1y, V2x,V2y, V3x,V3y ]^T

    Constant-velocity model:
        A = [[I8, dt*I8],
             [ 0,    I8]]

    Process noise:
        Q = q_process * I16   (simple scaled identity; no coupling)

    Measurement:
        z = positions (8x1), H = [I8 | 0]
    """

    def __init__(self, cfg: CornerKFConfig):
        if cfg.fps <= 0:
            raise ValueError("fps must be > 0")
        self.dt = 1.0 / float(cfg.fps)

        # Dimensions
        self.np = 8   # position components
        self.nv = 8   # velocity components
        self.nx = self.np + self.nv  # 16
        self.nz = self.np            # 8

        I8 = _I(self.np)
        Z8 = np.zeros((self.np, self.np), dtype=np.float64)
        dt = self.dt

        # State transition (CV)
        self.A = np.block([
            [ I8, dt*I8],
            [ Z8,   I8 ],
        ])

        # Measurement (positions only)
        self.H = np.hstack([I8, np.zeros((self.np, self.nv), dtype=np.float64)])

        # Process noise (scaled identity)
        self.q_process = float(cfg.q_process)
        self.Q = self.q_process * _I(self.nx)

        # Measurement covariance
        if cfg.R is None:
            self.R = 1e-3 * _I(self.nz)
        else:
            self.R = np.array(cfg.R, dtype=np.float64)
            if self.R.shape != (self.nz, self.nz):
                raise ValueError("R must be 8x8 (mm^2)")

        # Initial state and covariance
        self.x = np.zeros((self.nx, 1), dtype=np.float64)
        self.P = np.diag(
            [cfg.p0_pos]*self.np + [cfg.p0_vel]*self.nv
        ).astype(np.float64)

        # Scratch
        self._S = np.zeros((self.nz, self.nz), dtype=np.float64)
        self._K = np.zeros((self.nx, self.nz), dtype=np.float64)
        self._y = np.zeros((self.nz, 1), dtype=np.float64)

    # ---- simple tuning knob ----
    def set_q_process(self, q: float) -> None:
        self.q_process = float(q)
        self.Q = self.q_process * _I(self.nx)

    # ---- filter core ----
    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z_pos_mm: np.ndarray) -> None:
        z = np.asarray(z_pos_mm, dtype=np.float64).reshape(self.nz, 1)
        self._y = z - (self.H @ self.x)
        self._S = self.H @ self.P @ self.H.T + self.R
        self._K = self.P @ self.H.T @ np.linalg.inv(self._S)
        I = _I(self.nx)
        # Joseph form for numerical stability
        KH = self._K @ self.H
        self.x = self.x + (self._K @ self._y)
        self.P = (I - KH) @ self.P @ (I - KH).T + self._K @ self.R @ self._K.T

    def step(self, z_pos_mm: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        self.predict()
        if z_pos_mm is not None:
            self.update(z_pos_mm)
        pos = self.x[:self.np, 0].copy()
        vel = self.x[self.np:, 0].copy()
        return pos, vel

    def reset(self, pos0: Optional[np.ndarray] = None, vel0: Optional[np.ndarray] = None,
              p0_pos: Optional[float] = None, p0_vel: Optional[float] = None) -> None:
        self.x[:] = 0.0
        if pos0 is not None:
            p = np.asarray(pos0, dtype=np.float64).reshape(self.np)
            self.x[:self.np, 0] = p
        if vel0 is not None:
            v = np.asarray(vel0, dtype=np.float64).reshape(self.nv)
            self.x[self.np:, 0] = v
        if p0_pos is None or p0_vel is None:
            Ppos = np.diag(np.full(self.np, self.P[0,0], dtype=np.float64))
            Pvel = np.diag(np.full(self.nv, self.P[self.np,self.np], dtype=np.float64))
        self.P = np.diag(
            [p0_pos if p0_pos is not None else 1e3]*self.np +
            [p0_vel if p0_vel is not None else 1e3]*self.nv
        ).astype(np.float64)

    def get_center_mm(self) -> Tuple[float, float]:
        px = self.x[:self.np, 0]  # 8 positions
        cx = float(np.mean(px[0::2]))  - CENTER_STAGE_X  # average x's
        cy = float(np.mean(px[1::2]))  - CENTER_STAGE_Y  # average y's
        return cx, cy
