# ASCII only
# kf_corner.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

"""
State x (16x1), order per corner (OpenCV ArUco: TL, TR, BR, BL):
    x = [ C0x,C0y, C1x,C1y, C2x,C2y, C3x,C3y,  V0x,V0y, V1x,V1y, V2x,V2y, V3x,V3y ]^T
Positions first (8), then velocities (8).

Continuous-time model (random-acceleration):
    d/dt [pos] = [vel]
    d/dt [vel] = w(t)     (white accel noise, 8x1)

In block form:
    x_dot = A_c x + G_c w,  with
       A_c = [[0 I],
              [0 0]]      (I = I8)
       G_c = [[0],
              [I]]

Zero-order hold discretization with sample time dt:
    A_d = [[I, dt*I],
           [0,    I]]
    G_d = [[0.5*dt^2 * I],
           [     dt  * I]]

Measurement z (8x1): corner positions only
    z = H x + v, with H = [I 0] (8x16)

Process covariance per-step:
    Q = G_d * Qw * G_d^T, where Qw is 8x8 accel-noise power (mm^2 / s^3).
Measurement covariance:
    R = 8x8 (mm^2), from your corner stats (C0x,C0y,C1x,C1y,... order).
"""

def _I(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float64)

@dataclass
class CornerKFConfig:
    fps: float                         # camera rate (Hz)
    q_accel: float = 1.0               # scalar accel-noise power (mm^2/s^3), will scale I8
    R: Optional[np.ndarray] = None     # 8x8 measurement covariance in mm^2 (positions only)

class CornerKalman:
    def __init__(self, cfg: CornerKFConfig):
        if cfg.fps <= 0:
            raise ValueError("fps must be > 0")
        self.dt = 1.0 / float(cfg.fps)

        # Dimensions
        self.np = 8  # pos components (C0x,C0y,C1x,C1y,C2x,C2y,C3x,C3y)
        self.nv = 8  # vel components
        self.nx = self.np + self.nv     # 16
        self.nz = self.np               # 8

        I8 = _I(self.np)
        Z8 = np.zeros((self.np, self.np), dtype=np.float64)

        # Discrete A, G (closed form for constant-velocity)
        dt = self.dt
        self.A = np.block([
            [ I8, dt*I8],
            [ Z8,  I8  ],
        ])
        self.G = np.vstack([
            0.5 * (dt*dt) * I8,
            dt * I8
        ])  # (16x8)

        # Measurement matrix H: select positions
        self.H = np.hstack([I8, np.zeros((self.np, self.nv), dtype=np.float64)])  # (8x16)

        # Process noise power Qw (accel); start diagonal, tunable
        self.Qw = (float(cfg.q_accel) * _I(self.nv))  # (8x8)

        # Per-step process covariance
        self.Q = self.G @ self.Qw @ self.G.T  # (16x16)

        # Measurement covariance
        if cfg.R is None:
            # conservative diag default (mm^2) — replace with your measured R
            self.R = 1e-3 * _I(self.nz)
        else:
            self.R = np.array(cfg.R, dtype=np.float64)
            if self.R.shape != (self.nz, self.nz):
                raise ValueError("R must be 8x8 (mm^2)")

        # Init state and covariance (big P lets the filter accept the first measurements quickly)
        self.x = np.zeros((self.nx, 1), dtype=np.float64)
        self.P = 1e3 * _I(self.nx)

        # Pre-alloc scratch
        self._S = np.zeros((self.nz, self.nz), dtype=np.float64)
        self._K = np.zeros((self.nx, self.nz), dtype=np.float64)
        self._y = np.zeros((self.nz, 1), dtype=np.float64)

    # ------- tuning knobs -------
    def set_q_accel(self, q_accel: float) -> None:
        """Set scalar accel noise power (mm^2/s^3). Bigger -> faster tracking, less smoothing."""
        self.Qw[:, :] = 0.0
        self.Qw[np.diag_indices(self.nv)] = float(q_accel)
        self.Q = self.G @ self.Qw @ self.G.T

    def set_Qw(self, Qw: np.ndarray) -> None:
        """Set full 8x8 accel noise power (lets you add cross-correlations if desired)."""
        Qw = np.array(Qw, dtype=np.float64)
        if Qw.shape != (self.nv, self.nv):
            raise ValueError("Qw must be 8x8")
        self.Qw = Qw
        self.Q = self.G @ self.Qw @ self.G.T

    def set_R(self, R: np.ndarray) -> None:
        R = np.array(R, dtype=np.float64)
        if R.shape != (self.nz, self.nz):
            raise ValueError("R must be 8x8")
        self.R = R

    # ------- core filter -------
    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z_pos_mm: np.ndarray) -> None:
        """
        z_pos_mm: shape (8,) or (8,1) — positions of the 4 corners in mm, order:
            [C0x,C0y, C1x,C1y, C2x,C2y, C3x,C3y]
        """
        z = np.asarray(z_pos_mm, dtype=np.float64).reshape(self.nz, 1)

        # innovation
        self._y = z - (self.H @ self.x)

        # innovation cov
        self._S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        self._K = self.P @ self.H.T @ np.linalg.inv(self._S)

        # state/cov update (Joseph form is robust; standard form is fine here)
        self.x = self.x + (self._K @ self._y)
        I = _I(self.nx)
        KH = self._K @ self.H
        self.P = (I - KH) @ self.P @ (I - KH).T + self._K @ self.R @ self._K.T

    def step(self, z_pos_mm: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        One frame:
          - predict always
          - if measurement present, update
        Returns:
          (pos_mm (8,), vel_mm_per_s (8,))
        """
        self.predict()
        if z_pos_mm is not None:
            self.update(z_pos_mm)

        pos = self.x[:self.np, 0].copy()
        vel = self.x[self.np:, 0].copy()
        return pos, vel

    # Convenience getters
    def get_center_mm(self) -> Tuple[float, float]:
        px = self.x[:self.np, 0]  # 8
        cx = float(np.mean(px[0::2]))  # average x of four corners
        cy = float(np.mean(px[1::2]))  # average y of four corners
        return cx, cy
