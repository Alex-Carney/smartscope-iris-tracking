# ASCII only
# kf_corner.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

def _I(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float64)

@dataclass
class CornerKFConfig:
    fps: float                         # camera rate (Hz)
    # Process structure: "independent", "common_xy", "common_xy_plus_eps"
    process_mode: str = "common_xy_plus_eps"
    # Common XY accel powers (mm^2/s^3); used when process_mode != "independent"
    q_common_x: float = 1e6
    q_common_y: float = 1e6
    # Small per-corner accel power (mm^2/s^3) for flexibility; only used in *_plus_eps
    q_eps: float = 1e2
    # Legacy independent accel level (mm^2/s^3), used when process_mode=="independent"
    q_accel: float = 1.0
    # Measurement covariance 8x8 (mm^2) for [C0x,C0y,C1x,C1y,C2x,C2y,C3x,C3y]
    R: Optional[np.ndarray] = None

class CornerKalman:
    """
    State x (16):
        [ C0x,C0y, C1x,C1y, C2x,C2y, C3x,C3y,  V0x,V0y, V1x,V1y, V2x,V2y, V3x,V3y ]^T

    Discrete constant-velocity model (ZOH):
        A = [[I, dt*I],
             [0,    I]]   with I = I8

    Process noise built from one or more drivers:
      - "independent": 8 independent acceleration drivers (one per velocity comp).
      - "common_xy"  : 2 drivers (a_x, a_y) shared by all x-vels and all y-vels.
      - "common_xy_plus_eps": common drivers + tiny per-corner drivers (epsilon).

    We build Q as sum of contributions:
        Q = Σ G_i * Qw_i * G_i^T
      where G_i = [[0.5*dt^2 * B_i],
                   [    dt    * B_i]]
      and B_i maps inputs to the 8 velocity components.
    """

    def __init__(self, cfg: CornerKFConfig):
        if cfg.fps <= 0:
            raise ValueError("fps must be > 0")
        self.dt = 1.0 / float(cfg.fps)

        # Dimensions
        self.np = 8  # pos comps (C0x,C0y,...C3y)
        self.nv = 8  # vel comps (V0x,V0y,...V3y)
        self.nx = self.np + self.nv
        self.nz = self.np

        I8 = _I(self.np)
        Z8 = np.zeros((self.np, self.np), dtype=np.float64)
        dt = self.dt

        # State transition (discrete)
        self.A = np.block([
            [ I8, dt*I8],
            [ Z8,  I8  ],
        ])

        # Measurement (positions only)
        self.H = np.hstack([I8, np.zeros((self.np, self.nv), dtype=np.float64)])

        # Build process noise structure
        self.process_mode = cfg.process_mode
        self.q_common_x = float(cfg.q_common_x)
        self.q_common_y = float(cfg.q_common_y)
        self.q_eps      = float(cfg.q_eps)
        self.q_accel    = float(cfg.q_accel)

        self._rebuild_Q()

        # Measurement covariance
        if cfg.R is None:
            self.R = 1e-3 * _I(self.nz)
        else:
            self.R = np.array(cfg.R, dtype=np.float64)
            if self.R.shape != (self.nz, self.nz):
                raise ValueError("R must be 8x8 (mm^2)")

        # Init
        self.x = np.zeros((self.nx, 1), dtype=np.float64)
        self.P = 1e3 * _I(self.nx)

        # Scratch
        self._S = np.zeros((self.nz, self.nz), dtype=np.float64)
        self._K = np.zeros((self.nx, self.nz), dtype=np.float64)
        self._y = np.zeros((self.nz, 1), dtype=np.float64)

    # ----- process structure (Q) builders -----
    def _B_common_xy(self) -> np.ndarray:
        """
        Map [a_x, a_y] to velocity vector [V0x,V0y,V1x,V1y,V2x,V2y,V3x,V3y]^T
        B_common is 8x2:
          each corner gets [1,0] on its x row and [0,1] on its y row.
        """
        B = np.zeros((self.nv, 2), dtype=np.float64)
        for i in range(4):
            B[2*i + 0, 0] = 1.0  # x row gets a_x
            B[2*i + 1, 1] = 1.0  # y row gets a_y
        return B  # (8x2)

    def _B_independent(self) -> np.ndarray:
        """
        Independent drivers for each velocity component.
        """
        return _I(self.nv)  # (8x8)

    def _G_from_B(self, B: np.ndarray) -> np.ndarray:
        """
        Build discrete G = [[0.5*dt^2*B],
                            [    dt   *B]]
        """
        dt = self.dt
        top = 0.5 * (dt*dt) * B
        bot = dt * B
        return np.vstack([top, bot])  # (16 x m)

    def _Q_from_B_Qw(self, B: np.ndarray, Qw: np.ndarray) -> np.ndarray:
        G = self._G_from_B(B)
        return G @ Qw @ G.T  # (16x16)

    def _rebuild_Q(self) -> None:
        """
        Recompute self.Q from current process_mode and q_* settings.
        """
        mode = self.process_mode.lower()
        if mode not in ("independent", "common_xy", "common_xy_plus_eps"):
            raise ValueError("process_mode must be one of: independent, common_xy, common_xy_plus_eps")

        Q = np.zeros((self.nx, self.nx), dtype=np.float64)

        if mode == "independent":
            # 8 drivers, diagonal power q_accel
            B_ind = self._B_independent()          # (8x8)
            Qw_ind = self.q_accel * _I(self.nv)    # (8x8)
            Q += self._Q_from_B_Qw(B_ind, Qw_ind)

        elif mode == "common_xy":
            # 2 drivers: a_x, a_y
            Bc = self._B_common_xy()                              # (8x2)
            Qw_c = np.diag([self.q_common_x, self.q_common_y])    # (2x2)
            Q += self._Q_from_B_Qw(Bc, Qw_c)

        else:  # "common_xy_plus_eps"
            # common XY + tiny per-velocity drivers
            Bc = self._B_common_xy()                              # (8x2)
            Qw_c = np.diag([self.q_common_x, self.q_common_y])    # (2x2)
            Q += self._Q_from_B_Qw(Bc, Qw_c)

            if self.q_eps > 0.0:
                B_ind = self._B_independent()                     # (8x8)
                Qw_eps = self.q_eps * _I(self.nv)                 # (8x8)
                Q += self._Q_from_B_Qw(B_ind, Qw_eps)

        self.Q = Q

    # ----- public tuning knobs -----
    def set_process_mode(self, mode: str) -> None:
        self.process_mode = mode
        self._rebuild_Q()

    def set_q_common(self, qx: float, qy: float) -> None:
        self.q_common_x = float(qx)
        self.q_common_y = float(qy)
        self._rebuild_Q()

    def set_q_eps(self, q_eps: float) -> None:
        self.q_eps = float(q_eps)
        self._rebuild_Q()

    def set_q_accel(self, q_accel: float) -> None:
        self.q_accel = float(q_accel)
        self._rebuild_Q()

    def set_R(self, R: np.ndarray) -> None:
        R = np.array(R, dtype=np.float64)
        if R.shape != (self.nz, self.nz):
            raise ValueError("R must be 8x8")
        self.R = R

    # ----- filter core -----
    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z_pos_mm: np.ndarray) -> None:
        z = np.asarray(z_pos_mm, dtype=np.float64).reshape(self.nz, 1)
        self._y = z - (self.H @ self.x)
        self._S = self.H @ self.P @ self.H.T + self.R
        self._K = self.P @ self.H.T @ np.linalg.inv(self._S)
        I = _I(self.nx)
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

    def get_center_mm(self) -> Tuple[float, float]:
        px = self.x[:self.np, 0]  # 8
        cx = float(np.mean(px[0::2]))
        cy = float(np.mean(px[1::2]))
        return cx, cy
