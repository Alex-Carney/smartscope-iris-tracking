# ASCII only
# noise_adaptive_filter.py
from __future__ import annotations
from typing import Optional, Tuple

class NoiseAdaptiveFilter2D:
    """
    Apply filtering ONLY to sub-noise jitter; bypass (publish raw) for real motion.

    Behavior:
      - Compare (x,y) delta vs LAST PUBLISHED (x,y).
      - If delta <= noise floor => FEED filters; once warm, publish FILTERED.
      - If delta >  noise floor => BYPASS (publish raw) and RESET filters.

    Works with any causal per-axis filter that implements:
      process(x: float) -> Optional[float], reset(), (optional) copy()

    Args:
        fx, fy:   per-axis filter instances (independent state)
        use_radial: if True, use radial gate sqrt(dx^2+dy^2) <= floor_mm
                    else use per-axis |dx|<=floor_x_mm and |dy|<=floor_y_mm
        floor_mm, floor_x_mm, floor_y_mm: thresholds in mm
    """

    def __init__(
        self,
        fx,
        fy,
        *,
        use_radial: bool,
        floor_mm: float,
        floor_x_mm: float,
        floor_y_mm: float,
    ):
        self.fx = fx
        self.fy = fy
        self.use_radial = bool(use_radial)
        self.floor_mm = float(floor_mm)
        self.floor_x_mm = float(floor_x_mm)
        self.floor_y_mm = float(floor_y_mm)

        # simple counters (optional)
        self.n_raw_bypass = 0
        self.n_warming = 0
        self.n_filtered = 0

    @staticmethod
    def _in_noise(dx: float, dy: float, use_radial: bool, f_mm: float, fx_mm: float, fy_mm: float) -> bool:
        if use_radial:
            return (dx*dx + dy*dy) ** 0.5 <= f_mm
        else:
            return (abs(dx) <= fx_mm) and (abs(dy) <= fy_mm)

    def process(self, x: float, y: float, last_published: Optional[Tuple[float, float]]):
        """
        Returns: (out_x, out_y, mode)
          mode ∈ {"raw_bypass","warming","filtered"}
        """
        if last_published is None:
            # First packet: publish raw to avoid initial smear; ensure filters start clean
            self.fx.reset(); self.fy.reset()
            self.n_raw_bypass += 1
            return float(x), float(y), "raw_bypass"

        dx = float(x) - float(last_published[0])
        dy = float(y) - float(last_published[1])

        if not self._in_noise(dx, dy, self.use_radial, self.floor_mm, self.floor_x_mm, self.floor_y_mm):
            # Real movement -> publish raw and reset filters
            self.fx.reset(); self.fy.reset()
            self.n_raw_bypass += 1
            return float(x), float(y), "raw_bypass"

        # Noise region: feed filters; publish filtered when warm
        fxv = self.fx.process(float(x))
        fyv = self.fy.process(float(y))
        if fxv is None or fyv is None:
            # warming: still publish raw to avoid lag on tiny dithers
            self.n_warming += 1
            return float(x), float(y), "warming"

        self.n_filtered += 1
        return float(fxv), float(fyv), "filtered"

    def summary(self) -> str:
        total = self.n_raw_bypass + self.n_warming + self.n_filtered
        if total == 0:
            return "NoiseAdaptiveFilter2D: no samples"
        return (
            f"NoiseAdaptiveFilter2D: total={total}, "
            f"raw_bypass={self.n_raw_bypass}, warming={self.n_warming}, filtered={self.n_filtered}"
        )
