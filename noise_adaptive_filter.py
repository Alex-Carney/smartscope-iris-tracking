# ASCII only
# noise_adaptive_filter.py
from __future__ import annotations
from typing import Optional, Tuple

class NoiseAdaptiveDualFloor2D:
    """
    Two-layer noise logic for 2D streams:

      - Layer 1 (T1): "raw noise floor"
          If delta vs LAST PUBLISHED > T1 => real motion:
              -> publish RAW
              -> RESET filters (no smear)
          Else (<= T1) => noise region:
              -> FEED filters, and when warm:
                 * compute filtered delta vs last published
                 * apply Layer 2 (T2) gate:
                     if <= T2: DROP (no publish)
                     else    : publish FILTERED
              -> while warming:
                     either DROP (recommended) or publish RAW (configurable)

    Works with any causal per-axis filter implementing:
        process(x: float) -> Optional[float]
        reset()
        (optional) copy()

    Thresholds:
        - use_radial=True -> compare sqrt(dx^2+dy^2) to floors
        - else            -> compare per-axis |dx|, |dy|

    Counters:
        n_raw_bypass: real-motion raw publishes (above T1)
        n_filtered  : filtered publishes (noise region, above T2)
        n_warm_drop : noise region, warming, dropped
        n_filt_drop : noise region, filtered, but under T2 -> dropped
    """

    def __init__(
        self,
        fx,
        fy,
        *,
        use_radial: bool,
        # Layer 1 thresholds (raw floor)
        floor1_mm: float,
        floor1_x_mm: float,
        floor1_y_mm: float,
        # Layer 2 thresholds (post-filter drop floor)
        floor2_mm: float,
        floor2_x_mm: float,
        floor2_y_mm: float,
        drop_during_warmup: bool = True,
    ):
        self.fx = fx
        self.fy = fy
        self.use_radial = bool(use_radial)

        self.floor1_mm = float(floor1_mm)
        self.floor1_x_mm = float(floor1_x_mm)
        self.floor1_y_mm = float(floor1_y_mm)

        self.floor2_mm = float(floor2_mm)
        self.floor2_x_mm = float(floor2_x_mm)
        self.floor2_y_mm = float(floor2_y_mm)

        self.drop_during_warmup = bool(drop_during_warmup)

        # Counters
        self.n_raw_bypass = 0
        self.n_filtered = 0
        self.n_warm_drop = 0
        self.n_filt_drop = 0

    @staticmethod
    def _under_floor(dx: float, dy: float, use_radial: bool, f_mm: float, fx_mm: float, fy_mm: float) -> bool:
        if use_radial:
            return (dx * dx + dy * dy) ** 0.5 <= f_mm
        else:
            return (abs(dx) <= fx_mm) and (abs(dy) <= fy_mm)

    def process(
        self,
        x: float,
        y: float,
        last_published: Optional[Tuple[float, float]],
    ) -> Tuple[bool, Optional[float], Optional[float], str]:
        """
        Decide what to publish.

        Returns:
            (publish, out_x, out_y, mode)

            publish == True  -> out_x, out_y are not None (send)
            publish == False -> out_x, out_y are None   (drop)

            mode in:
                "raw_bypass"   : real motion above T1, published RAW
                "filtered"     : noise region, filtered above T2, published FILTERED
                "warming_drop" : noise region, warming, dropped
                "filtered_drop": noise region, filtered under T2, dropped
        """
        if last_published is None:
            # First publish is RAW and we start filters clean
            self.fx.reset(); self.fy.reset()
            self.n_raw_bypass += 1
            return True, float(x), float(y), "raw_bypass"

        last_x, last_y = float(last_published[0]), float(last_published[1])
        dx = float(x) - last_x
        dy = float(y) - last_y

        # LAYER 1: raw floor (T1)
        if not self._under_floor(dx, dy, self.use_radial, self.floor1_mm, self.floor1_x_mm, self.floor1_y_mm):
            # Real motion -> publish RAW, reset filters
            self.fx.reset(); self.fy.reset()
            self.n_raw_bypass += 1
            return True, float(x), float(y), "raw_bypass"

        # In noise region: feed filters
        fxv = self.fx.process(float(x))
        fyv = self.fy.process(float(y))

        if fxv is None or fyv is None:
            # Warming phase
            if self.drop_during_warmup:
                self.n_warm_drop += 1
                return False, None, None, "warming_drop"
            else:
                # Optionally leak raw during warmup
                self.n_raw_bypass += 1
                return True, float(x), float(y), "raw_bypass"

        # LAYER 2: post-filter floor (T2)
        dxf = float(fxv) - last_x
        dyf = float(fyv) - last_y
        if self._under_floor(dxf, dyf, self.use_radial, self.floor2_mm, self.floor2_x_mm, self.floor2_y_mm):
            self.n_filt_drop += 1
            return False, None, None, "filtered_drop"

        # Filtered movement above T2 -> publish filtered
        self.n_filtered += 1
        return True, float(fxv), float(fyv), "filtered"

    def summary(self) -> str:
        total = self.n_raw_bypass + self.n_filtered + self.n_warm_drop + self.n_filt_drop
        if total == 0:
            return "NoiseAdaptiveDualFloor2D: no samples"
        return (
            f"NoiseAdaptiveDualFloor2D: total={total}, "
            f"raw_bypass={self.n_raw_bypass}, filtered={self.n_filtered}, "
            f"warming_drop={self.n_warm_drop}, filtered_drop={self.n_filt_drop}"
        )
