from typing import Optional, Tuple

class NoiseGate:
    def __init__(self, enable: bool = False, use_radial: bool = False, floor_mm: float = 0.05, floor_x_mm: float = 0.05, floor_y_mm: float = 0.05):
        self.enable = enable
        self.use_radial = use_radial
        self.floor_mm = floor_mm
        self.floor_x_mm = floor_x_mm
        self.floor_y_mm = floor_y_mm
        self.last_sent: Optional[Tuple[float, float]] = None

    def should_send(self, x_mm: float, y_mm: float) -> bool:
        if not self.enable:
            self.last_sent = (x_mm, y_mm)
            return True
        if self.last_sent is None:
            self.last_sent = (x_mm, y_mm)
            return True
        dx = x_mm - self.last_sent[0]
        dy = y_mm - self.last_sent[1]
        if self.use_radial:
            dr = (dx * dx + dy * dy) ** 0.5
            if dr <= self.floor_mm:
                return False
        else:
            if abs(dx) <= self.floor_x_mm and abs(dy) <= self.floor_y_mm:
                return False
        self.last_sent = (x_mm, y_mm)
        return True