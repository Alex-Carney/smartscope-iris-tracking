from typing import List, Tuple
import time
import numpy as np

class BasicBenchmark:
    def __init__(self):
        self.positions: List[Tuple[float, float]] = []
        self.fps_samples: List[float] = []
        self.last_ts: float = 0.0
        self.frames_processed: int = 0
        self.frames_with_marker: int = 0
        self.frames_skipped_by_noise_gate: int = 0

    def mark_processed(self):
        self.frames_processed += 1

    def mark_with_marker(self):
        self.frames_with_marker += 1

    def mark_skipped(self):
        self.frames_skipped_by_noise_gate += 1

    def tick_fps(self):
        now = time.perf_counter()
        if self.last_ts != 0.0:
            dt = now - self.last_ts
            if dt > 0:
                self.fps_samples.append(1.0 / dt)
        self.last_ts = now

    def add_position(self, x_mm: float, y_mm: float):
        self.positions.append((x_mm, y_mm))

    def print_summary(self, title_suffix: str = ""):
        if not self.positions:
            print("\nNo samples collected.")
        else:
            xs = np.array([p[0] for p in self.positions], dtype=np.float64)
            ys = np.array([p[1] for p in self.positions], dtype=np.float64)
            x_med = float(np.median(xs))
            y_med = float(np.median(ys))
            x_dev = np.abs(xs - x_med)
            y_dev = np.abs(ys - y_med)
            x_std = float(np.std(xs))
            y_std = float(np.std(ys))
            x_max = float(np.max(x_dev))
            y_max = float(np.max(y_dev))
            x_p999 = float(np.percentile(x_dev, 99.9))
            y_p999 = float(np.percentile(y_dev, 99.9))
            print("\n--- Jitter Statistics" + (" " + title_suffix if title_suffix else "") + " ---")
            print(f"Median X: {x_med:.10f}")
            print(f"Median Y: {y_med:.10f}")
            print(f"Std Dev      - X: {x_std:.10f}, Y: {y_std:.10f}")
            print(f"Max dMedian  - X: {x_max:.10f}, Y: {y_max:.10f}")
            print(f"99.9% d      - X: {x_p999:.10f}, Y: {y_p999:.10f}")
            print("\n--- Suggested Noise Floor (99.9 percent of jitter ignored) ---")
            print(f"X: {x_p999:.10f}")
            print(f"Y: {y_p999:.10f}")
        if self.fps_samples:
            avg_fps = float(np.mean(self.fps_samples))
            std_fps = float(np.std(self.fps_samples))
            print("\n--- FPS Statistics ---")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Std Dev FPS: {std_fps:.2f}")
        else:
            print("\nNo FPS data collected.")
        print("\n--- Frame Accounting ---")
        print(f"Frames processed: {self.frames_processed}")
        print(f"Frames with marker: {self.frames_with_marker}")
        print(f"Frames skipped by noise gate: {self.frames_skipped_by_noise_gate}")
        print(f"Samples kept: {len(self.positions)}")