# ASCII only
# frame_accounting.py
from __future__ import annotations
from typing import Dict, List, Optional
import time
import numpy as np
import matplotlib.pyplot as plt

class TimeAccounting:
    """
    Per-frame timing for named processing steps.

    Usage:
        fa = TimeAccounting(out_path="frame_accounting.png")
        fa.start_frame()
        ... work ...
        fa.mark("decode")
        ... work ...
        fa.mark("undistort")
        ... work ...
        fa.end_frame()
        ...
        fa.finish()
    """

    def __init__(
        self,
        out_path: str = "frame_accounting.png",
        budgets_fps: List[float] = (30.0, 60.0, 90.0),
        title: str = "Frame Accounting (avg durations)",
        xtick_rotation: float = 45.0,        # NEW: rotate x tick labels
        wrap_colon: bool = True,             # NEW: split "group:step" onto 2 lines
    ):
        self.out_path = out_path
        self.budgets_fps = list(budgets_fps)
        self.title = title
        self.xtick_rotation = float(xtick_rotation)
        self.wrap_colon = bool(wrap_colon)

        self._t0: Optional[float] = None
        self._t_last: Optional[float] = None

        # Per-step durations across frames
        self._durations: Dict[str, List[float]] = {}
        self._step_order: List[str] = []   # first-seen order for nice plotting

        # Per-frame totals (t_end - t0)
        self._frame_total_ms: List[float] = []

    # ---------- instrumentation ----------
    def start_frame(self) -> None:
        t = time.perf_counter()
        self._t0 = t
        self._t_last = t

    def mark(self, name: str) -> None:
        """
        Record elapsed time since last mark (ms) for `name`.
        First time a name is used it is added to plotting order.
        """
        if self._t_last is None:
            return
        t = time.perf_counter()
        dt_ms = (t - self._t_last) * 1000.0
        if name not in self._durations:
            self._durations[name] = []
            self._step_order.append(name)
        self._durations[name].append(dt_ms)
        self._t_last = t

    def end_frame(self) -> None:
        """
        Store total frame time (t_now - t0).
        """
        if self._t0 is None:
            return
        t = time.perf_counter()
        total_ms = (t - self._t0) * 1000.0
        self._frame_total_ms.append(total_ms)
        self._t0 = None
        self._t_last = None

    # ---------- reporting ----------
    def _avg_step_times(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._durations.items() if len(v) > 0}

    def _avg_total(self) -> float:
        return float(np.mean(self._frame_total_ms)) if self._frame_total_ms else float("nan")

    def _format_labels(self, labels: List[str]) -> List[str]:
        if not self.wrap_colon:
            return labels
        out = []
        for s in labels:
            # split at first colon -> "group:\nstep"
            if ":" in s:
                a, b = s.split(":", 1)
                out.append(f"{a}:\n{b}")
            else:
                out.append(s)
        return out

    def finish(self) -> Optional[str]:
        if not self._frame_total_ms:
            print("FrameAccounting: no frames recorded.")
            return None

        avg_steps = self._avg_step_times()
        step_labels = [s for s in self._step_order if s in avg_steps]
        step_means = [avg_steps[s] for s in step_labels]
        total_mean = self._avg_total()

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(14, 6))  # a bit wider helps
        x = np.arange(len(step_labels) + 1)  # steps + TOTAL
        heights = step_means + [sum(step_means)]  # show sum of steps as "Total"
        bars = ax.bar(x, heights, width=0.7)

        # Annotate bars
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width()/2.0,
                h,
                f"{h:.2f} ms",
                ha="center", va="bottom", fontsize=9
            )

        # Budget lines (30/60/90 fps)
        for fps in self.budgets_fps:
            ms = 1000.0 / fps
            ax.axhline(ms, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(x[-1] + 0.35, ms, f"{int(fps)} FPS ({ms:.2f} ms)", color="tab:red",
                    va="center", fontsize=9)

        # Pretty axes
        labels_fmt = self._format_labels(step_labels + ["Total"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels_fmt,
            rotation=self.xtick_rotation,
            ha="right",
            rotation_mode="anchor"
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylabel("Average duration [ms]")
        ax.set_title(self.title + f"\nAvg frame total = {total_mean:.2f} ms")
        ax.grid(True, axis="y", alpha=0.3)

        # Extra bottom margin so rotated labels don't clip
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        print(f"FrameAccounting: saved {self.out_path}")

        # Console summary
        print("\n--- FrameAccounting: mean durations (ms) ---")
        for s, m in zip(step_labels, step_means):
            print(f"{s:>18s}: {m:8.3f} ms")
        print(f"{'Total (sum steps)':>18s}: {sum(step_means):8.3f} ms")
        print(f"{'Avg frame total':>18s}: {total_mean:8.3f} ms")

        return self.out_path
