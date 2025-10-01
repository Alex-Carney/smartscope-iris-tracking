# ASCII only
# smartscope_aruco/filters/visualize.py
import numpy as np
import matplotlib.pyplot as plt

from boxcar import Boxcar
from ema import EMA
from sgq import CausalSavGol

def build_filter(config):
    kind = config["kind"].lower()
    if kind == "boxcar":
        return Boxcar(config["N"])
    elif kind == "ema":
        return EMA(config["alpha"])
    elif kind in ("sg", "sgc", "sg_causal"):
        return CausalSavGol(config["window"], config["polyorder"])
    else:
        raise SystemExit(f"Unknown filter kind: {config['kind']}")

def main():
    # Hardcoded parameters - modify these values directly in the script
    config = {
        "kind": "boxcar",           # boxcar | ema | sg (causal)
        "N": 3,                     # boxcar length
        "alpha": 0.1,              # EMA alpha
        "window": 9,                # SG window length
        "polyorder": 2,             # SG poly order
        "fps": 90.0,                # fps to show delay in ms
        "max_lags": 200,            # max lags for IIR plot
        "tol": 1e-4                 # tail tolerance for IIR
    }

    filt = build_filter(config)
    lags, w = filt.impulse_response(max_lags=config["max_lags"], tol=config["tol"])
    delay_samp = filt.effective_delay_samples()
    delay_ms = 1000.0 * delay_samp / config["fps"] if config["fps"] > 0 else None

    # plot bars at lags 0, -1, -2, ...
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, w, width=0.8, align="center")
    ax.set_xlabel("Lag (samples)   0=now, -1=1 sample ago, ...")
    ax.set_ylabel("Weight")
    ttl = str(filt)
    if delay_ms is not None:
        ttl += f"   (delay ≈ {delay_ms:.2f} ms)"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)
    # tidy axes to include entire tail
    ax.set_xlim(min(lags) - 1, 1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
