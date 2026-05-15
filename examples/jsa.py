#!/usr/bin/env python3
"""
jsa.py — Joint Spectral Amplitude and Schmidt decomposition.

Computes the complex C₀₀(λ_s, λ_i) on a 2-D grid with the broadband
pump envelope α(λ_p) applied, then performs Schmidt decomposition to
extract the Schmidt number K.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP          = config.w0_p
WS          = config.w0_s
WI          = config.w0_i
N_LAM       = 150
N_Z         = 64
HALF_RANGE  = 0.002        # ±half-width around λ_s0 [µm]
FWHM_NM     = None         # None → take from config
DATA_DIR    = "data"
SAVE_DIR    = "plots"


def main():
    style.use()

    result = compute.jsa(
        wp=WP, ws=WS, wi=WI,
        n_lam=N_LAM, n_z=N_Z,
        half_range=HALF_RANGE,
        fwhm_nm=FWHM_NM,
    )
    print(f"  Schmidt number  K = {result['K']:.3f}  "
          f"(integration {result['time_s']:.2f}s)")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "jsa.npz"),
             C00=result["C00"], C00_norm=result["C00_norm"],
             lam_s=result["lam_s"] * 1e3, lam_i=result["lam_i"] * 1e3,
             schmidt_values=result["schmidt_values"], K=result["K"],
             **result["params"])

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = f"wp{WP:.0f}_ws{WS:.0f}"
    plotting.jsa    (result, save=os.path.join(SAVE_DIR, f"jsa_{tag}.png"))
    plotting.schmidt(result, save=os.path.join(SAVE_DIR, f"schmidt_{tag}.png"))
    print(f"  Plots saved in {SAVE_DIR}/")
    plt.show()


if __name__ == "__main__":
    main()
