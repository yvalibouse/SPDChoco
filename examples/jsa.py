#!/usr/bin/env python3
"""
jsa.py — Joint Spectral Amplitude and Schmidt decomposition.

Computes the complex C₀₀(λ_s, λ_i) on a 2-D grid with the broadband
pump envelope α(λ_p) applied.  Optionally applies a symmetric bandpass
filter on both arms (same filter spec as :mod:`brightness_vs_T`) before
Schmidt decomposition, so the reported K is the filtered Schmidt number
an experimentalist would measure through identical filters.

If you want the *intrinsic* JSA of the source, keep ``FILTER_SHAPE =
"none"``.  In that case ``HALF_RANGE`` (the integration window) acts as
the de facto rect filter on both arms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import _path  # noqa: F401  — make chocospdc importable from anywhere
from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP          = config.w0_p
WS          = config.w0_s
WI          = config.w0_i
N_LAM       = 150
N_Z         = 64
HALF_RANGE  = 0.004        # numerical half-width of (λ_s, λ_i) grid [µm]
FWHM_NM     = None         # None → take from config

# ── Symmetric filter (same in both arms) ────────────────────────────
FILTER_SHAPE     = "rect"  # "none" | "rect" | "gauss" | "file"
FILTER_CENTER_NM = None    # None → λ_s0 from config
FILTER_BW_NM     = 4.0     # ignored if FILTER_SHAPE == "none"
FILTER_FILE      = None    # path to text file (2 cols: λ_nm, T_percent) for shape="file"

DATA_DIR    = "data"
SAVE_DIR    = "plots"


def main():
    style.use()

    result = compute.jsa(
        wp=WP, ws=WS, wi=WI,
        n_lam=N_LAM, n_z=N_Z,
        half_range=HALF_RANGE,
        fwhm_nm=FWHM_NM,
        filter_center_nm=FILTER_CENTER_NM,
        filter_bw_nm=FILTER_BW_NM,
        filter_shape=FILTER_SHAPE,
        filter_file=FILTER_FILE,
    )
    if FILTER_SHAPE == "none":
        tag_f = "no filter"
    elif FILTER_SHAPE == "file":
        tag_f = f"file {os.path.basename(str(FILTER_FILE))} (both arms)"
    else:
        tag_f = f"{FILTER_SHAPE} {FILTER_BW_NM:.3g} nm (both arms)"
    print(f"  Schmidt number  K = {result['K']:.3f}  "
          f"({tag_f})   integration {result['time_s']:.2f}s")

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
