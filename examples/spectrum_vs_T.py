#!/usr/bin/env python3
"""
spectrum_vs_T.py — Signal spectrum vs crystal temperature.

A single computation produces both a waterfall plot (coarse T grid) and
an optional fine-grained heatmap.  Toggle the flags in the parameters
block to choose what to plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import _path  # noqa: F401  — make chocospdc importable from anywhere
from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP            = config.w0_p
WS            = config.w0_s
LAM_RANGE     = (1.540, 1.560)      # signal-wavelength range [µm]
N_LAM         = 400                  # signal grid resolution
FWHM_NM       = None                 # None → take from config
N_Z           = 48

# Waterfall: coarse T grid
T_WATERFALL   = (config.T - 20.0, config.T + 20.0)   # °C
DT_WATERFALL  = 4.0                                  # °C step

# Heatmap: fine T grid
T_HEATMAP     = (config.T - 20.0, config.T + 40.0)   # °C
DT_HEATMAP    = 0.2                                  # °C step

PLOT_WATERFALL = True
PLOT_HEATMAP   = True
LOG_HEATMAP    = False

DATA_DIR  = "data"
SAVE_DIR  = "plots"


def main():
    style.use()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    if PLOT_WATERFALL:
        res_w = compute.c00_heatmap(
            T_range=T_WATERFALL, dt=DT_WATERFALL,
            lam_range=LAM_RANGE, n_lam=N_LAM,
            wp=WP, ws=WS, n_z=N_Z, fwhm_nm=FWHM_NM,
        )
        np.savez(os.path.join(DATA_DIR, "spectrum_waterfall.npz"),
                 T=res_w["T"], lam_nm=res_w["lam_nm"],
                 spectra=res_w["spectra"], **res_w["params"])
        f = os.path.join(SAVE_DIR,
                         f"spectrum_waterfall_fwhm{res_w['params']['fwhm_nm']:.2f}nm.png")
        plotting.spectrum_waterfall(res_w, save=f)
        print(f"  Waterfall plot → {f}")

    if PLOT_HEATMAP:
        res_h = compute.c00_heatmap(
            T_range=T_HEATMAP, dt=DT_HEATMAP,
            lam_range=LAM_RANGE, n_lam=N_LAM,
            wp=WP, ws=WS, n_z=N_Z, fwhm_nm=FWHM_NM,
        )
        np.savez(os.path.join(DATA_DIR, "spectrum_heatmap.npz"),
                 T=res_h["T"], lam_nm=res_h["lam_nm"],
                 spectra=res_h["spectra"], **res_h["params"])
        f = os.path.join(SAVE_DIR,
                         f"spectrum_heatmap_fwhm{res_h['params']['fwhm_nm']:.2f}nm.png")
        plotting.spectrum_heatmap(res_h, save=f, log=LOG_HEATMAP)
        print(f"  Heatmap plot   → {f}")

    plt.show()


if __name__ == "__main__":
    main()
