#!/usr/bin/env python3
"""
brightness_vs_T.py — Brightness vs T with a tunable spectral filter.

For fixed focusing (w_p, w_s) and pump bandwidth, computes the signal-arm
brightness as a function of T, integrated over one or more filter
shapes/widths.  Useful for matching simulations to a measured fibre +
bandpass detection setup.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP             = config.w0_p
WS             = config.w0_s

T_RANGE        = (config.T, config.T + 25.0)        # °C
DT             = 0.05                                # °C step

FILTER_CENTER  = config.lambda0_s * 1e3              # filter centre [nm]
FILTER_BW_NM   = [4.0]                               # filter BW(s) [nm]; list ⇒ overlay
FILTER_SHAPE   = "rect"                              # rect | gauss | none

LAM_RANGE_NM   = None      # None → auto from filter centre + max BW
N_LAM          = 400
N_Z            = 64
FWHM_NM        = None      # None → take from config

NORMALIZE_B    = True      # plot each B(T) curve normalised to its peak
LOG_AXIS       = False
T_REF          = None      # reference T for spectrum panel (None = peak)

DATA_DIR  = "data"
SAVE_DIR  = "plots"


def main():
    style.use()

    result = compute.brightness_vs_T(
        T_range=T_RANGE, dt=DT,
        filter_center_nm=FILTER_CENTER,
        filter_bw_nm=FILTER_BW_NM,
        filter_shape=FILTER_SHAPE,
        wp=WP, ws=WS,
        lam_range_nm=LAM_RANGE_NM, n_lam=N_LAM,
        n_z=N_Z, fwhm_nm=FWHM_NM,
    )

    # Peak summary
    print("\n  Peak brightness:")
    for iF, bw in enumerate(np.atleast_1d(result["filter_bw_nm"])):
        iT = int(np.argmax(result["brightness"][iF]))
        tag = "no filter" if FILTER_SHAPE == "none" else f"BW = {bw:.3g} nm"
        print(f"    {tag:>18s}   T_peak = {result['T'][iT]:7.3f}°C  "
              f"B_peak = {result['brightness'][iF, iT]:.4e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "brightness_vs_T.npz"),
             T=result["T"], lam_nm=result["lam_nm"],
             spectra=result["spectra"], brightness=result["brightness"],
             filter_bw_nm=result["filter_bw_nm"],
             filter_center_nm=result["filter_center_nm"],
             filter_shape=result["filter_shape"],
             **result["params"])

    os.makedirs(SAVE_DIR, exist_ok=True)
    bw_tag = (FILTER_SHAPE if FILTER_SHAPE == "none"
              else f"{FILTER_SHAPE}_" + "-".join(f"{b:.3g}" for b in FILTER_BW_NM) + "nm")
    fname = os.path.join(SAVE_DIR,
                         f"brightness_vs_T_wp{WP:.0f}_ws{WS:.0f}_{bw_tag}.png")
    plotting.brightness_vs_T(result, save=fname,
                             normalize=NORMALIZE_B, log=LOG_AXIS, t_ref=T_REF)
    print(f"  Plot   → {fname}")
    plt.show()


if __name__ == "__main__":
    main()
