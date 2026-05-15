#!/usr/bin/env python3
"""
modes.py — Mode-resolved signal/idler spectra with broadband pump.

Computes P_{0,p_i}(λ_s) and P_{p_s,0}(λ_s) for all LG radial-mode pairs
up to P_MAX, then prints η_s, η_i, η.  Edit the parameters block.
"""

import os
import matplotlib.pyplot as plt

from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP        = config.w0_p       # pump waist [µm]
WS        = config.w0_s       # signal/idler waist [µm]
P_MAX     = 10                # max LG radial order  (0 = auto from ξ)
N_Z       = 256               # GL z-nodes
N_R       = 64                # GL r-nodes
N_LAM     = 0                 # spectral grid (0 = auto from config)
N_I       = 0                 # λ_i quadrature pts (0 = auto)
FWHM_NM   = None              # None → take from config
SAVE_DIR  = "plots"


def main():
    style.use()

    result = compute.modes_spectrum(
        wp=WP, ws=WS,
        p_max=P_MAX,
        n_z=N_Z, n_r=N_R,
        n_lam=N_LAM, n_i=N_I,
        fwhm_nm=FWHM_NM,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = os.path.join(SAVE_DIR, f"modes_wp{WP:.0f}_ws{WS:.0f}.png")
    plotting.modes_spectrum(result, save=fname)
    print(f"\n  Plot → {fname}")
    plt.show()


if __name__ == "__main__":
    main()
