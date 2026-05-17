#!/usr/bin/env python3
"""
modes.py — Mode-resolved signal/idler spectra with broadband pump.

Computes P_{0,p_i}(λ_s) and P_{p_s,0}(λ_s) for all LG radial-mode pairs
up to P_MAX, then prints η_s, η_i, η.  Edit the parameters block.
"""

import os
import matplotlib.pyplot as plt

import _path  # noqa: F401  — make chocospdc importable from anywhere
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

# ── Symmetric filter (same in both arms) ────────────────────────────
FILTER_SHAPE     = "none"     # "none" | "rect" | "gauss" | "file"
FILTER_CENTER_NM = None       # None → λ_s0 from config (rect/gauss only)
FILTER_BW_NM     = 4.0        # ignored for "none" / "file"
FILTER_FILE      = 'filter/Thorlabs-FBH1550-4.txt'       # path to text file (2 cols: λ_nm, T_percent)

SAVE_DIR  = "plots"


def main():
    style.use()

    result = compute.modes_spectrum(
        wp=WP, ws=WS,
        p_max=P_MAX,
        n_z=N_Z, n_r=N_R,
        n_lam=N_LAM, n_i=N_I,
        fwhm_nm=FWHM_NM,
        filter_center_nm=FILTER_CENTER_NM,
        filter_bw_nm=FILTER_BW_NM,
        filter_shape=FILTER_SHAPE,
        filter_file=FILTER_FILE,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = f"_wp{WP:.0f}_ws{WS:.0f}"
    if result["has_filter"]:
        if FILTER_SHAPE == "file":
            tag += f"_filterfile_{os.path.basename(str(FILTER_FILE))}"
        else:
            tag += f"_filter{FILTER_SHAPE}{FILTER_BW_NM:.2g}nm"
    fname = os.path.join(SAVE_DIR, f"modes{tag}.png")
    plotting.modes_spectrum(result, save=fname)
    print(f"\n  Plot → {fname}")
    plt.show()


if __name__ == "__main__":
    main()
