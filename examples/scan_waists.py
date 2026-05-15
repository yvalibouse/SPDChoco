#!/usr/bin/env python3
"""
scan_waists.py — 2-D scan of heralding efficiency and brightness over waists.

Maps η and brightness over a (w_p, w_s) grid using the C kernel, then
locates the brightest point on a target-η contour.  Edit the
parameters at the top of the script.
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize    import brentq, minimize_scalar
import matplotlib.pyplot as plt

from chocospdc import compute, plotting, style


# ── Parameters ──────────────────────────────────────────────────────
WP_RANGE   = (25.0, 100.0)         # pump waist range [µm]
WS_RANGE   = (25.0, 100.0)         # signal/idler waist range [µm]
N_W        = 30                    # grid points per axis
N_Z        = 48                    # Gauss–Legendre z-nodes
N_R        = 64                    # GL r-nodes (method 0 only)
N_LAM      = 30                    # spectral grid points (PM direction)
METHOD     = 1                     # 0 = LG mode sum, 1 = double-z analytic
P_MAX_MIN  = 6                     # minimum P_MAX when method = 0
H_TARGET   = 0.80                  # target η for the optimum search
FWHM_NM    = None                  # None → take from config
SAVE_DIR   = "plots"
DATA_DIR   = "scan_data"


def find_optimum(wp, ws, B_map, H_map, h_target):
    """Locate brightest point on the η = h_target contour."""
    Hi = RegularGridInterpolator((ws, wp), H_map,
                                 method="cubic",
                                 bounds_error=False, fill_value=None)
    Bi = RegularGridInterpolator((ws, wp), B_map,
                                 method="cubic",
                                 bounds_error=False, fill_value=None)
    best_B = -1.0
    best_wp = best_ws = None
    wp_fine = np.linspace(wp[0], wp[-1], 200)
    ws_scan = np.linspace(ws[0], ws[-1], 300)
    for wp_v in wp_fine:
        h_vals = np.array([float(Hi((w, wp_v))) - h_target for w in ws_scan])
        for i in range(len(h_vals) - 1):
            if h_vals[i] * h_vals[i + 1] < 0:
                try:
                    ws_root = brentq(
                        lambda w: float(Hi((w, wp_v))) - h_target,
                        ws_scan[i], ws_scan[i + 1])
                    b = float(Bi((ws_root, wp_v)))
                    if b > best_B:
                        best_B, best_wp, best_ws = b, wp_v, ws_root
                except ValueError:
                    pass
    return Hi, Bi, (best_wp, best_ws, best_B) if best_wp is not None else None


def max_eta_slice(wp, ws, Hi, Bi):
    """For each w_p, find the w_s that maximises η, with the corresponding B."""
    wp_fine = np.linspace(wp[0], wp[-1], 300)
    eta_max = np.empty_like(wp_fine)
    ws_opt  = np.empty_like(wp_fine)
    B_at_opt = np.empty_like(wp_fine)
    for j, wp_v in enumerate(wp_fine):
        res = minimize_scalar(
            lambda w: -float(Hi((w, wp_v))),
            bounds=(ws[0], ws[-1]), method="bounded",
            options={"xatol": 0.01})
        ws_v        = float(res.x)
        eta_max[j]  = -float(res.fun)
        ws_opt[j]   = ws_v
        B_at_opt[j] = float(Bi((ws_v, wp_v)))
    return wp_fine, eta_max, ws_opt, B_at_opt


def main():
    style.use()

    result = compute.waist_scan(
        wp_range=WP_RANGE, ws_range=WS_RANGE,
        n_w=N_W, n_z=N_Z, n_r=N_R, n_lam=N_LAM,
        method=METHOD, p_max_min=P_MAX_MIN,
        fwhm_nm=FWHM_NM,
    )

    # Optimum on the target-η contour
    Hi, Bi, best = find_optimum(result["wp"], result["ws"],
                                result["B"], result["H"], H_TARGET)
    if best is not None:
        wp_b, ws_b, B_b = best
        print(f"\n  Best brightness on η = {H_TARGET} contour:")
        print(f"    w_p = {wp_b:.1f} µm,  w_s = {ws_b:.1f} µm,  B = {B_b:.4f}")
    else:
        print(f"\n  No η = {H_TARGET} contour found in the data.")

    # Save raw data
    os.makedirs(DATA_DIR, exist_ok=True)
    tag = f"_fwhm{result['params']['fwhm_nm']:.2f}nm_m{METHOD}"
    np.savez(os.path.join(DATA_DIR, f"scan_{N_W}{tag}.npz"),
             wp=result["wp"], ws=result["ws"],
             S2=result["S2"], B=result["B"], H=result["H"],
             method=METHOD, fwhm_nm=result["params"]["fwhm_nm"])

    # Plots
    os.makedirs(SAVE_DIR, exist_ok=True)
    plotting.waist_scan(result, best=best,
                        save=os.path.join(SAVE_DIR, f"waist_scan{tag}.png"))

    wp_f, eta_m, ws_o, B_o = max_eta_slice(result["wp"], result["ws"], Hi, Bi)
    plotting.waist_max_eta(wp_f, eta_m, ws_o, B_o,
                           fwhm_nm=result["params"]["fwhm_nm"],
                           save=os.path.join(SAVE_DIR, f"waist_maxeta{tag}.png"))
    plt.show()


if __name__ == "__main__":
    main()
