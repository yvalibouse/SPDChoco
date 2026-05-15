#!/usr/bin/env python3
"""
hom_vs_T.py — HOM dip vs crystal temperature: simulation + plots.

For each temperature, computes the complex JSA, derives the HOM
visibility curve V(τ) and the Schmidt number K(T).  Fits the deepest
dip to a Gaussian and reports the FWHM and visibility.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP          = config.w0_p
WS          = config.w0_s

T_RANGE     = (23.5, 50.0)         # °C
DT          = 1.0                  # °C step

N_LAM       = 120                  # spectral grid per axis
N_Z         = 64                   # GL z-nodes
HALF_RANGE  = 0.002                # spectral half-width [µm]

TAU_MAX_PS  = 10.0                 # τ half-range [ps]
N_TAU       = 401                  # τ grid points (odd → τ = 0 on grid)

FWHM_NM     = None                 # None → take from config

REUSE       = False                # if True, skip sim and re-plot from .npz
DATA_NPZ    = "data/hom_vs_T.npz"
SAVE_DIR    = "plots"


def dip_model(x, c, a, x0, sigma):
    return c - a * np.exp(-(x - x0)**2 / (2.0 * sigma**2))


def fit_best_dip(tau_fs, dip):
    """Fit the half-amplitude dip with a Gaussian; return (popt, perr)."""
    y = dip / 2.0
    mask = y < 0.51
    if mask.sum() < 8:
        thr  = y.min() + 0.5 * (np.median(y) - y.min())
        mask = y < thr
    p0 = [float(np.median(y)),
          float(np.median(y)) - float(y.min()),
          float(tau_fs[np.argmin(y)]),
          (tau_fs.max() - tau_fs.min()) / 20.0]
    popt, pcov = curve_fit(dip_model, tau_fs[mask], y[mask], p0=p0)
    return popt, np.sqrt(np.diag(pcov))


def main():
    style.use()
    os.makedirs(os.path.dirname(DATA_NPZ) or ".", exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    if REUSE and os.path.isfile(DATA_NPZ):
        print(f"  --reuse: loading {DATA_NPZ}")
        d = np.load(DATA_NPZ)
        result = dict(
            T=d["T"], tau_ps=d["tau_ps"],
            V_tau=d["V_tau"], V0=d["V0"], dip=d["dip"],
            K_eff=d["K_eff"], i_tau0=int(d["i_tau0"]),
            lam_nm=d["lam_nm"],
            params=dict(wp=float(d["wp"]), ws=float(d["ws"]),
                        fwhm_nm=float(d["fwhm_nm"])),
        )
    else:
        result = compute.hom_vs_T(
            T_range=T_RANGE, dt=DT,
            wp=WP, ws=WS,
            n_lam=N_LAM, n_z=N_Z,
            half_range=HALF_RANGE,
            tau_max_ps=TAU_MAX_PS, n_tau=N_TAU,
            fwhm_nm=FWHM_NM,
        )
        np.savez(
            DATA_NPZ,
            T=result["T"], tau_ps=result["tau_ps"],
            V_tau=result["V_tau"], V0=result["V0"],
            dip=result["dip"], K_eff=result["K_eff"],
            i_tau0=result["i_tau0"], lam_nm=result["lam_nm"],
            wp=WP, ws=WS,
            fwhm_nm=result["params"]["fwhm_nm"],
        )
        print(f"  Saved → {DATA_NPZ}")

    # Best dip + fit
    iT_best = int(np.argmax(result["V0"]))
    T_best  = float(result["T"][iT_best])
    tau_fs  = result["tau_ps"] * 1e3
    print(f"\n  Best dip at T = {T_best:.2f}°C   V(0) = {result['V0'][iT_best]:+.4f}")

    try:
        popt, perr = fit_best_dip(tau_fs, result["dip"][iT_best])
        fit = (popt, perr)
    except RuntimeError as e:
        print(f"  Fit failed: {e}")
        fit = None

    tag = (f"wp{WP:.0f}_ws{WS:.0f}_L{config.L/1e3:.0f}mm_"
           f"fwhm{result['params']['fwhm_nm']:.2f}nm")
    plotting.hom_best_dip(
        result, fit=fit,
        save=os.path.join(SAVE_DIR, f"hom_best_dip_{tag}.png"))
    plotting.hom_visibility_vs_T(
        result,
        save=os.path.join(SAVE_DIR, f"hom_visibility_vs_T_{tag}.png"))
    plt.show()


if __name__ == "__main__":
    main()
