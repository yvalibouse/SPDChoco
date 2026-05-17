#!/usr/bin/env python3
"""
hom_vs_T.py — HOM dip vs crystal temperature: simulation + plots.

For each temperature, computes the complex JSA, derives the HOM
visibility curve V(τ) and the Schmidt number K(T).  Fits the deepest
dip to a Gaussian and reports the FWHM and visibility.

A symmetric bandpass filter (same in both arms) can be applied to model
identical interference filters in front of each detector.  Default
``FILTER_SHAPE = "none"`` leaves the JSA unfiltered (so ``HALF_RANGE``
acts as the de facto rect filter).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import _path  # noqa: F401  — make chocospdc importable from anywhere
from chocospdc import compute, plotting, style, config


# ── Parameters ──────────────────────────────────────────────────────
WP          = config.w0_p
WS          = config.w0_s

T_RANGE     = (35.0, 60.0)         # °C
DT          = 1.0                  # °C step

N_LAM       = 120                  # spectral grid per axis
N_Z         = 64                   # GL z-nodes
HALF_RANGE  = 0.008                # numerical half-width [µm]

TAU_MAX_PS  = 10.0                 # τ half-range [ps]
N_TAU       = 401                  # τ grid points (odd → τ = 0 on grid)

FWHM_NM     = None                 # None → take from config

# ── Symmetric filter (same in both arms) ────────────────────────────
FILTER_SHAPE     = "file"          # "none" | "rect" | "gauss" | "file"
FILTER_CENTER_NM = None            # None → λ_s0 from config
FILTER_BW_NM     = 4.0             # ignored if FILTER_SHAPE == "none"
FILTER_FILE      = 'filter/Thorlabs-FBH1550-4.txt'            # path to text file (2 cols: λ_nm, T_percent) for shape="file"

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
        d = np.load(DATA_NPZ, allow_pickle=True)
        result = dict(
            T=d["T"], tau_ps=d["tau_ps"],
            V_tau=d["V_tau"], V0=d["V0"], dip=d["dip"],
            K_eff=d["K_eff"], i_tau0=int(d["i_tau0"]),
            lam_nm=d["lam_nm"],
            has_filter=bool(d.get("has_filter", False)),
            params=dict(wp=float(d["wp"]), ws=float(d["ws"]),
                        fwhm_nm=float(d["fwhm_nm"]),
                        filter_center_nm=(None if d["filter_center_nm"].item() is None
                                          else float(d["filter_center_nm"])),
                        filter_bw_nm=float(d["filter_bw_nm"]),
                        filter_shape=str(d["filter_shape"])),
        )
    else:
        result = compute.hom_vs_T(
            T_range=T_RANGE, dt=DT,
            wp=WP, ws=WS,
            n_lam=N_LAM, n_z=N_Z,
            half_range=HALF_RANGE,
            tau_max_ps=TAU_MAX_PS, n_tau=N_TAU,
            fwhm_nm=FWHM_NM,
            filter_center_nm=FILTER_CENTER_NM,
            filter_bw_nm=FILTER_BW_NM,
            filter_shape=FILTER_SHAPE,
            filter_file=FILTER_FILE,
        )
        np.savez(
            DATA_NPZ,
            T=result["T"], tau_ps=result["tau_ps"],
            V_tau=result["V_tau"], V0=result["V0"],
            dip=result["dip"], K_eff=result["K_eff"],
            i_tau0=result["i_tau0"], lam_nm=result["lam_nm"],
            has_filter=result["has_filter"],
            wp=WP, ws=WS,
            fwhm_nm=result["params"]["fwhm_nm"],
            filter_center_nm=np.array(result["params"]["filter_center_nm"], dtype=object),
            filter_bw_nm=result["params"]["filter_bw_nm"] or 0.0,
            filter_shape=result["params"]["filter_shape"],
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
    if result["has_filter"]:
        if FILTER_SHAPE == "file":
            tag += f"_filterfile_{os.path.basename(str(FILTER_FILE))}"
        else:
            tag += f"_filter{FILTER_SHAPE}{FILTER_BW_NM:.2g}nm"
    plotting.hom_best_dip(
        result, fit=fit,
        save=os.path.join(SAVE_DIR, f"hom_best_dip_{tag}.png"))
    plotting.hom_visibility_vs_T(
        result,
        save=os.path.join(SAVE_DIR, f"hom_visibility_vs_T_{tag}.png"))
    plt.show()


if __name__ == "__main__":
    main()
