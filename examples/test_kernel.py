#!/usr/bin/env python3
"""
test_kernel.py — Regression test: method 0 (LG sum) vs method 1 (double-z).

Both methods should agree to within ~1e-2 (typically much better) on
S₂, S₁_s, S₁_i and the symmetric heralding η.  The double-z method is
exact in the LG sum but uses a different quadrature path, so this is a
useful end-to-end consistency check.
"""

import time

import numpy as np

import _path  # noqa: F401  — make chocospdc importable from anywhere
from chocospdc          import compute, config
from chocospdc.sellmeier import CRYSTAL_NAMES, TYPE_NAMES


WAISTS    = [(40, 40), (50, 50), (60, 45), (35, 70)]
P_MAX     = 10
N_Z, N_R  = 48, 64
N_LAM     = 50
TOL       = 1e-2


def _eta(s2, s1_s, s1_i):
    return float(np.sqrt(s2 / s1_i * s2 / s1_s)) if s1_s and s1_i else 0.0


def main():
    fwhm_nm = config.FWHM_PUMP * 1e3
    print(f"{'='*70}")
    print(f"  Regression: {CRYSTAL_NAMES[config.CRYSTAL]}, "
          f"{TYPE_NAMES[config.SPDC_TYPE]}, L = {config.L/1e3:.0f} mm")
    print(f"  Pump FWHM = {fwhm_nm:.3f} nm,  N_Z = {N_Z},  N_LAM = {N_LAM}")
    print(f"{'='*70}\n")

    print(f"{'wp':>5} {'ws':>5} {'qty':>6} {'M0 LG':>14} {'M1 dbl-z':>14} {'Δ':>10}")
    print("-" * 70)

    ok = True
    for wp, ws in WAISTS:
        m0 = compute.pair_singles(wp, ws, method=0,
                                  p_max=P_MAX, n_z=N_Z, n_r=N_R, n_lam=N_LAM)
        m1 = compute.pair_singles(wp, ws, method=1,
                                  p_max=P_MAX, n_z=N_Z, n_r=N_R, n_lam=N_LAM)
        for name, v0, v1 in zip(["S2", "S1_s", "S1_i"], m0, m1):
            d = abs(v1 - v0) / max(abs(v0), 1e-30)
            if d > TOL:
                ok = False
            mark = "✓" if d <= TOL else "✗"
            print(f"{wp:5.0f} {ws:5.0f} {name:>6}  "
                  f"{v0:14.6e}  {v1:14.6e}  {d:10.2e} {mark}")
    print()

    print(f"{'wp':>5} {'ws':>5} {'η M0':>10} {'η M1':>10}")
    print("-" * 36)
    for wp, ws in WAISTS:
        m0 = compute.pair_singles(wp, ws, method=0,
                                  p_max=P_MAX, n_z=N_Z, n_r=N_R, n_lam=N_LAM)
        m1 = compute.pair_singles(wp, ws, method=1,
                                  p_max=P_MAX, n_z=N_Z, n_r=N_R, n_lam=N_LAM)
        print(f"{wp:5.0f} {ws:5.0f}  {_eta(*m0):10.6f}  {_eta(*m1):10.6f}")

    print("\n  Timing (100 calls, w_p = w_s = 50):")
    for label, m in [("M0 LG    ", 0), ("M1 dbl-z ", 1)]:
        t0 = time.time()
        for _ in range(100):
            compute.pair_singles(50, 50, method=m,
                                 p_max=P_MAX, n_z=N_Z, n_r=N_R, n_lam=N_LAM)
        print(f"    {label}: {(time.time() - t0) / 100 * 1e3:.2f} ms")

    print(f"\n  {'PASS' if ok else 'FAIL'} (tol = {TOL})")


if __name__ == "__main__":
    main()
