"""
chocospdc.compute — High-level simulation API.

Every "compute_X" used to be a 250-line standalone script.  Here they
are functions that take parameters and return result dicts; the scripts
are reduced to a few lines that call these and plot.

Functions
---------
    waist_scan          2-D scan of S2, η, brightness over (w_p, w_s)
    modes_spectrum      Mode-resolved P_{0,p_i}, P_{p_s,0}(λ_s)
    jsa                 Complex C₀₀(λ_s, λ_i), normalised, Schmidt decomp
    c00_heatmap         |C₀₀|² on a (T × λ_s × λ_i) grid → signal spectra
    hom_vs_T            HOM visibility V(τ) at every temperature
    pair_singles        Single waist pair: (S2, S1_s, S1_i)
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from . import config as cfg
from . import grids
from . import quadrature as qd
from .beams import beam_w, inv_R, gouy, trapz
from .kernel  import get_lib
from .sellmeier import (n_pump, n_signal, n_idler,
                        CRYSTAL_NAMES, TYPE_NAMES)


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def _sigma_lp_from_fwhm_nm(fwhm_nm: float) -> tuple[float, float]:
    """Pump FWHM [nm] → (σ_λp [µm], FWHM [µm])."""
    fwhm_um  = fwhm_nm * 1e-3
    sigma_lp = fwhm_um / (2.0 * np.sqrt(np.log(2)))
    return sigma_lp, fwhm_um


def _xi(L, lam, n, w):
    return L * lam / (2.0 * np.pi * n * w * w)


def _adaptive_pmax(L, lams, ns, ws, p_min=4):
    xi_max = max(_xi(L, lam, n, w) for lam, n, w in zip(lams, ns, ws))
    return max(p_min, int(np.ceil(1.5 * xi_max)) + 2)


# ══════════════════════════════════════════════════════════════════════
#  1.  2-D waist scan
# ══════════════════════════════════════════════════════════════════════
def waist_scan(
    *,
    wp_range=(10.0, 100.0),
    ws_range=(10.0, 100.0),
    n_w=30,
    n_z=48,
    n_r=64,
    n_lam=30,
    p_max_min=6,
    method=1,                                 # 0 = LG sum,  1 = double-z analytic
    fwhm_nm=None,                             # default from config
    half_range=None,                          # default from config
    verbose=True,
):
    """Map S₂, brightness, and symmetric heralding η over (w_p, w_s).

    Returns
    -------
    dict with keys:
        wp, ws, S2, B, H, method, time_s, params
    """
    lib = get_lib()

    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, fwhm_um = _sigma_lp_from_fwhm_nm(fwhm_nm)
    sigma_pump = 4.0 * fwhm_um / np.sqrt(2.0 * np.log(2))

    if half_range is None:
        half_range = cfg.spectral_half_range

    # Spectral grid
    k_s, k_i, k_p, dk, w_spec, alpha_sq, n_u, n_v = grids.build_uv_grid(
        n_lam, half_range, sigma_pump,
        cfg.T, cfg.CRYSTAL, cfg.SPDC_TYPE,
        cfg.lambda0_p, cfg.lambda0_s, cfg.lambda0_i,
        cfg.mismatch)
    n_spec = len(k_s)

    # Quadrature
    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)
    r_ref, r_weights = qd.gauss_legendre_r(n_r)

    # Waist arrays
    wp_arr = np.ascontiguousarray(np.linspace(*wp_range, n_w), dtype=np.float64)
    ws_arr = np.ascontiguousarray(np.linspace(*ws_range, n_w), dtype=np.float64)

    S2_map = np.zeros((n_w, n_w), dtype=np.float64)
    H_map  = np.zeros((n_w, n_w), dtype=np.float64)

    if verbose:
        method_name = "LG mode sum" if method == 0 else "double-z analytic"
        print(f"\n{'═'*60}")
        print(f"  2-D waist scan — {CRYSTAL_NAMES[cfg.CRYSTAL]}, "
              f"{TYPE_NAMES[cfg.SPDC_TYPE]}")
        print(f"{'═'*60}")
        print(f"  L = {cfg.L/1e3:.1f} mm   T = {cfg.T} °C   "
              f"pump FWHM = {fwhm_nm:.3f} nm")
        print(f"  Grid {n_w}×{n_w},  N_Z = {n_z},  N_R = {n_r}")
        print(f"  Spectral pts: {n_spec}  (rotated {n_u}×{n_v} mesh)")
        print(f"  Method      : {method} — {method_name}")

    t0 = time.time()
    lib.scan_waists(
        n_w, n_w, wp_arr, ws_arr,
        n_spec, k_s, k_i, dk, w_spec, alpha_sq, k_p,
        cfg.k0_p, cfg.L, cfg.offset_w0,
        cfg.lambda0_p, cfg.lambda0_s, cfg.lambda0_i,
        cfg.n0_p, cfg.n0_s, cfg.n0_i,
        cfg.k0_s, cfg.k0_i,
        p_max_min,
        n_z, z_nodes, z_weights, chi_z,
        n_r, r_ref, r_weights,
        int(method),
        S2_map, H_map,
    )
    elapsed = time.time() - t0
    B_map = S2_map / S2_map.max()

    if verbose:
        print(f"  done in {elapsed:.2f}s  "
              f"({elapsed / n_w**2 * 1e3:.1f} ms/pair)")
        print(f"{'═'*60}")

    return dict(
        wp=wp_arr, ws=ws_arr,
        S2=S2_map, B=B_map, H=H_map,
        method=method, time_s=elapsed,
        params=dict(n_w=n_w, n_z=n_z, n_r=n_r, n_lam=n_lam,
                    fwhm_nm=fwhm_nm, half_range=half_range,
                    p_max_min=p_max_min),
    )


# ══════════════════════════════════════════════════════════════════════
#  2.  Single waist pair (S2, S1_s, S1_i)
# ══════════════════════════════════════════════════════════════════════
def pair_singles(
    wp, ws, *,
    wi=None,
    method=1,
    p_max=10,
    n_z=48,
    n_r=64,
    n_lam=30,
    fwhm_nm=None,
    half_range=None,
):
    """Compute S₂, S₁_s, S₁_i for a single (w_p, w_s, w_i) triplet."""
    lib = get_lib()
    if wi is None:
        wi = ws
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, fwhm_um = _sigma_lp_from_fwhm_nm(fwhm_nm)
    sigma_pump = 4.0 * fwhm_um / np.sqrt(2.0 * np.log(2))
    if half_range is None:
        half_range = cfg.spectral_half_range

    k_s, k_i, k_p, dk, w_spec, alpha_sq, *_ = grids.build_uv_grid(
        n_lam, half_range, sigma_pump,
        cfg.T, cfg.CRYSTAL, cfg.SPDC_TYPE,
        cfg.lambda0_p, cfg.lambda0_s, cfg.lambda0_i,
        cfg.mismatch)
    n_spec = len(k_s)

    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)
    r_ref, r_weights = qd.gauss_legendre_r(n_r)

    S2  = ctypes.c_double()
    S1s = ctypes.c_double()
    S1i = ctypes.c_double()
    lib.compute_single(
        n_spec, k_s, k_i, dk, w_spec, alpha_sq, k_p,
        int(p_max),
        wp, ws, wi,
        cfg.k0_p, cfg.L, cfg.offset_w0,
        cfg.k0_s, cfg.k0_i,
        n_z, z_nodes, z_weights, chi_z,
        n_r, r_ref, r_weights,
        int(method),
        ctypes.byref(S2), ctypes.byref(S1s), ctypes.byref(S1i),
    )
    return S2.value, S1s.value, S1i.value


# ══════════════════════════════════════════════════════════════════════
#  3.  Mode-resolved spectra
# ══════════════════════════════════════════════════════════════════════
def modes_spectrum(
    *,
    wp=None, ws=None, wi=None,
    p_max=10,
    n_z=256,
    n_r=64,
    n_lam=0,
    n_i=0,
    fwhm_nm=None,
    half_range=None,
    verbose=True,
):
    """Mode-resolved P_{0,p_i}(λ_s) and P_{p_s,0}(λ_s) on a λ_s grid.

    The λ_i integration and the pump α² envelope are folded into per-
    block weights inside the grid builder, so the C kernel returns
    integrated spectra directly.
    """
    lib = get_lib()
    if wp is None: wp = cfg.w0_p
    if ws is None: ws = cfg.w0_s
    if wi is None: wi = ws
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, _ = _sigma_lp_from_fwhm_nm(fwhm_nm)
    if half_range is None:
        half_range = cfg.spectral_half_range
    if n_lam <= 0:
        n_lam = cfg.n_lambda_recommended

    if p_max <= 0:
        p_max = _adaptive_pmax(
            cfg.L,
            (cfg.lambda0_p, cfg.lambda0_s, cfg.lambda0_i),
            (cfg.n0_p, cfg.n0_s, cfg.n0_i),
            (wp, ws, wi))

    dlam = np.linspace(-half_range, half_range, n_lam)
    lam_s = cfg.lambda0_s - dlam

    k_s, k_i, k_p, dk, weights, n_i = grids.build_modes_grid(
        lam_s, sigma_lp, n_i,
        cfg.T, cfg.CRYSTAL, cfg.SPDC_TYPE, cfg.lambda0_p, cfg.KG)

    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)
    r_ref, r_weights = qd.gauss_legendre_r(n_r)

    n_modes = p_max + 1
    P_0i = np.zeros(n_modes * n_lam, dtype=np.float64)
    P_i0 = np.zeros(n_modes * n_lam, dtype=np.float64)

    if verbose:
        xi_p = _xi(cfg.L, cfg.lambda0_p, cfg.n0_p, wp)
        xi_s = _xi(cfg.L, cfg.lambda0_s, cfg.n0_s, ws)
        xi_i = _xi(cfg.L, cfg.lambda0_i, cfg.n0_i, wi)
        print(f"\n{'═'*60}")
        print(f"  Mode-resolved spectrum — {CRYSTAL_NAMES[cfg.CRYSTAL]}, "
              f"{TYPE_NAMES[cfg.SPDC_TYPE]}")
        print(f"{'═'*60}")
        print(f"  L = {cfg.L/1e3:.0f} mm   T = {cfg.T} °C")
        print(f"  w_p = {wp:.1f} µm,  w_s = {ws:.1f} µm  "
              f"(ξ_p={xi_p:.3f}, ξ_s={xi_s:.3f})")
        print(f"  P_MAX = {p_max},  N_Z = {n_z},  N_R = {n_r}")
        print(f"  N_λ = {n_lam},  N_i = {n_i}  "
              f"(total {n_lam*n_i} spectral pts)")

    t0 = time.time()
    lib.compute_modes_spectrum(
        n_lam, n_i, p_max,
        k_s, k_i, k_p, dk, weights,
        wp, ws, wi,
        cfg.k0_p, cfg.L, cfg.offset_w0,
        cfg.k0_s, cfg.k0_i,
        n_z, z_nodes, z_weights, chi_z,
        n_r, r_ref, r_weights,
        P_0i, P_i0,
    )
    elapsed = time.time() - t0

    P_0i = P_0i.reshape(n_modes, n_lam)
    P_i0 = P_i0.reshape(n_modes, n_lam)

    dlam_step = float(dlam[1] - dlam[0])
    S2   = trapz(P_0i[0],          dx=dlam_step)
    S1_s = trapz(P_0i.sum(axis=0), dx=dlam_step)
    S1_i = trapz(P_i0.sum(axis=0), dx=dlam_step)
    eta_s = S2 / S1_i if S1_i > 0 else 0.0
    eta_i = S2 / S1_s if S1_s > 0 else 0.0
    eta   = float(np.sqrt(eta_s * eta_i))

    if verbose:
        print(f"  done in {elapsed:.2f}s")
        print(f"  η_s = {eta_s:.4f},  η_i = {eta_i:.4f},  η = {eta:.4f}")
        print(f"{'═'*60}")

    return dict(
        delta_lam=dlam, lam_s=lam_s,
        P_0i=P_0i, P_i0=P_i0,
        p_max=p_max,
        eta=eta, eta_s=eta_s, eta_i=eta_i,
        S2=S2, S1_s=S1_s, S1_i=S1_i,
        time_s=elapsed,
        params=dict(wp=wp, ws=ws, wi=wi,
                    n_z=n_z, n_r=n_r, n_lam=n_lam, n_i=n_i,
                    fwhm_nm=fwhm_nm, half_range=half_range),
    )


# ══════════════════════════════════════════════════════════════════════
#  4.  Joint Spectral Amplitude (complex)
# ══════════════════════════════════════════════════════════════════════
def jsa(
    *,
    wp=None, ws=None, wi=None,
    n_lam=150, n_z=64,
    half_range=0.002,
    T=None,
    fwhm_nm=None,
):
    """Complex JSA C₀₀(λ_s, λ_i) with α(λ_p) envelope; Schmidt decomp.

    Pure NumPy: the z-integral is vectorised over the (n_lam × n_lam)
    spectral grid.  Returns a result dict including the Schmidt number K
    and the normalised JSA.
    """
    if wp is None: wp = cfg.w0_p
    if ws is None: ws = cfg.w0_s
    if wi is None: wi = ws
    if T is None:  T = cfg.T
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, _ = _sigma_lp_from_fwhm_nm(fwhm_nm)

    lam_1d = np.linspace(cfg.lambda0_s - half_range,
                         cfg.lambda0_s + half_range, n_lam)
    lam_s, lam_i = np.meshgrid(lam_1d, lam_1d, indexing="ij")

    # Vectorised Sellmeier (no np.vectorize)
    ns = n_signal(lam_s, T, cfg.CRYSTAL, cfg.SPDC_TYPE)
    ni = n_idler (lam_i, T, cfg.CRYSTAL, cfg.SPDC_TYPE)
    lam_p = lam_s * lam_i / (lam_s + lam_i)
    npx   = n_pump(lam_p, T, cfg.CRYSTAL, cfg.SPDC_TYPE)

    k_s = 2 * np.pi * ns  / lam_s
    k_i = 2 * np.pi * ni  / lam_i
    k_p = 2 * np.pi * npx / lam_p
    dk  = k_p - k_s - k_i + cfg.mismatch

    norm_alpha = 1.0 / (np.pi**0.25 * np.sqrt(sigma_lp))
    alpha      = norm_alpha * np.exp(-0.5 * ((lam_p - cfg.lambda0_p) / sigma_lp)**2)

    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)

    half_L = 0.5 * cfg.L
    off_p = cfg.offset_w0 + half_L * (k_p / cfg.k0_p - 1.0)
    off_s = cfg.offset_w0 + half_L * (k_s / cfg.k0_s - 1.0)
    off_i = cfg.offset_w0 + half_L * (k_i / cfg.k0_i - 1.0)

    PREFAC = (2.0 / np.pi)**1.5
    C00 = np.zeros((n_lam, n_lam), dtype=np.complex128)

    t0 = time.time()
    for jz in range(n_z):
        zj   = z_nodes[jz]
        wjc  = z_weights[jz] * chi_z[jz]

        z_p, z_s, z_i = zj + off_p, zj + off_s, zj + off_i
        w_p = beam_w(z_p, k_p, wp)
        w_s = beam_w(z_s, k_s, ws)
        w_i = beam_w(z_i, k_i, wi)

        Re_A = 1/w_p**2 + 1/w_s**2 + 1/w_i**2
        Im_A = (-0.5 * k_p * inv_R(z_p, k_p, wp)
                + 0.5 * k_s * inv_R(z_s, k_s, ws)
                + 0.5 * k_i * inv_R(z_i, k_i, wi))
        A    = Re_A + 1j * Im_A

        gp = gouy(z_p, k_p, wp)
        gs = gouy(z_s, k_s, ws)
        gi = gouy(z_i, k_i, wi)

        I0 = (PREFAC / (w_p * w_s * w_i)) * np.exp(1j * (-gp + gs + gi)) / (2.0 * A)
        C00 += wjc * np.exp(1j * dk * zj) * I0

    C00 *= 2.0 * np.pi * alpha
    elapsed = time.time() - t0

    # Schmidt decomposition
    dlam_s = lam_1d[1] - lam_1d[0]
    norm_C = np.sqrt(np.sum(np.abs(C00)**2) * dlam_s * dlam_s)
    C00_n  = C00 / (norm_C + 1e-300)

    sv = np.linalg.svd(C00_n, compute_uv=False)
    sv_n = sv * np.sqrt(dlam_s * dlam_s)
    lam_k = sv_n**2
    lam_k /= lam_k.sum()
    K = float(1.0 / np.sum(lam_k**2))

    return dict(
        lam_s=lam_1d, lam_i=lam_1d,
        C00=C00, C00_norm=C00_n,
        schmidt_values=lam_k,
        K=K,
        time_s=elapsed,
        params=dict(wp=wp, ws=ws, wi=wi, T=T,
                    n_lam=n_lam, n_z=n_z,
                    half_range=half_range, fwhm_nm=fwhm_nm),
    )


# ══════════════════════════════════════════════════════════════════════
#  5.  C₀₀ heatmap (T, λ_s) → integrated signal spectra
# ══════════════════════════════════════════════════════════════════════
def c00_heatmap(
    *,
    T_range,
    dt,
    lam_range,
    n_lam,
    wp=None, ws=None, wi=None,
    n_z=48,
    n_i=0,
    fwhm_nm=None,
    verbose=True,
):
    """|C₀₀(λ_s, T)|² integrated over λ_i with the pump envelope.

    Returns
    -------
    dict with keys:
        T, lam_nm, spectra (N_T × N_lam), |c00sq_3d| (N_T×N_lam×N_i)
    """
    lib = get_lib()
    if wp is None: wp = cfg.w0_p
    if ws is None: ws = cfg.w0_s
    if wi is None: wi = ws
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, _ = _sigma_lp_from_fwhm_nm(fwhm_nm)

    T_arr   = np.arange(T_range[0], T_range[1] + 0.5 * dt, dt)
    lam_arr = np.linspace(lam_range[0], lam_range[1], n_lam)
    N_T     = len(T_arr)

    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)

    k_s, k_i, k_p, dk, alpha_sq, n_i_actual = grids.build_heatmap_grid(
        T_arr, lam_arr, sigma_lp, n_i,
        cfg.CRYSTAL, cfg.SPDC_TYPE, cfg.lambda0_p, cfg.KG)
    N_tot = len(k_s)

    for a in (k_s, k_i, k_p, dk, alpha_sq):
        a.flags.writeable = True
    c00sq = np.empty(N_tot, dtype=np.float64)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  C₀₀ heatmap — {CRYSTAL_NAMES[cfg.CRYSTAL]}, "
              f"{TYPE_NAMES[cfg.SPDC_TYPE]}")
        print(f"{'═'*60}")
        print(f"  L = {cfg.L/1e3:.1f} mm   QPM at {cfg.T_QPM} °C")
        print(f"  w_p = {wp:.1f}, w_s = {ws:.1f}   pump FWHM = {fwhm_nm:.3f} nm")
        print(f"  T: {T_arr[0]:.2f} – {T_arr[-1]:.2f} °C  ({N_T} pts, ΔT = {dt})")
        print(f"  λ_s: {lam_arr[0]*1e3:.2f} – {lam_arr[-1]*1e3:.2f} nm  "
              f"({n_lam} pts)")
        print(f"  Total grid pts: {N_tot} (N_i = {n_i_actual})")

    t0 = time.time()
    lib.compute_c00_heatmap(
        N_tot, k_s, k_i, k_p, dk,
        wp, ws, wi,
        cfg.L, cfg.offset_w0,
        cfg.k0_p, cfg.k0_s, cfg.k0_i,
        n_z, z_nodes, z_weights, chi_z,
        c00sq,
    )
    elapsed = time.time() - t0

    # ∫ dλ_i |α|² |C₀₀|²  with trapezoidal weights
    c00sq_3d = c00sq.reshape(N_T, n_lam, n_i_actual)
    alpha_3d = alpha_sq.reshape(N_T, n_lam, n_i_actual)
    w_trap   = np.ones(n_i_actual); w_trap[0] = 0.5; w_trap[-1] = 0.5
    spectra  = np.einsum("ijk,k->ij", alpha_3d * c00sq_3d, w_trap)

    if verbose:
        print(f"  done in {elapsed:.2f}s")
        print(f"{'═'*60}")

    return dict(
        T=T_arr, lam_nm=lam_arr * 1e3,
        spectra=spectra,
        c00sq_3d=c00sq_3d, alpha_3d=alpha_3d,
        n_i=n_i_actual, time_s=elapsed,
        params=dict(wp=wp, ws=ws, wi=wi,
                    n_z=n_z, n_lam=n_lam,
                    fwhm_nm=fwhm_nm, dt=dt),
    )


# ══════════════════════════════════════════════════════════════════════
#  6.  HOM visibility vs T
# ══════════════════════════════════════════════════════════════════════
_C_UM_PER_PS = 299.792458


def _hom_visibility(C00, lam_1d, tau_ps):
    """V(τ) for a complex JSA on a symmetric grid."""
    dlam  = lam_1d[1] - lam_1d[0]
    G     = C00 * np.conj(C00.T)
    omega = 2.0 * np.pi * _C_UM_PER_PS / lam_1d
    h_s   = np.exp( 1j * np.outer(omega, tau_ps))
    h_i   = np.exp(-1j * np.outer(omega, tau_ps))
    V     = np.einsum("ij,ij->j", h_s, G @ h_i) * dlam * dlam
    denom = np.sum(np.abs(C00)**2) * dlam * dlam + 1e-300
    return np.real(V) / denom


def hom_vs_T(
    *,
    T_range,
    dt,
    wp=None, ws=None,
    n_lam=120, n_z=64,
    half_range=0.002,
    tau_max_ps=10.0,
    n_tau=401,
    fwhm_nm=None,
    verbose=True,
):
    """Sweep T, compute the complex JSA at each, derive V(τ) and K(T)."""
    if wp is None: wp = cfg.w0_p
    if ws is None: ws = cfg.w0_s
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3

    if n_tau % 2 == 0:                  # keep τ = 0 on the grid
        n_tau += 1

    T_arr  = np.arange(T_range[0], T_range[1] + 0.5 * dt, dt)
    tau_ps = np.linspace(-tau_max_ps, tau_max_ps, n_tau)
    i_tau0 = int(np.argmin(np.abs(tau_ps)))
    lam_1d = np.linspace(cfg.lambda0_s - half_range,
                         cfg.lambda0_s + half_range, n_lam)
    N_T = len(T_arr)

    V_tau = np.empty((N_T, n_tau))
    K_eff = np.empty(N_T)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  HOM vs T — {CRYSTAL_NAMES[cfg.CRYSTAL]}, "
              f"{TYPE_NAMES[cfg.SPDC_TYPE]}")
        print(f"{'═'*60}")
        print(f"  T: {T_arr[0]:.2f} – {T_arr[-1]:.2f} °C  ({N_T} pts)")
        print(f"  λ-grid ±{half_range*1e3:.2f} nm  ({n_lam}×{n_lam} pts)")
        print(f"  τ-scan ±{tau_max_ps:.2f} ps  ({n_tau} pts)")
        print(f"  w_p = {wp:.1f}, w_s = {ws:.1f},  pump FWHM = {fwhm_nm:.3f} nm")

    t0 = time.time()
    for iT, T_val in enumerate(T_arr):
        result = jsa(wp=wp, ws=ws, wi=ws,
                     T=T_val, n_lam=n_lam, n_z=n_z,
                     half_range=half_range, fwhm_nm=fwhm_nm)
        C00 = result["C00"]
        V_tau[iT] = _hom_visibility(C00, lam_1d, tau_ps)

        absC = np.abs(C00)
        norm = float(np.sum(absC**2))
        if norm > 0:
            sv = np.linalg.svd(C00 / np.sqrt(norm), compute_uv=False)
            lam_k = sv**2
            lam_k /= lam_k.sum()
            K_eff[iT] = 1.0 / np.sum(lam_k**2)
        else:
            K_eff[iT] = np.nan
        if verbose:
            print(f"    T = {T_val:7.2f}°C   "
                  f"V(0) = {V_tau[iT, i_tau0]:+.4f}   "
                  f"K = {K_eff[iT]:5.2f}")
    elapsed = time.time() - t0

    if verbose:
        print(f"  total: {elapsed:.2f}s")
        print(f"{'═'*60}")

    return dict(
        T=T_arr, tau_ps=tau_ps,
        V_tau=V_tau, V0=V_tau[:, i_tau0],
        dip=1.0 - V_tau,
        K_eff=K_eff, i_tau0=i_tau0,
        lam_nm=lam_1d * 1e3,
        time_s=elapsed,
        params=dict(wp=wp, ws=ws,
                    n_lam=n_lam, n_z=n_z,
                    half_range=half_range,
                    tau_max_ps=tau_max_ps, n_tau=n_tau,
                    fwhm_nm=fwhm_nm, dt=dt),
    )


# ══════════════════════════════════════════════════════════════════════
#  7.  Brightness vs T with a tunable filter
# ══════════════════════════════════════════════════════════════════════
def make_filter(lam_um, center_um, bw_um, shape):
    """Filter transmission profile on the signal grid.  Peak = 1."""
    if shape == "none" or bw_um is None or bw_um <= 0:
        return np.ones_like(lam_um)
    dl = lam_um - center_um
    if shape == "rect":
        return (np.abs(dl) <= 0.5 * bw_um).astype(np.float64)
    if shape == "gauss":
        sigma = bw_um / (2.0 * np.sqrt(np.log(2.0)))
        return np.exp(-(dl / sigma)**2)
    raise ValueError(f"unknown filter shape: {shape!r} (rect|gauss|none)")


def brightness_vs_T(
    *,
    T_range, dt,
    filter_center_nm, filter_bw_nm, filter_shape="rect",
    wp=None, ws=None,
    lam_range_nm=None, n_lam=400,
    n_z=64, n_i=0,
    fwhm_nm=None,
    verbose=True,
):
    """Compute B(T) = ∫dλ_s f(λ_s) ∫dλ_i |α|² |C₀₀|² for several filter BWs."""
    if wp is None: wp = cfg.w0_p
    if ws is None: ws = cfg.w0_s
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3

    if not hasattr(filter_bw_nm, "__len__"):
        filter_bw_nm = [filter_bw_nm]
    bw_um  = np.asarray(filter_bw_nm, dtype=np.float64) * 1e-3
    ctr_um = filter_center_nm * 1e-3
    max_bw = float(bw_um.max()) if bw_um.size else 0.0
    half_um = max(1.5 * max_bw, 10.0e-3)

    if lam_range_nm is None:
        lam_lo = max(ctr_um - half_um, 0.5 * cfg.lambda0_p)
        lam_hi = ctr_um + half_um
    else:
        lam_lo, lam_hi = lam_range_nm[0] * 1e-3, lam_range_nm[1] * 1e-3

    hm = c00_heatmap(
        T_range=T_range, dt=dt,
        lam_range=(lam_lo, lam_hi), n_lam=n_lam,
        wp=wp, ws=ws,
        n_z=n_z, n_i=n_i,
        fwhm_nm=fwhm_nm, verbose=verbose,
    )

    lam_um  = hm["lam_nm"] * 1e-3
    spectra = hm["spectra"]
    dlam_um = lam_um[1] - lam_um[0]

    w_trap_s     = np.full(n_lam, dlam_um)
    w_trap_s[0] *= 0.5; w_trap_s[-1] *= 0.5

    if filter_shape == "none":
        bw_um = np.array([0.0])

    N_F = len(bw_um)
    filters    = np.empty((N_F, n_lam))
    brightness = np.empty((N_F, len(hm["T"])))
    for iF, bw in enumerate(bw_um):
        f_lam = (np.ones_like(lam_um) if filter_shape == "none"
                 else make_filter(lam_um, ctr_um, bw, filter_shape))
        filters[iF]    = f_lam
        brightness[iF] = (spectra * (f_lam * w_trap_s)[None, :]).sum(axis=1)

    return dict(
        T=hm["T"], lam_nm=hm["lam_nm"],
        spectra=spectra,
        brightness=brightness,
        filters=filters,
        filter_bw_nm=np.asarray(filter_bw_nm, dtype=np.float64),
        filter_center_nm=filter_center_nm,
        filter_shape=filter_shape,
        params=dict(wp=wp, ws=ws,
                    n_lam=n_lam, n_z=n_z,
                    fwhm_nm=fwhm_nm, dt=dt),
    )
