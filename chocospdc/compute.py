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
    filter_center_nm=None,
    filter_bw_nm=None,
    filter_shape="none",
    filter_file=None,
    verbose=True,
):
    """Map S₂, brightness, and symmetric heralding η over (w_p, w_s).

    A symmetric bandpass filter (same in both arms) can be applied with
    the standard ``filter_*`` parameters.  Use ``filter_shape='file'``
    with ``filter_file=<path>`` for a custom filter from a two-column
    text file ``(λ_nm, T_percent)``.  Default ``filter_shape='none'``
    leaves the integration window (``half_range``) as the implicit rect,
    so existing scripts produce identical results.

    Returns
    -------
    dict with keys:
        wp, ws, S2, B, H, method, time_s, has_filter, params
    """
    lib = get_lib()

    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, fwhm_um = _sigma_lp_from_fwhm_nm(fwhm_nm)
    sigma_pump = 4.0 * fwhm_um / np.sqrt(2.0 * np.log(2))

    if half_range is None:
        half_range = cfg.spectral_half_range

    # Spectral grid — now also returns lam_s, lam_i for filter evaluation
    (k_s, k_i, k_p, dk, w_spec, alpha_sq,
     lam_s, lam_i, n_u, n_v) = grids.build_uv_grid(
        n_lam, half_range, sigma_pump,
        cfg.T, cfg.CRYSTAL, cfg.SPDC_TYPE,
        cfg.lambda0_p, cfg.lambda0_s, cfg.lambda0_i,
        cfg.mismatch)
    n_spec = len(k_s)

    # Symmetric filter arrays
    if filter_center_nm is None:
        filter_center_nm = cfg.lambda0_s * 1e3
    ctr_um = filter_center_nm * 1e-3
    bw_um  = (filter_bw_nm or 0.0) * 1e-3
    has_filter = _filter_spec_active(filter_shape, filter_bw_nm, filter_file)
    if has_filter:
        f_s_arr = _build_filter(lam_s, center_um=ctr_um, bw_um=bw_um,
                                shape=filter_shape, file=filter_file)
        f_i_arr = _build_filter(lam_i, center_um=ctr_um, bw_um=bw_um,
                                shape=filter_shape, file=filter_file)
    else:
        f_s_arr = np.ones_like(lam_s)
        f_i_arr = np.ones_like(lam_i)
    f_s_arr = np.ascontiguousarray(f_s_arr, dtype=np.float64)
    f_i_arr = np.ascontiguousarray(f_i_arr, dtype=np.float64)

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
        if has_filter:
            if filter_shape == "file":
                print(f"  Filter on both arms: file = {filter_file}")
            else:
                print(f"  Filter on both arms: {filter_shape}, "
                      f"BW = {filter_bw_nm:.3f} nm, "
                      f"centre = {filter_center_nm:.3f} nm")
        else:
            print(f"  Filter on both arms: none "
                  f"(integration window acts as implicit rect)")

    t0 = time.time()
    lib.scan_waists(
        n_w, n_w, wp_arr, ws_arr,
        n_spec, k_s, k_i, dk, w_spec, alpha_sq,
        f_s_arr, f_i_arr,
        k_p,
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
    B_map = S2_map / max(S2_map.max(), 1e-300)

    if verbose:
        print(f"  done in {elapsed:.2f}s  "
              f"({elapsed / n_w**2 * 1e3:.1f} ms/pair)")
        print(f"{'═'*60}")

    return dict(
        wp=wp_arr, ws=ws_arr,
        S2=S2_map, B=B_map, H=H_map,
        method=method, time_s=elapsed,
        has_filter=has_filter,
        params=dict(n_w=n_w, n_z=n_z, n_r=n_r, n_lam=n_lam,
                    fwhm_nm=fwhm_nm, half_range=half_range,
                    p_max_min=p_max_min,
                    filter_center_nm=filter_center_nm,
                    filter_bw_nm=filter_bw_nm,
                    filter_shape=filter_shape,
                    filter_file=filter_file),
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
    filter_center_nm=None,
    filter_bw_nm=None,
    filter_shape="none",
    filter_file=None,
):
    """Compute S₂, S₁_s, S₁_i for a single (w_p, w_s, w_i) triplet.

    Symmetric-filter parameters match :func:`waist_scan` and the rest of
    the API.  Default ``filter_shape='none'`` reproduces the legacy
    "integration window as implicit rect" behaviour.
    """
    lib = get_lib()
    if wi is None:
        wi = ws
    if fwhm_nm is None:
        fwhm_nm = cfg.FWHM_PUMP * 1e3
    sigma_lp, fwhm_um = _sigma_lp_from_fwhm_nm(fwhm_nm)
    sigma_pump = 4.0 * fwhm_um / np.sqrt(2.0 * np.log(2))
    if half_range is None:
        half_range = cfg.spectral_half_range

    (k_s, k_i, k_p, dk, w_spec, alpha_sq,
     lam_s, lam_i, _n_u, _n_v) = grids.build_uv_grid(
        n_lam, half_range, sigma_pump,
        cfg.T, cfg.CRYSTAL, cfg.SPDC_TYPE,
        cfg.lambda0_p, cfg.lambda0_s, cfg.lambda0_i,
        cfg.mismatch)
    n_spec = len(k_s)

    # Symmetric filter arrays
    if filter_center_nm is None:
        filter_center_nm = cfg.lambda0_s * 1e3
    ctr_um = filter_center_nm * 1e-3
    bw_um  = (filter_bw_nm or 0.0) * 1e-3
    has_filter = _filter_spec_active(filter_shape, filter_bw_nm, filter_file)
    if has_filter:
        f_s_arr = _build_filter(lam_s, center_um=ctr_um, bw_um=bw_um,
                                shape=filter_shape, file=filter_file)
        f_i_arr = _build_filter(lam_i, center_um=ctr_um, bw_um=bw_um,
                                shape=filter_shape, file=filter_file)
    else:
        f_s_arr = np.ones_like(lam_s)
        f_i_arr = np.ones_like(lam_i)
    f_s_arr = np.ascontiguousarray(f_s_arr, dtype=np.float64)
    f_i_arr = np.ascontiguousarray(f_i_arr, dtype=np.float64)

    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)
    r_ref, r_weights = qd.gauss_legendre_r(n_r)

    S2  = ctypes.c_double()
    S1s = ctypes.c_double()
    S1i = ctypes.c_double()
    lib.compute_single(
        n_spec, k_s, k_i, dk, w_spec, alpha_sq,
        f_s_arr, f_i_arr,
        k_p,
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
    filter_center_nm=None,
    filter_bw_nm=None,
    filter_shape="none",
    filter_file=None,
    verbose=True,
):
    """Mode-resolved P_{0,p_i}(λ_s) and P_{p_s,0}(λ_s) on a λ_s grid.

    The C kernel returns two flavours of each output:

      * ``P_*_bare``  : Σ_{λ_i}  α² · |C|²            (no idler filter)
      * ``P_*_fi``    : Σ_{λ_i}  α² · f_i(λ_i) · |C|²  (idler filter applied)

    From these, the three observables are assembled with the correct
    detection topology:

        S₂   = ∫ dλ_s f_s(λ_s) · P_0i_fi[0]            (both detectors fire)
        S₁_s = ∫ dλ_s f_s(λ_s) · Σ_p P_0i_bare[p]      (signal-arm only)
        S₁_i =          ∫ dλ_s Σ_p P_i0_fi[p]          (idler-arm only)

    With ``filter_shape='none'`` the bare and filtered arrays coincide
    and the result reproduces the legacy behaviour exactly.
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

    k_s, k_i, k_p, dk, weights, n_i, lam_i_2d = grids.build_modes_grid(
        lam_s, sigma_lp, n_i,
        cfg.T, cfg.CRYSTAL, cfg.SPDC_TYPE, cfg.lambda0_p, cfg.KG)

    # Symmetric filter
    if filter_center_nm is None:
        filter_center_nm = cfg.lambda0_s * 1e3
    ctr_um = filter_center_nm * 1e-3
    bw_um  = (filter_bw_nm or 0.0) * 1e-3
    has_filter = _filter_spec_active(filter_shape, filter_bw_nm, filter_file)
    if has_filter:
        f_s_1d  = _build_filter(lam_s,     center_um=ctr_um, bw_um=bw_um,
                                shape=filter_shape, file=filter_file)
        f_i_2d  = _build_filter(lam_i_2d,  center_um=ctr_um, bw_um=bw_um,
                                shape=filter_shape, file=filter_file)
    else:
        f_s_1d = np.ones_like(lam_s)
        f_i_2d = np.ones_like(lam_i_2d)
    f_i_flat = np.ascontiguousarray(f_i_2d.ravel(), dtype=np.float64)

    z_nodes, z_weights, chi_z = qd.gauss_legendre_z(cfg.L, n_z)
    r_ref, r_weights = qd.gauss_legendre_r(n_r)

    n_modes = p_max + 1
    P_0i_b = np.zeros(n_modes * n_lam, dtype=np.float64)
    P_0i_f = np.zeros(n_modes * n_lam, dtype=np.float64)
    P_i0_b = np.zeros(n_modes * n_lam, dtype=np.float64)
    P_i0_f = np.zeros(n_modes * n_lam, dtype=np.float64)

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
        if has_filter:
            tag = (f"file = {filter_file}" if filter_shape == "file"
                   else f"{filter_shape}, BW = {filter_bw_nm:.3f} nm, "
                        f"centre = {filter_center_nm:.3f} nm")
            print(f"  Filter on both arms: {tag}")
        else:
            print(f"  Filter on both arms: none "
                  f"(integration window acts as implicit rect)")

    t0 = time.time()
    lib.compute_modes_spectrum(
        n_lam, n_i, p_max,
        k_s, k_i, k_p, dk, weights,
        f_i_flat,
        wp, ws, wi,
        cfg.k0_p, cfg.L, cfg.offset_w0,
        cfg.k0_s, cfg.k0_i,
        n_z, z_nodes, z_weights, chi_z,
        n_r, r_ref, r_weights,
        P_0i_b, P_0i_f, P_i0_b, P_i0_f,
    )
    elapsed = time.time() - t0

    P_0i_b = P_0i_b.reshape(n_modes, n_lam)
    P_0i_f = P_0i_f.reshape(n_modes, n_lam)
    P_i0_b = P_i0_b.reshape(n_modes, n_lam)
    P_i0_f = P_i0_f.reshape(n_modes, n_lam)

    # Heralding with correct filter weights on each accumulator
    dlam_step = float(dlam[1] - dlam[0])
    S2   = trapz(f_s_1d * P_0i_f[0],          dx=dlam_step)
    S1_s = trapz(f_s_1d * P_0i_b.sum(axis=0), dx=dlam_step)
    S1_i = trapz(         P_i0_f.sum(axis=0), dx=dlam_step)
    eta_s = S2 / S1_i if S1_i > 0 else 0.0
    eta_i = S2 / S1_s if S1_s > 0 else 0.0
    eta   = float(np.sqrt(eta_s * eta_i))

    if verbose:
        print(f"  done in {elapsed:.2f}s")
        print(f"  η_s = {eta_s:.4f},  η_i = {eta_i:.4f},  η = {eta:.4f}")
        print(f"{'═'*60}")

    return dict(
        delta_lam=dlam, lam_s=lam_s,
        # bare spectra (no filter applied to either axis) — for plotting
        P_0i=P_0i_b, P_i0=P_i0_b,
        # filtered spectra (idler filter applied inside) — diagnostics
        P_0i_fi=P_0i_f, P_i0_fi=P_i0_f,
        f_s=f_s_1d,
        p_max=p_max,
        eta=eta, eta_s=eta_s, eta_i=eta_i,
        S2=S2, S1_s=S1_s, S1_i=S1_i,
        has_filter=has_filter,
        time_s=elapsed,
        params=dict(wp=wp, ws=ws, wi=wi,
                    n_z=n_z, n_r=n_r, n_lam=n_lam, n_i=n_i,
                    fwhm_nm=fwhm_nm, half_range=half_range,
                    filter_center_nm=filter_center_nm,
                    filter_bw_nm=filter_bw_nm,
                    filter_shape=filter_shape,
                    filter_file=filter_file),
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
    filter_center_nm=None,
    filter_bw_nm=None,
    filter_shape="none",
    filter_file=None,
):
    """Complex JSA C₀₀(λ_s, λ_i) with α(λ_p) envelope; Schmidt decomp.

    A symmetric bandpass filter (same f on both arms) can be applied
    *on top of* the spectral integration window.

      * ``half_range`` is the half-width of the (λ_s, λ_i) grid.  It is
        a numerical knob: make it large enough that the JSA has decayed
        to zero at the edges.
      * ``filter_*`` parameters apply a physical bandpass filter to the
        JSA before computing |C₀₀|², the Schmidt decomposition, and any
        downstream observable.  The same filter is applied to both
        arms.  Default ``filter_shape='none'`` ⇒ no filter (matches the
        legacy behaviour of using ``half_range`` as an implicit rect).

    Pure NumPy: the z-integral is vectorised over the (n_lam × n_lam)
    spectral grid.  Returns a result dict including the filtered Schmidt
    number K and both the bare and filtered JSAs.
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
    C00_bare = np.zeros((n_lam, n_lam), dtype=np.complex128)

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
        C00_bare += wjc * np.exp(1j * dk * zj) * I0

    C00_bare *= 2.0 * np.pi * alpha
    elapsed = time.time() - t0

    # ── Symmetric filter (same on both arms) ─────────────────────
    if filter_center_nm is None:
        filter_center_nm = cfg.lambda0_s * 1e3
    if filter_bw_nm is None:
        filter_bw_nm = 0.0
    ctr_um = filter_center_nm * 1e-3
    bw_um  = filter_bw_nm * 1e-3

    has_filter = _filter_spec_active(filter_shape, filter_bw_nm, filter_file)
    if has_filter:
        f_1d = _build_filter(lam_1d, center_um=ctr_um, bw_um=bw_um,
                             shape=filter_shape, file=filter_file)
        if filter_shape != "file" and bw_um > 1.8 * half_range:
            import warnings
            warnings.warn(
                f"filter_bw_nm ({filter_bw_nm:.3g}) is comparable to or "
                f"larger than 2·half_range ({2*half_range*1e3:.3g} nm); "
                "consider widening half_range so the filter wings fit.",
                RuntimeWarning, stacklevel=2)
        C00 = C00_bare * f_1d[:, None] * f_1d[None, :]
    else:
        f_1d = np.ones_like(lam_1d)
        C00  = C00_bare

    # ── Schmidt decomposition on the (filtered or bare) JSA ──────
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
        C00=C00, C00_bare=C00_bare, C00_norm=C00_n,
        filter_1d=f_1d,
        schmidt_values=lam_k,
        K=K,
        has_filter=has_filter,
        time_s=elapsed,
        params=dict(wp=wp, ws=ws, wi=wi, T=T,
                    n_lam=n_lam, n_z=n_z,
                    half_range=half_range, fwhm_nm=fwhm_nm,
                    filter_center_nm=filter_center_nm,
                    filter_bw_nm=filter_bw_nm,
                    filter_shape=filter_shape,
                    filter_file=filter_file),
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

    k_s, k_i, k_p, dk, alpha_sq, n_i_actual, lam_i_grid = grids.build_heatmap_grid(
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
        lam_i_grid=lam_i_grid,            # (N_lam, n_i) per-(λ_s) idler [µm]
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
    filter_center_nm=None,
    filter_bw_nm=None,
    filter_shape="none",
    filter_file=None,
    verbose=True,
):
    """Sweep T, compute the complex JSA at each, derive V(τ) and K(T).

    Same symmetric-filter API as :func:`jsa` and :func:`brightness_vs_T`:
    set ``filter_shape`` to ``"rect"`` or ``"gauss"`` and provide
    ``filter_bw_nm`` (and optionally ``filter_center_nm``) to simulate
    identical bandpass filters in front of both detectors.  Default
    ``filter_shape='none'`` leaves the JSA unfiltered (the integration
    window ``half_range`` then acts as the de facto rect filter).
    """
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

    has_filter = _filter_spec_active(filter_shape, filter_bw_nm, filter_file)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  HOM vs T — {CRYSTAL_NAMES[cfg.CRYSTAL]}, "
              f"{TYPE_NAMES[cfg.SPDC_TYPE]}")
        print(f"{'═'*60}")
        print(f"  T: {T_arr[0]:.2f} – {T_arr[-1]:.2f} °C  ({N_T} pts)")
        print(f"  λ-grid ±{half_range*1e3:.2f} nm  ({n_lam}×{n_lam} pts)")
        print(f"  τ-scan ±{tau_max_ps:.2f} ps  ({n_tau} pts)")
        print(f"  w_p = {wp:.1f}, w_s = {ws:.1f},  pump FWHM = {fwhm_nm:.3f} nm")
        if has_filter:
            if filter_shape == "file":
                print(f"  Filter on both arms: file = {filter_file}")
            else:
                print(f"  Filter on both arms: {filter_shape}, "
                      f"centre = {filter_center_nm or cfg.lambda0_s*1e3:.3f} nm, "
                      f"BW = {filter_bw_nm:.3f} nm")
        else:
            print(f"  Filter on both arms: none "
                  f"(integration window acts as implicit rect)")

    t0 = time.time()
    for iT, T_val in enumerate(T_arr):
        result = jsa(wp=wp, ws=ws, wi=ws,
                     T=T_val, n_lam=n_lam, n_z=n_z,
                     half_range=half_range, fwhm_nm=fwhm_nm,
                     filter_center_nm=filter_center_nm,
                     filter_bw_nm=filter_bw_nm,
                     filter_shape=filter_shape,
                     filter_file=filter_file)
        C00 = result["C00"]                          # filtered if has_filter else bare
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
        has_filter=has_filter,
        time_s=elapsed,
        params=dict(wp=wp, ws=ws,
                    n_lam=n_lam, n_z=n_z,
                    half_range=half_range,
                    tau_max_ps=tau_max_ps, n_tau=n_tau,
                    fwhm_nm=fwhm_nm, dt=dt,
                    filter_center_nm=filter_center_nm,
                    filter_bw_nm=filter_bw_nm,
                    filter_shape=filter_shape,
                    filter_file=filter_file),
    )


# ══════════════════════════════════════════════════════════════════════
#  7.  Brightness vs T with a tunable filter
# ══════════════════════════════════════════════════════════════════════
def make_filter(lam_um, center_um, bw_um, shape, file=None):
    """Filter transmission profile on a wavelength grid.

    Parameters
    ----------
    lam_um : ndarray
        Wavelength grid [µm] (any shape).
    center_um, bw_um : float
        Centre [µm] and full-width / FWHM [µm] for ``"rect"`` / ``"gauss"``.
        Ignored for ``"none"`` and ``"file"``.
    shape : str
        ``"rect"`` | ``"gauss"`` | ``"file"`` | ``"none"``.
    file : str or path, optional
        Path to a two-column text file with ``(λ_nm, T_percent)`` rows.
        Required when ``shape == "file"``; ignored otherwise.

    Returns
    -------
    ndarray
        Filter transmission in [0, 1], same shape as ``lam_um``.
    """
    if shape == "file":
        if file is None:
            raise ValueError("shape='file' requires a `file` path")
        return _filter_from_file(file, lam_um)
    if shape == "none" or bw_um is None or bw_um <= 0:
        return np.ones_like(lam_um)
    dl = lam_um - center_um
    if shape == "rect":
        return (np.abs(dl) <= 0.5 * bw_um).astype(np.float64)
    if shape == "gauss":
        sigma = bw_um / (2.0 * np.sqrt(np.log(2.0)))
        return np.exp(-(dl / sigma)**2)
    raise ValueError(f"unknown filter shape: {shape!r} (rect|gauss|file|none)")


def _filter_from_file(path, lam_um):
    """Load (λ_nm, T_percent) text file and linearly interpolate to lam_um.

    The file is whitespace- or comma-separated; '#' starts a comment.
    Transmissions are clipped to [0, 1]; out-of-range wavelengths return 0.
    """
    data = np.loadtxt(str(path), comments="#", delimiter=None,
                      ndmin=2, dtype=float)
    if data.shape[1] < 2:
        # try comma delimiter on second pass
        data = np.loadtxt(str(path), comments="#", delimiter=",",
                          ndmin=2, dtype=float)
    if data.shape[1] < 2:
        raise ValueError(f"filter file {path!r}: expected two columns "
                         "(lambda_nm, T_percent)")
    lam_nm_f = data[:, 0]
    T_pct_f  = data[:, 1]
    idx = np.argsort(lam_nm_f)
    lam_nm_f = lam_nm_f[idx]
    T = np.clip(T_pct_f[idx], 0.0, 100.0) / 100.0
    return np.interp(np.asarray(lam_um) * 1e3, lam_nm_f, T,
                     left=0.0, right=0.0)


def _filter_spec_active(shape, bw_nm, file):
    """True iff a filter is actually being applied."""
    if shape == "file":
        return file is not None
    if shape == "none":
        return False
    return (bw_nm or 0) > 0


def _build_filter(lam_um, *, center_um, bw_um, shape, file):
    """Internal: build the filter array on `lam_um` from the unified spec."""
    return make_filter(lam_um, center_um, bw_um, shape, file=file)


def brightness_vs_T(
    *,
    T_range, dt,
    filter_center_nm, filter_bw_nm, filter_shape="rect",
    filter_file=None,
    wp=None, ws=None,
    lam_range_nm=None, n_lam=400,
    n_z=64, n_i=0,
    fwhm_nm=None,
    verbose=True,
):
    """Coincidence brightness B(T) with the **same filter on both arms**.

    For each (center, BW, shape) we compute

        B(T) = ∫∫ dλ_s dλ_i  f(λ_s) f(λ_i) |α(λ_p)|² |C₀₀(λ_s, λ_i; T)|²

    where the same scalar filter function f is applied to both the
    signal and idler axes — i.e. identical bandpass filters in front of
    each detector.  Several filter widths can be passed as a list and
    are overlaid on the plot.
    """
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

    lam_um     = hm["lam_nm"] * 1e-3                # signal axis [µm], (N_lam,)
    lam_i_um   = hm["lam_i_grid"]                   # idler axis  [µm], (N_lam, n_i)
    spectra    = hm["spectra"]                      # unfiltered reference
    c00sq_3d   = hm["c00sq_3d"]                     # (N_T, N_lam, n_i)
    alpha_3d   = hm["alpha_3d"]                     # (N_T, N_lam, n_i)
    n_i_actual = hm["n_i"]

    # Signal-axis trapezoidal weights
    dlam_s_um    = lam_um[1] - lam_um[0]
    w_s          = np.full(n_lam, dlam_s_um); w_s[0] *= 0.5; w_s[-1] *= 0.5

    # Per-(λ_s) idler sub-grid spacing & trapezoidal weights
    dlam_i_per_s = (lam_i_um[:, -1] - lam_i_um[:, 0]) / max(n_i_actual - 1, 1)  # (N_lam,)
    w_i          = np.ones(n_i_actual); w_i[0] = 0.5; w_i[-1] = 0.5

    if filter_shape == "none" or filter_shape == "file":
        bw_um = np.array([0.0])

    N_F = len(bw_um)
    filters    = np.empty((N_F, n_lam))                # signal-axis filter (for plot overlay)
    brightness = np.empty((N_F, len(hm["T"])))
    for iF, bw in enumerate(bw_um):
        if filter_shape == "none" and filter_file is None:
            f_s = np.ones_like(lam_um)
            f_i = np.ones_like(lam_i_um)
        else:
            f_s = _build_filter(lam_um,   center_um=ctr_um, bw_um=bw,
                                shape=filter_shape, file=filter_file)
            f_i = _build_filter(lam_i_um, center_um=ctr_um, bw_um=bw,
                                shape=filter_shape, file=filter_file)
        filters[iF] = f_s
        # Per-T, per-λ_s idler-arm integral
        inner_si = np.einsum("tsi,i->ts",
                             alpha_3d * c00sq_3d * f_i[None, :, :],
                             w_i) * dlam_i_per_s[None, :]
        # Then signal-arm integral
        brightness[iF] = (inner_si * (f_s * w_s)[None, :]).sum(axis=1)

    return dict(
        T=hm["T"], lam_nm=hm["lam_nm"],
        spectra=spectra,
        brightness=brightness,
        filters=filters,
        filter_bw_nm=np.asarray(filter_bw_nm, dtype=np.float64),
        filter_center_nm=filter_center_nm,
        filter_shape=filter_shape,
        filter_file=filter_file,
        params=dict(wp=wp, ws=ws,
                    n_lam=n_lam, n_z=n_z,
                    fwhm_nm=fwhm_nm, dt=dt,
                    filter_file=filter_file),
    )
