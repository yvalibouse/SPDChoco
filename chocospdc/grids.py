"""
chocospdc.grids — Spectral grid builders for broadband-pump SPDC.

All builders use vectorised Sellmeier calls (no Python scalar loops over
wavelength) and return C-contiguous float64 arrays ready to pass to the
C kernel.

Three builders are provided:

  * :func:`build_uv_grid`
        Single-temperature, rotated (u, v) mesh.  u = Δλ_s + Δλ_i (pump
        direction), v = Δλ_s − Δλ_i (PM direction).  Used by the 2-D
        waist scan.

  * :func:`build_heatmap_grid`
        Temperature × signal-wavelength × idler-wavelength grid, with
        per-(λ_s) idler sub-grid centred on energy conservation.  Used
        by the spectrum-vs-T heatmap, waterfall and brightness scripts.

  * :func:`build_modes_grid`
        Single temperature, (λ_s × λ_i) grid with per-block trapezoidal
        weights and α² envelope folded in.  Used by the mode-resolved
        spectrum kernel.
"""

import numpy as np

from .sellmeier import n_pump, n_signal, n_idler


# ──────────────────────────────────────────────────────────────────────
#  CORE PRIMITIVE — (λ_s, λ_i) → (k_s, k_i, k_p, Δk)
# ──────────────────────────────────────────────────────────────────────
def lam_to_k(lam_s, lam_i, T, crystal, spdc_type, KG):
    """Vectorised conversion from wavelengths to wave vectors and Δk.

    The pump wavelength is set by energy conservation:
        λ_p = λ_s λ_i / (λ_s + λ_i).
    """
    lam_s = np.asarray(lam_s, dtype=np.float64)
    lam_i = np.asarray(lam_i, dtype=np.float64)
    twopi = 2.0 * np.pi

    ns = n_signal(lam_s, T, crystal, spdc_type)
    ni = n_idler (lam_i, T, crystal, spdc_type)
    k_s = twopi * ns / lam_s
    k_i = twopi * ni / lam_i

    lam_p = lam_s * lam_i / (lam_s + lam_i)
    n_p   = n_pump(lam_p, T, crystal, spdc_type)
    k_p   = twopi * n_p / lam_p

    dk = k_p - k_s - k_i - KG
    return k_s, k_i, k_p, dk


# ──────────────────────────────────────────────────────────────────────
#  (u, v) ROTATED GRID — single T, broadband pump
# ──────────────────────────────────────────────────────────────────────
def build_uv_grid(n_v, half_range, sigma_pump,
                  T, crystal, spdc_type,
                  lambda0_p, lambda0_s, lambda0_i,
                  mismatch):
    """Rotated (u, v) grid for the broadband 2-D waist scan.

    Parameters
    ----------
    n_v : int
        Number of grid points along v (the PM direction).
    half_range : float
        Half-width along v [µm].
    sigma_pump : float
        4·(field 1/e half-width).  Grid extent uses sigma_pump; the
        physical envelope |α|² uses sigma_pump/4.
    mismatch : float
        −KG.  See :mod:`chocospdc.config`.

    Returns
    -------
    k_s, k_i, k_p, dk, w_spec, alpha_sq, lam_s, lam_i : ndarray (N_active,)
        Only points with |α|² above threshold are returned.  ``lam_s``
        and ``lam_i`` are the physical signal/idler wavelengths at each
        active spectral point [µm] — used for symmetric filtering.
    N_u, N_v : int
        Full grid dimensions before threshold filtering.
    """
    v_max = half_range
    u_max = 4.0 * sigma_pump
    n_u = max(7, int(np.ceil(2.0 * u_max / (sigma_pump / 4.0))))
    if n_u % 2 == 0:
        n_u += 1

    u_arr = np.linspace(-u_max, u_max, n_u)
    v_arr = np.linspace(-v_max, v_max, n_v)
    du, dv = u_arr[1] - u_arr[0], v_arr[1] - v_arr[0]

    wu = np.full(n_u, du); wu[0] *= 0.5; wu[-1] *= 0.5
    wv = np.full(n_v, dv); wv[0] *= 0.5; wv[-1] *= 0.5

    pump_env = np.exp(-2.0 * (u_arr / sigma_pump)**2)
    active_u = np.where(pump_env >= 1e-10)[0]

    iu_grid, iv_grid = np.meshgrid(active_u, np.arange(n_v), indexing="ij")
    iu, iv = iu_grid.ravel(), iv_grid.ravel()

    u, v = u_arr[iu], v_arr[iv]
    dls, dli = 0.5 * (u + v), 0.5 * (u - v)
    lam_s = lambda0_s + dls
    lam_i = lambda0_i + dli

    valid = (lam_s > 0) & (lam_i > 0)
    lam_s, lam_i = lam_s[valid], lam_i[valid]
    iu, iv = iu[valid], iv[valid]

    twopi = 2.0 * np.pi
    ns = n_signal(lam_s, T, crystal, spdc_type)
    ni = n_idler (lam_i, T, crystal, spdc_type)
    k_s = twopi * ns / lam_s
    k_i = twopi * ni / lam_i

    lam_p = lam_s * lam_i / (lam_s + lam_i)
    n_p   = n_pump(lam_p, T, crystal, spdc_type)
    k_p   = twopi * n_p / lam_p
    dk    = k_p - k_s - k_i + mismatch

    # |α|² normalised so ∫|α|² dλ_p = 1.
    s = sigma_pump / 4.0
    norm_sq  = np.sqrt(2.0 / np.pi) / s
    alpha_sq = norm_sq * np.exp(-2.0 * ((lam_p - lambda0_p) / s)**2)

    w_spec = 0.5 * wu[iu] * wv[iv]

    arrays = (k_s, k_i, k_p, dk, w_spec, alpha_sq, lam_s, lam_i)
    arrays = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in arrays)
    return arrays + (n_u, n_v)


# ──────────────────────────────────────────────────────────────────────
#  HEATMAP GRID — T × λ_s × λ_i, broadband pump
# ──────────────────────────────────────────────────────────────────────
def build_heatmap_grid(T_arr, lam_s_arr, sigma_lp, n_i,
                       crystal, spdc_type, lambda0_p, KG):
    """(T, λ_s, λ_i) flat grid with α² envelope.

    n_i = 0 → auto-pick number of idler quadrature points.

    Returns ``(k_s, k_i, k_p, dk, alpha_sq, n_i, lam_i_2d)`` where the
    first five arrays each have length ``N_T × N_lam × n_i`` and
    ``lam_i_2d`` (shape ``N_lam × n_i``) holds the per-(λ_s) idler grid
    — needed for symmetric filtering and proper sub-grid integration.
    """
    T_arr     = np.asarray(T_arr, dtype=np.float64)
    lam_s_arr = np.asarray(lam_s_arr, dtype=np.float64)
    N_T, N_lam = len(T_arr), len(lam_s_arr)

    if n_i <= 0:
        n_i = min(101, max(21, int(np.ceil(32 * sigma_lp / 1e-4))))
    if n_i % 2 == 0:
        n_i += 1

    twopi  = 2.0 * np.pi
    half_i = 16.0 * sigma_lp

    # Per-λ_s idler sub-grid centred on energy conservation
    lam_i_0 = lam_s_arr * lambda0_p / (lam_s_arr - lambda0_p)
    li_lo   = np.maximum(lam_i_0 - half_i, 0.5 * lambda0_p)
    li_hi   = lam_i_0 + half_i

    t_frac   = np.linspace(0.0, 1.0, n_i)
    lam_i_2d = li_lo[:, None] + (li_hi - li_lo)[:, None] * t_frac
    lam_s_2d = np.broadcast_to(lam_s_arr[:, None], (N_lam, n_i))
    lam_p_2d = lam_s_2d * lam_i_2d / (lam_s_2d + lam_i_2d)

    norm_sq     = 1.0 / (sigma_lp * np.sqrt(np.pi))
    alpha_sq_2d = norm_sq * np.exp(-((lam_p_2d - lambda0_p) / sigma_lp)**2)

    lam_s_flat = np.ascontiguousarray(lam_s_2d.ravel())
    lam_i_flat = np.ascontiguousarray(lam_i_2d.ravel())
    lam_p_flat = np.ascontiguousarray(lam_p_2d.ravel())
    alpha_flat = alpha_sq_2d.ravel()
    M          = N_lam * n_i

    N_tot = N_T * M
    k_s_out, k_i_out, k_p_out, dk_out = (np.empty(N_tot) for _ in range(4))
    alpha_out = np.empty(N_tot)

    for iT, T_val in enumerate(T_arr):
        off = iT * M
        ns = n_signal(lam_s_flat, T_val, crystal, spdc_type)
        ni = n_idler (lam_i_flat, T_val, crystal, spdc_type)
        npx = n_pump (lam_p_flat, T_val, crystal, spdc_type)

        ks = twopi * ns / lam_s_flat
        ki = twopi * ni / lam_i_flat
        kp = twopi * npx / lam_p_flat

        k_s_out[off:off + M] = ks
        k_i_out[off:off + M] = ki
        k_p_out[off:off + M] = kp
        dk_out [off:off + M] = kp - ks - ki - KG
        alpha_out[off:off + M] = alpha_flat

    return (k_s_out, k_i_out, k_p_out, dk_out, alpha_out, n_i,
            np.ascontiguousarray(lam_i_2d))


# ──────────────────────────────────────────────────────────────────────
#  MODES GRID — single T, (λ_s × λ_i) with combined weights
# ──────────────────────────────────────────────────────────────────────
def build_modes_grid(lam_s_arr, sigma_lp, n_i,
                     T, crystal, spdc_type, lambda0_p, KG):
    """Single-T (λ_s × λ_i) grid with weights = dλ_i × trap × α².

    Used by :func:`compute.modes_spectrum`.  The combined weights are
    ready to be summed inside the C kernel.
    """
    lam_s_arr = np.asarray(lam_s_arr, dtype=np.float64)
    N_lam = len(lam_s_arr)

    if n_i <= 0:
        n_i = min(101, max(21, int(np.ceil(32 * sigma_lp / 1e-4))))
    if n_i % 2 == 0:
        n_i += 1

    twopi  = 2.0 * np.pi
    half_i = 16.0 * sigma_lp

    lam_i_0 = lam_s_arr * lambda0_p / (lam_s_arr - lambda0_p)
    li_lo   = np.maximum(lam_i_0 - half_i, 0.5 * lambda0_p)
    li_hi   = lam_i_0 + half_i

    t_frac   = np.linspace(0.0, 1.0, n_i)
    lam_i_2d = li_lo[:, None] + (li_hi - li_lo)[:, None] * t_frac
    lam_s_2d = np.broadcast_to(lam_s_arr[:, None], (N_lam, n_i))

    dli_1d      = (li_hi - li_lo) / (n_i - 1)
    trap        = np.ones(n_i); trap[0] = 0.5; trap[-1] = 0.5

    lam_p_2d    = lam_s_2d * lam_i_2d / (lam_s_2d + lam_i_2d)
    norm_sq     = 1.0 / (sigma_lp * np.sqrt(np.pi))
    alpha_sq_2d = norm_sq * np.exp(-((lam_p_2d - lambda0_p) / sigma_lp)**2)

    weights_2d  = dli_1d[:, None] * trap[None, :] * alpha_sq_2d

    lam_s_flat = np.ascontiguousarray(lam_s_2d.ravel())
    lam_i_flat = np.ascontiguousarray(lam_i_2d.ravel())
    lam_p_flat = np.ascontiguousarray(lam_p_2d.ravel())

    ns = n_signal(lam_s_flat, T, crystal, spdc_type)
    ni = n_idler (lam_i_flat, T, crystal, spdc_type)
    npx = n_pump (lam_p_flat, T, crystal, spdc_type)

    k_s = twopi * ns / lam_s_flat
    k_i = twopi * ni / lam_i_flat
    k_p = twopi * npx / lam_p_flat
    dk  = k_p - k_s - k_i - KG

    arrays = (k_s, k_i, k_p, dk, weights_2d.ravel())
    arrays = tuple(np.ascontiguousarray(a, dtype=np.float64) for a in arrays)
    return arrays + (n_i, np.ascontiguousarray(lam_i_2d))
