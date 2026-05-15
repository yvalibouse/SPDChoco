"""
chocospdc.config — User-editable experiment configuration.
═════════════════════════════════════════════════════════════════════

Edit the USER PARAMETERS block.  All derived quantities (refractive
indices, QPM grating, recommended spectral grid, etc.) are computed
automatically from those.

Verify the configuration with:
    python -m chocospdc.config
"""

import numpy as np
from .sellmeier import (
    n_pump, n_signal, n_idler,
    CRYSTAL_KTP, CRYSTAL_KTP_F, CRYSTAL_BBO, CRYSTAL_PPLN, CRYSTAL_PPLN_MgO,
    TYPE_0, TYPE_I, TYPE_II,
    CRYSTAL_NAMES, TYPE_NAMES,
)


# ═════════════════════════════════════════════════════════════════════
#  USER PARAMETERS
# ═════════════════════════════════════════════════════════════════════

CRYSTAL   = CRYSTAL_KTP_F      # KTP | KTP_F | BBO | PPLN | PPLN_MgO
SPDC_TYPE = TYPE_II            # TYPE_0 (e→ee) | TYPE_I (e→oo) | TYPE_II (e→eo)

T   = 30.0                     # working crystal temperature  [°C]
L   = 30_000.0                 # crystal length               [µm]

# Central wavelengths [µm]
lambda0_p = 0.775
lambda0_s = 1.55
lambda0_i = 1.55

# Beam waists (1/e² intensity radius) at the crystal centre [µm]
w0_p = 25.0                    # pump
w0_s = 30.0                    # signal collection mode
w0_i = 30.0                    # idler  collection mode

# Pump spectral FWHM (intensity) [µm].  Must be > 0.
# For a narrow-linewidth CW laser use e.g. 1e-6 µm = 1 pm.
FWHM_PUMP = 1e-5

# Temperature at which the QPM grating was designed.
# Set equal to T for perfect phase matching at degeneracy.
T_QPM = 20.0

# Longitudinal offset of the collection-mode waist from the crystal
# centre  [µm].  Positive = towards the output face.
OFFSET_W0 = 0.0


# ═════════════════════════════════════════════════════════════════════
#  DERIVED QUANTITIES  (do not edit below)
# ═════════════════════════════════════════════════════════════════════

if FWHM_PUMP <= 0:
    raise ValueError("FWHM_PUMP must be > 0 (every real pump has finite BW).")

# Indices and wave vectors at the central wavelengths
n0_p = n_pump  (lambda0_p, T, CRYSTAL, SPDC_TYPE)
n0_s = n_signal(lambda0_s, T, CRYSTAL, SPDC_TYPE)
n0_i = n_idler (lambda0_i, T, CRYSTAL, SPDC_TYPE)

k0_p = 2.0 * np.pi * n0_p / lambda0_p
k0_s = 2.0 * np.pi * n0_s / lambda0_s
k0_i = 2.0 * np.pi * n0_i / lambda0_i

offset_w0 = OFFSET_W0


# ── QPM grating vector ──────────────────────────────────────────────
_n_p_q = n_pump  (lambda0_p, T_QPM, CRYSTAL, SPDC_TYPE)
_n_s_q = n_signal(lambda0_s, T_QPM, CRYSTAL, SPDC_TYPE)
_n_i_q = n_idler (lambda0_i, T_QPM, CRYSTAL, SPDC_TYPE)

_dk_q = (2 * np.pi) * (_n_p_q / lambda0_p
                       - _n_s_q / lambda0_s
                       - _n_i_q / lambda0_i)

Lambda_poling = 2.0 * np.pi / abs(_dk_q)
KG       = np.sign(_dk_q) * 2.0 * np.pi / Lambda_poling
mismatch = -KG                           # legacy alias (Δk_eff = k_p − k_s − k_i + mismatch)

dk_degeneracy = k0_p - k0_s - k0_i + mismatch


# ── Pump bandwidth conversions ──────────────────────────────────────
# |α(λ_p)|² = exp(−(Δλ/σ_lp)²)   →   FWHM(intensity) = 2·σ_lp·√ln2
SIGMA_LP   = FWHM_PUMP / (2.0 * np.sqrt(np.log(2)))
# Convention used by grids.build_grid for the rotated (u,v) mesh:
SIGMA_PUMP = 4.0 * FWHM_PUMP / np.sqrt(2.0 * np.log(2))


# ── Spectral half-range estimate (GVM- or GVD-limited) ──────────────
_dl   = 1e-5
_ng_s = n0_s - lambda0_s * (n_signal(lambda0_s + _dl, T, CRYSTAL, SPDC_TYPE)
                            - n_signal(lambda0_s - _dl, T, CRYSTAL, SPDC_TYPE)) / (2 * _dl)
_ng_i = n0_i - lambda0_i * (n_idler (lambda0_i + _dl, T, CRYSTAL, SPDC_TYPE)
                            - n_idler (lambda0_i - _dl, T, CRYSTAL, SPDC_TYPE)) / (2 * _dl)
GVM_si = abs(_ng_i - _ng_s)

if GVM_si > 1e-4:                                       # GVM-limited
    sinc_fwhm           = 0.886 * lambda0_s**2 / (GVM_si * L)
    spectral_half_range = max(10.0 * sinc_fwhm, 0.001)
    _bw_regime          = "GVM-limited"
    _beta2_s            = None
else:                                                   # GVD-limited
    _d2n = (n_signal(lambda0_s + _dl, T, CRYSTAL, SPDC_TYPE)
            - 2.0 * n0_s
            + n_signal(lambda0_s - _dl, T, CRYSTAL, SPDC_TYPE)) / _dl**2
    _beta2_s = lambda0_s**3 / (2.0 * np.pi) * abs(_d2n)
    if _beta2_s > 1e-12:
        sinc_fwhm = (lambda0_s**2 / np.pi) * np.sqrt(2.783 / (_beta2_s * L))
    else:
        sinc_fwhm = 0.006
    spectral_half_range = max(10.0 * sinc_fwhm, 0.020)
    _bw_regime          = "GVD-limited"

n_lambda_recommended = max(150, int(2 * spectral_half_range / (sinc_fwhm / 10)))


# ── Focusing parameters ξ = L/(2·z_R) ───────────────────────────────
xi_p = L * lambda0_p / (2.0 * np.pi * n0_p * w0_p**2)
xi_s = L * lambda0_s / (2.0 * np.pi * n0_s * w0_s**2)
xi_i = L * lambda0_i / (2.0 * np.pi * n0_i * w0_i**2)


# ═════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═════════════════════════════════════════════════════════════════════

def summary():
    """Pretty-print the current configuration with phase-matching warning."""
    line = "═" * 60
    print(f"{line}\n  SPDC source configuration\n{line}")
    print(f"  Crystal     : {CRYSTAL_NAMES[CRYSTAL]}")
    print(f"  SPDC type   : {TYPE_NAMES[SPDC_TYPE]}")
    print(f"  Temperature : {T} °C")
    print(f"  Length      : {L / 1e3:.1f} mm   ({L:.0f} µm)")
    print(f"  λ_p = {lambda0_p*1e3:7.2f} nm   n_p = {n0_p:.6f}   k_p = {k0_p:.4f} µm⁻¹")
    print(f"  λ_s = {lambda0_s*1e3:7.2f} nm   n_s = {n0_s:.6f}   k_s = {k0_s:.4f} µm⁻¹")
    print(f"  λ_i = {lambda0_i*1e3:7.2f} nm   n_i = {n0_i:.6f}   k_i = {k0_i:.4f} µm⁻¹")
    print(f"  QPM design T: {T_QPM} °C   →   Λ = {Lambda_poling:.3f} µm")
    print(f"  Δk(degen.)  : {dk_degeneracy:+.6e} µm⁻¹  "
          f"({dk_degeneracy * L / 2:+.2f} rad)")
    print(f"  Pump BW     : FWHM = {FWHM_PUMP*1e3:.3f} nm   "
          f"(σ_λp = {SIGMA_LP*1e3:.3f} nm)")
    print(f"  w_p = {w0_p:.0f} µm  (ξ_p = {xi_p:.3f})")
    print(f"  w_s = {w0_s:.0f} µm  (ξ_s = {xi_s:.3f})")
    print(f"  w_i = {w0_i:.0f} µm  (ξ_i = {xi_i:.3f})")
    if _beta2_s is None:
        print(f"  GVM_si      : {GVM_si:.4e}   ({_bw_regime})")
    else:
        print(f"  β₂(λ_s)     : {_beta2_s:.4e} µm²   ({_bw_regime})")
    print(f"  Sinc FWHM   : {sinc_fwhm*1e3:.3f} nm")
    print(f"  Suggested λ_s half-range : ±{spectral_half_range*1e3:.2f} nm  "
          f"({n_lambda_recommended} pts)")
    if abs(dk_degeneracy * L / 2) > 1.0:
        print()
        print(f"  ⚠  Δk·L/2 = {dk_degeneracy * L / 2:+.1f} rad ≫ π.")
        print(f"     Source is NOT phase-matched at degeneracy.")
        print(f"     To fix at degeneracy, set T_QPM = {T} in config.py.")
    print(line)


if __name__ == "__main__":
    summary()
