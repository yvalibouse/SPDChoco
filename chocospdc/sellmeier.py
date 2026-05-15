"""
chocospdc.sellmeier — Multi-crystal Sellmeier equations.

Pure NumPy, vectorised.  Same numerical values as the legacy module —
fully drop-in compatible.

Crystals supported:
    CRYSTAL_KTP        Kato & Takaoka 2002
    CRYSTAL_KTP_F      Fradkin et al. 1999 / redoptronics
    CRYSTAL_BBO        Kato & Takaoka 2002
    CRYSTAL_PPLN       Gayer 2008 (e-ray) + Zelmon 1997 (o-ray)
    CRYSTAL_PPLN_MgO   Gayer 2008 (both rays, 5 mol% MgO-doped cLN)

SPDC types:
    TYPE_0   e → e + e
    TYPE_I   e → o + o
    TYPE_II  e → e + o   (signal extraordinary, idler ordinary)
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  Identifiers
# ──────────────────────────────────────────────────────────────────────
CRYSTAL_KTP, CRYSTAL_BBO, CRYSTAL_PPLN, CRYSTAL_KTP_F, CRYSTAL_PPLN_MgO = range(5)
TYPE_0, TYPE_I, TYPE_II = 0, 1, 2

CRYSTAL_NAMES = {
    CRYSTAL_KTP:      "KTP (Kato)",
    CRYSTAL_KTP_F:    "KTP (Fradkin)",
    CRYSTAL_BBO:      "BBO",
    CRYSTAL_PPLN:     "PPLN",
    CRYSTAL_PPLN_MgO: "MgO:PPLN",
}
TYPE_NAMES = {
    TYPE_0:  "Type 0 (e→ee)",
    TYPE_I:  "Type I (e→oo)",
    TYPE_II: "Type II (e→eo)",
}


# ──────────────────────────────────────────────────────────────────────
#  KTP — Kato & Takaoka 2002
# ──────────────────────────────────────────────────────────────────────
_KTP_SELL = np.array([
    [2.16747, 0.83733, 0.04611, 0.01713],
    [2.19229, 0.83547, 0.04970, 0.01621],
    [2.25411, 1.06543, 0.05486, 0.02140],
])
_KTP_THERMO = np.array([
    [0.1717, -0.5353, 0.8416,  0.1201],
    [0.1997, -0.4063, 0.5154,  0.5425],
    [0.9221, -2.9220, 3.6677, -0.1897],
])
_KTP_T_REF = 25.0


def _ktp(lam, T, axis):
    A, B, C, D = _KTP_SELL[axis]
    t0, t1, t2, t3 = _KTP_THERMO[axis]
    n0 = np.sqrt(A + B * lam**2 / (lam**2 - C) - D * lam**2)
    dn = (t0 / lam**3 + t1 / lam**2 + t2 / lam + t3) * 1e-5
    return n0 + dn * (T - _KTP_T_REF)


# ──────────────────────────────────────────────────────────────────────
#  KTP-Fradkin — Fradkin et al. 1999 / redoptronics
# ──────────────────────────────────────────────────────────────────────
_KTPF_X       = (2.10468, 0.89342, 0.04438, 0.01036)
_KTPF_Y_SHORT = (2.14559, 0.87629, 0.0485,  0.01173)
_KTPF_Y_LONG  = (2.0993,  0.922683, 0.0467695, 0.0138408)
_KTPF_Z       = (1.9446,  1.3617,  0.047,   0.01491)
_KTPF_DNDT    = (1.1e-5, 1.3e-5, 1.6e-5)
_KTPF_T_REF   = 20.0


def _ktpf_sellmeier(lam2, coeffs):
    A, B, C, D = coeffs
    return np.sqrt(A + B * lam2 / (lam2 - C) - D * lam2)


def _ktpf(lam, T, axis):
    lam2 = lam * lam
    if axis == 0:
        n_ref = _ktpf_sellmeier(lam2, _KTPF_X)
    elif axis == 1:                       # y-axis: piecewise in wavelength
        n_short = _ktpf_sellmeier(lam2, _KTPF_Y_SHORT)
        n_long  = _ktpf_sellmeier(lam2, _KTPF_Y_LONG)
        n_ref   = np.where(np.asarray(lam) < 1.2, n_short, n_long)
    else:
        n_ref = _ktpf_sellmeier(lam2, _KTPF_Z)
    return n_ref + _KTPF_DNDT[axis] * (T - _KTPF_T_REF)


# ──────────────────────────────────────────────────────────────────────
#  BBO — Kato & Takaoka 2002
# ──────────────────────────────────────────────────────────────────────
_BBO_SELL  = np.array([
    [2.7359, 0.01878, 0.01822, 0.01354],   # ordinary
    [2.3753, 0.01224, 0.01667, 0.01516],   # extraordinary
])
_BBO_DNDT  = (-16.6e-6, -9.3e-6)
_BBO_T_REF = 20.0


def _bbo(lam, T, pol):
    A, B, C, D = _BBO_SELL[pol]
    n0 = np.sqrt(A + B / (lam**2 - C) - D * lam**2)
    return n0 + _BBO_DNDT[pol] * (T - _BBO_T_REF)


# ──────────────────────────────────────────────────────────────────────
#  PPLN — Gayer 2008 (e-ray)  +  Zelmon 1997 (o-ray)
# ──────────────────────────────────────────────────────────────────────
_PPLN_E    = (5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2)
_PPLN_E_TH = (2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4)

_PPLN_O      = (2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60)
_PPLN_O_DNDT = -0.9e-6
_PPLN_T_REF  = 24.5


def _ppln_e(lam, T):
    a1, a2, a3, a4, a5, a6 = _PPLN_E
    b1, b2, b3, b4 = _PPLN_E_TH
    f = (T - 24.5) * (T + 570.5)
    n2 = (a1 + b1 * f
          + (a2 + b2 * f) / (lam**2 - (a3 + b3 * f)**2)
          + (a4 + b4 * f) / (lam**2 - a5**2)
          - a6 * lam**2)
    return np.sqrt(n2)


def _ppln_o(lam, T):
    B1, C1, B2, C2, B3, C3 = _PPLN_O
    l2 = lam**2
    n0 = np.sqrt(1.0
                 + B1 * l2 / (l2 - C1)
                 + B2 * l2 / (l2 - C2)
                 + B3 * l2 / (l2 - C3))
    return n0 + _PPLN_O_DNDT * (T - _PPLN_T_REF)


# ──────────────────────────────────────────────────────────────────────
#  MgO:PPLN — Gayer et al., APB 91, 343 (2008)  (5 mol% MgO-doped cLN)
#  e-ray identical to undoped (_ppln_e); o-ray has its own coefficients.
# ──────────────────────────────────────────────────────────────────────
_PPLN_MGO_O    = (5.653,    0.1185,    0.2091,   89.61,   10.85,  1.97e-2)
_PPLN_MGO_O_TH = (7.941e-7, 3.134e-8, -4.641e-9, -2.188e-6)


def _ppln_mgo_o(lam, T):
    a1, a2, a3, a4, a5, a6 = _PPLN_MGO_O
    b1, b2, b3, b4 = _PPLN_MGO_O_TH
    f = (T - 24.5) * (T + 570.5)
    n2 = (a1 + b1 * f
          + (a2 + b2 * f) / (lam**2 - (a3 + b3 * f)**2)
          + (a4 + b4 * f) / (lam**2 - a5**2)
          - a6 * lam**2)
    return np.sqrt(n2)


# ──────────────────────────────────────────────────────────────────────
#  Unified dispatch
# ──────────────────────────────────────────────────────────────────────
_N_O_DISPATCH = {
    CRYSTAL_KTP:      lambda l, T: _ktp(l, T, 2),
    CRYSTAL_KTP_F:    lambda l, T: _ktpf(l, T, 2),
    CRYSTAL_BBO:      lambda l, T: _bbo(l, T, 0),
    CRYSTAL_PPLN:     _ppln_o,
    CRYSTAL_PPLN_MgO: _ppln_mgo_o,
}
_N_E_DISPATCH = {
    CRYSTAL_KTP:      lambda l, T: _ktp(l, T, 1),
    CRYSTAL_KTP_F:    lambda l, T: _ktpf(l, T, 1),
    CRYSTAL_BBO:      lambda l, T: _bbo(l, T, 1),
    CRYSTAL_PPLN:     _ppln_e,
    CRYSTAL_PPLN_MgO: _ppln_e,
}


def n_o(lam, T, crystal):
    """Ordinary refractive index."""
    return _N_O_DISPATCH[crystal](lam, T)


def n_e(lam, T, crystal):
    """Extraordinary refractive index."""
    return _N_E_DISPATCH[crystal](lam, T)


def n_pump(lam, T, crystal, spdc_type):
    """Pump always travels as extraordinary."""
    return n_e(lam, T, crystal)


def n_signal(lam, T, crystal, spdc_type):
    if spdc_type == TYPE_I:
        return n_o(lam, T, crystal)
    return n_e(lam, T, crystal)         # TYPE_0 and TYPE_II: signal = e


def n_idler(lam, T, crystal, spdc_type):
    if spdc_type == TYPE_0:
        return n_e(lam, T, crystal)
    return n_o(lam, T, crystal)         # TYPE_I and TYPE_II: idler = o


def chi2_profile(z):
    """χ⁽²⁾(z) — placeholder for a longitudinal nonlinearity profile.
    Default: uniform (1.0).  Replace with e.g. a Gaussian apodisation
    by editing this function in your local copy."""
    return 1.0
