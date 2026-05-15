"""
chocospdc.beams — Gaussian beam helpers, vectorised.

All functions accept scalar or array inputs (k, w0 may be arrays of the
same shape as z) and return matching shapes.
"""

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0
try:
    trapz = np.trapezoid           # type: ignore[attr-defined]
except AttributeError:
    trapz = np.trapz               # type: ignore[attr-defined]


def beam_w(z, k, w0):
    """Beam radius w(z) inside the crystal."""
    zR = 0.5 * k * w0 * w0
    return w0 * np.sqrt(1.0 + (z / zR)**2)


def inv_R(z, k, w0):
    """1/R(z) — wavefront curvature.  Safe at z = 0."""
    zR = 0.5 * k * w0 * w0
    return z / (z * z + zR * zR + 1e-30)


def gouy(z, k, w0):
    """Fundamental Gouy phase ψ_0(z) = arctan(z / z_R)."""
    zR = 0.5 * k * w0 * w0
    return np.arctan2(z, zR)


def rayleigh(k, w0):
    """Rayleigh range z_R = ½ k w0²."""
    return 0.5 * k * w0 * w0


def focusing_xi(L, k, w0):
    """Focusing parameter ξ = L / (2 z_R)."""
    return L / (k * w0 * w0)
