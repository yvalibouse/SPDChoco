"""
chocospdc.quadrature — Gauss-Legendre quadrature helpers.

Builds the z-, r- and χ(z)-arrays that every C-kernel call needs.
Centralised here so scripts don't repeat the same five lines.
"""

import numpy as np

from .sellmeier import chi2_profile


def gauss_legendre_z(L: float, n_z: int):
    """Return (z_nodes, z_weights, chi_z) all C-contiguous float64.

    The interval is [-L/2, L/2], so for a crystal of length L the
    z-integral is automatically inside the crystal.  χ(z) is evaluated
    at every node using the user-overridable ``chi2_profile``.
    """
    ref, w = np.polynomial.legendre.leggauss(n_z)
    z_nodes   = np.ascontiguousarray(0.5 * L * ref)
    z_weights = np.ascontiguousarray(0.5 * L * w)
    chi_z = np.array([chi2_profile(z) for z in z_nodes], dtype=np.float64)
    chi_z = np.ascontiguousarray(chi_z)
    return z_nodes, z_weights, chi_z


def gauss_legendre_r(n_r: int):
    """Return (r_ref, r_weights) on the reference interval [-1, 1]."""
    ref, w = np.polynomial.legendre.leggauss(n_r)
    return np.ascontiguousarray(ref), np.ascontiguousarray(w)
