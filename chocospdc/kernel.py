"""
chocospdc.kernel — auto-build and load the C kernel (libspdc).

A single `kernel` module replaces the ctypes / build boilerplate that
used to be copy-pasted across every script.  The compiled library is
loaded lazily on first access via :func:`get_lib`.

Entry points (set up in `_register`):

    scan_waists                2-D waist scan (η, B maps)
    compute_single             single waist pair (S2, S1_s, S1_i)
    compute_c00_heatmap        |C₀₀|² on a flat (k_s, k_i, k_p, dk) array
    compute_modes_spectrum     mode-resolved P_{0,p_i}, P_{p_s,0}(λ_s)
    compute_c00_integrated     ∫dλ_i |α|² |C₀₀|²  (no large intermediate)
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from numpy.ctypeslib import ndpointer


# ──────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────
_PKG_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_PKG_DIR)
_SRC      = os.path.join(_PROJ_DIR, "spdc_kernel.c")
_IS_WIN   = platform.system() == "Windows"
_LIBNAME  = "libspdc.dll" if _IS_WIN else "libspdc.so"
_LIB      = os.path.join(_PROJ_DIR, _LIBNAME)


# ──────────────────────────────────────────────────────────────────────
#  Build
# ──────────────────────────────────────────────────────────────────────
def _find_gcc() -> str | None:
    gcc = shutil.which("gcc")
    if gcc:
        return gcc
    if _IS_WIN:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = os.path.join(conda_prefix, "Library",
                                     "mingw-w64", "bin", "gcc.exe")
            if os.path.isfile(candidate):
                return candidate
    return None


def build(force: bool = False) -> str:
    """Compile spdc_kernel.c → libspdc.{so,dll}.  Returns the library path."""
    if (not force
            and os.path.exists(_LIB)
            and os.path.getmtime(_SRC) <= os.path.getmtime(_LIB)):
        return _LIB

    gcc = _find_gcc()
    if gcc is None:
        sys.exit(
            "\nERROR: gcc not found.\n"
            "  Linux  : sudo apt install gcc libgomp1\n"
            "  Windows: conda install -c conda-forge m2w64-toolchain\n"
        )

    arch_flag = "-march=haswell" if _IS_WIN else "-march=native"
    cmd = [gcc, "-O3", arch_flag, "-ffast-math", "-fopenmp",
           "-shared", _SRC, "-o", _LIB]
    if not _IS_WIN:
        cmd.insert(6, "-fPIC")
        cmd.append("-lm")

    print(f"Building C kernel …  {' '.join(cmd)}")
    t0 = time.time()
    subprocess.check_call(cmd)
    print(f"  compiled in {time.time() - t0:.2f} s\n")
    return _LIB


# ──────────────────────────────────────────────────────────────────────
#  ctypes binding
# ──────────────────────────────────────────────────────────────────────
_dp   = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
_dp_w = ndpointer(ctypes.c_double, flags=("C_CONTIGUOUS", "WRITEABLE"))


def _register(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach restype / argtypes to every entry point used in Python."""

    lib.scan_waists.restype  = None
    lib.scan_waists.argtypes = [
        ctypes.c_int, ctypes.c_int,                                    # N_WP, N_WS
        _dp, _dp,                                                      # wp_arr, ws_arr
        ctypes.c_int,                                                  # N_SPEC
        _dp, _dp, _dp, _dp,                                            # k_s, k_i, dk, w_spec
        _dp,                                                           # alpha_sq
        _dp,                                                           # k_p_arr
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # k0_p, L, offset_w0
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # lam0_p, lam0_s, lam0_i
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # n0_p, n0_s, n0_i
        ctypes.c_double, ctypes.c_double,                              # k0_s, k0_i
        ctypes.c_int,                                                  # P_MAX_MIN
        ctypes.c_int, _dp, _dp,                                        # N_Z, z_nodes, z_weights
        _dp,                                                           # chi_z
        ctypes.c_int, _dp, _dp,                                        # N_R, r_ref, r_weights
        ctypes.c_int,                                                  # method
        _dp_w, _dp_w,                                                  # S2_map, H_map
    ]

    lib.compute_single.restype  = None
    lib.compute_single.argtypes = [
        ctypes.c_int,                                                  # N_SPEC
        _dp, _dp, _dp, _dp, _dp,                                       # k_s, k_i, dk, w_spec, alpha_sq
        _dp,                                                           # k_p_arr
        ctypes.c_int,                                                  # P_MAX
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # wp, ws, wi
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # k0_p, L, offset_w0
        ctypes.c_double, ctypes.c_double,                              # k0_s, k0_i
        ctypes.c_int, _dp, _dp, _dp,                                   # N_Z, z_nodes, z_weights, chi_z
        ctypes.c_int, _dp, _dp,                                        # N_R, r_ref, r_weights
        ctypes.c_int,                                                  # method
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    lib.compute_c00_heatmap.restype  = None
    lib.compute_c00_heatmap.argtypes = [
        ctypes.c_int,                                                  # N_total
        _dp, _dp, _dp, _dp,                                            # k_s, k_i, k_p, dk
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # wp, ws, wi
        ctypes.c_double, ctypes.c_double,                              # L, offset_w0
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # k0_p, k0_s, k0_i
        ctypes.c_int, _dp, _dp, _dp,                                   # N_Z, z_nodes, z_weights, chi_z
        _dp_w,                                                         # c00sq_out
    ]

    lib.compute_modes_spectrum.restype  = None
    lib.compute_modes_spectrum.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,                      # N_lam, N_i, P_MAX
        _dp, _dp, _dp, _dp,                                            # k_s, k_i, k_p, dk
        _dp,                                                           # weights
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # wp, ws, wi
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # k0_p, L, offset_w0
        ctypes.c_double, ctypes.c_double,                              # k0_s, k0_i
        ctypes.c_int, _dp, _dp, _dp,                                   # N_Z, z_nodes, z_weights, chi_z
        ctypes.c_int, _dp, _dp,                                        # N_R, r_ref, r_weights
        _dp_w, _dp_w,                                                  # P_0i_out, P_i0_out
    ]

    lib.compute_c00_integrated.restype  = None
    lib.compute_c00_integrated.argtypes = [
        ctypes.c_int, ctypes.c_int,                                    # N_outer, N_i
        _dp, _dp, _dp, _dp, _dp,                                       # k_s, k_i, k_p, dk, weights
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # wp, ws, wi
        ctypes.c_double, ctypes.c_double,                              # L, offset_w0
        ctypes.c_double, ctypes.c_double, ctypes.c_double,             # k0_p, k0_s, k0_i
        ctypes.c_int, _dp, _dp, _dp,                                   # N_Z, z_nodes, z_weights, chi_z
        _dp_w,                                                         # spectrum_out
    ]

    return lib


# ──────────────────────────────────────────────────────────────────────
#  Public accessor (singleton)
# ──────────────────────────────────────────────────────────────────────
_lib_handle: ctypes.CDLL | None = None


def get_lib(rebuild: bool = False) -> ctypes.CDLL:
    """Return the compiled, ctypes-registered libspdc handle.

    The library is built lazily on first call (or whenever the C source
    is newer than the binary).  Pass ``rebuild=True`` to force a rebuild.
    """
    global _lib_handle
    if _lib_handle is None or rebuild:
        build(force=rebuild)
        _lib_handle = _register(ctypes.CDLL(_LIB))
    return _lib_handle
