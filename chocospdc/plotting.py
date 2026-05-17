"""
chocospdc.plotting — Common figure recipes.

All figures use the cream facecolor and PALETTE defined in
:mod:`chocospdc.style`.  Each function takes a result dict (from
:mod:`chocospdc.compute`) and returns the Figure for further tweaking
or saving.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from . import style as st
from . import config as cfg
from .sellmeier import CRYSTAL_NAMES, TYPE_NAMES


# ──────────────────────────────────────────────────────────────────────
#  Shared utilities
# ──────────────────────────────────────────────────────────────────────
def _title(suffix=""):
    base = (f"{CRYSTAL_NAMES[cfg.CRYSTAL]}, {TYPE_NAMES[cfg.SPDC_TYPE]}, "
            f"L = {cfg.L/1e3:.0f} mm")
    return f"{base}\n{suffix}" if suffix else base


def _filter_tag(params, has_filter, prefix=""):
    """Build a short human-readable description of the symmetric filter,
    suitable for appending to a title.  Returns "" when no filter is active.
    """
    if not has_filter:
        return ""
    shape = params.get("filter_shape", "none")
    if shape == "file":
        import os
        path = params.get("filter_file") or "?"
        name = os.path.basename(str(path))
        return f"{prefix}filter (both arms): file = {name}"
    bw   = params.get("filter_bw_nm", 0.0)
    ctr  = params.get("filter_center_nm")
    kind = "FWHM" if shape == "gauss" else "BW"
    s = f"{prefix}filter (both arms): {shape}, {kind} = {bw:.3g} nm"
    if ctr is not None:
        s += f", centre = {ctr:.3f} nm"
    return s


# ══════════════════════════════════════════════════════════════════════
#  WAIST SCAN
# ══════════════════════════════════════════════════════════════════════
def waist_scan(result, best=None, save=None):
    """Two-panel η and brightness heatmaps with white contour lines.

    `best` is an optional (wp, ws, B) triple to mark with a star.
    """
    wp, ws = result["wp"], result["ws"]
    fwhm   = result["params"]["fwhm_nm"]

    # Filter tag when active
    suffix = f"pump FWHM = {fwhm:.2f} nm"
    ftag   = _filter_tag(result["params"], result.get("has_filter"), prefix="   ·   ")
    if ftag:
        suffix += ftag

    fig, (axH, axB) = st.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(_title(suffix), fontsize=13, y=0.98)
    Xc, Yc = np.meshgrid(wp, ws)

    # Heralding
    imH = axH.pcolormesh(Xc, Yc, result["H"], shading="gouraud",
                         cmap=st.CMAP_SEQ, vmin=0, vmax=1)
    csH = axH.contour(Xc, Yc, result["H"],
                      levels=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
                      colors=st.CONTOUR_LINE, linewidths=1.0)
    axH.clabel(csH, fmt="%.2f", fontsize=8)
    axH.set_xlabel(r"$w_p$ (µm)")
    axH.set_ylabel(r"$w_s$ (µm)")
    axH.set_title(r"Symmetric heralding $\eta = \sqrt{\eta_s\,\eta_i}$")
    fig.colorbar(imH, ax=axH, pad=0.02)

    # Brightness
    imB = axB.pcolormesh(Xc, Yc, result["B"], shading="gouraud",
                         cmap=st.CMAP_SEQ, vmin=0, vmax=1)
    csB = axB.contour(Xc, Yc, result["B"],
                      levels=[0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
                      colors=st.CONTOUR_LINE, linewidths=1.0)
    axB.clabel(csB, fmt="%.2f", fontsize=8)
    axB.set_xlabel(r"$w_p$ (µm)")
    axB.set_ylabel(r"$w_s$ (µm)")
    axB.set_title(r"Relative brightness $\mathcal{B}$")
    fig.colorbar(imB, ax=axB, pad=0.02)

    if best is not None:
        wp_b, ws_b, _ = best
        for ax in (axH, axB):
            ax.plot(wp_b, ws_b, "*", color="cyan", markersize=14,
                    markeredgecolor="white", markeredgewidth=1.0)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


def waist_max_eta(wp_fine, eta_max, ws_opt, B_at_opt, *, fwhm_nm, save=None):
    """Companion plot: max η vs w_p with optimal w_s and B overlaid."""
    fig, ax_eta = st.subplots(figsize=(7.5, 4.5))
    ax_B  = ax_eta.twinx()
    ax_ws = ax_eta.twinx()
    ax_ws.spines["right"].set_position(("axes", 1.18))

    c_eta, c_B, c_ws = st.ACCENT_BLUE, st.ACCENT_RED, st.ACCENT_TEAL

    l1, = ax_eta.plot(wp_fine, eta_max,  color=c_eta, lw=2.2,
                      label=r"max$_{w_s}\;\eta$")
    l2, = ax_B  .plot(wp_fine, B_at_opt, color=c_B,   lw=2.2,
                      label=r"$\mathcal{B}$ at opt. $w_s$")
    l3, = ax_ws .plot(wp_fine, ws_opt,   color=c_ws,  lw=1.6, ls="--",
                      alpha=0.85, label=r"opt. $w_s$")

    ax_eta.set_xlabel(r"$w_p$ (µm)")
    ax_eta.set_ylabel(r"Max heralding $\eta$",        color=c_eta)
    ax_B  .set_ylabel(r"Relative brightness $\mathcal{B}$", color=c_B)
    ax_ws .set_ylabel(r"Optimal $w_s$ (µm)",          color=c_ws)
    for axx, c in [(ax_eta, c_eta), (ax_B, c_B), (ax_ws, c_ws)]:
        axx.tick_params(axis="y", labelcolor=c)

    ax_eta.legend([l1, l2, l3], [h.get_label() for h in (l1, l2, l3)],
                  loc="lower right", fontsize=10)
    fig.suptitle(_title(f"pump FWHM = {fwhm_nm:.2f} nm"), fontsize=12, y=0.99)
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  MODE-RESOLVED SPECTRA
# ══════════════════════════════════════════════════════════════════════
def modes_spectrum(result, save=None):
    P_0i  = result["P_0i"]
    P_i0  = result["P_i0"]
    dlam  = result["delta_lam"]
    p_max = result["p_max"]
    wp    = result["params"]["wp"]
    ws    = result["params"]["ws"]
    fwhm  = result["params"]["fwhm_nm"]
    eta   = result["eta"]
    ftag  = _filter_tag(result["params"], result.get("has_filter"), prefix="\n")

    fig, ax = st.subplots(figsize=(9, 5))

    norm = P_0i[0].max()
    x    = dlam * 1e3                              # nm

    ax.plot(x, P_0i[0] / norm, color=st.INK, lw=2.0, zorder=10,
            label=r"$|C_{00}|^2$")
    for p in range(1, p_max + 1):
        c = st.PALETTE[(p - 1) % len(st.PALETTE)]
        ax.plot(x, P_0i[p] / norm, color=c, lw=1.3, alpha=0.5,
                label=fr"$|C_{{0{p}}}|^2,\,|C_{{{p}0}}|^2$")
        ax.plot(x, P_i0[p] / norm, color=c, lw=1.3, alpha=0.5,
                linestyle="--")

    ax.set_xlabel(r"$\Delta\lambda$ (nm)")
    ax.set_ylabel("Spectral density (normalised)")
    ax.set_title(_title(
        f"$w_p$ = {wp:.0f} µm,  $w_s$ = {ws:.0f} µm,  "
        f"FWHM = {fwhm:.2f} nm  →  η = {eta:.4f}" + ftag
    ))
    ax.legend(ncol=2, title=r"solid $(0,p)$  ·  dashed $(p,0)$",
              title_fontsize=11)
    st.despine(ax)
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  JSA
# ══════════════════════════════════════════════════════════════════════
def jsa(result, save=None):
    lam_s = result["lam_s"] * 1e3
    lam_i = result["lam_i"] * 1e3
    JSI   = np.abs(result["C00"])**2
    JSI  /= JSI.max()

    wp = result["params"]["wp"]
    ws = result["params"]["ws"]
    wi = result["params"]["wi"]
    K  = result["K"]
    T  = result["params"]["T"]
    fwhm = result["params"]["fwhm_nm"]

    # Filter tag if active
    tag = _filter_tag(result["params"], result.get("has_filter"), prefix="\n")

    Ls, Li = np.meshgrid(lam_s, lam_i, indexing="ij")

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    fig.set_facecolor(st.FACE)
    pcm = ax.pcolormesh(Ls, Li, np.sqrt(JSI), shading="gouraud", cmap=st.CMAP_SEQ)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$|C_{0,0}|$ (normalised)")

    ax.set_xlabel(r"$\lambda_s$ (nm)")
    ax.set_ylabel(r"$\lambda_i$ (nm)")
    ax.set_aspect("equal")
    fig.suptitle(
        f"{CRYSTAL_NAMES[cfg.CRYSTAL]}, {TYPE_NAMES[cfg.SPDC_TYPE]}, "
        f"L = {cfg.L/1e3:.0f} mm,  T = {T}°C\n"
        f"$w_p$ = {wp:.0f}, $w_s$ = {ws:.0f}, $w_i$ = {wi:.0f} µm   "
        f"FWHM = {fwhm:.2f} nm   K = {K:.2f}"
        + tag,
        fontsize=11)
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


def schmidt(result, save=None, n_show=15):
    lam_k = result["schmidt_values"]
    K     = result["K"]
    n_show = min(n_show, len(lam_k))

    fig, ax = st.subplots(figsize=(5.5, 3.7))
    ax.bar(range(n_show), lam_k[:n_show],
           color=st.ACCENT_BLUE, edgecolor=st.INK, linewidth=0.5)
    ax.set_xlabel(r"Schmidt mode index $k$")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title(f"Schmidt decomposition — K = {K:.2f}")
    st.despine(ax)
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  SPECTRUM vs T  (waterfall and heatmap)
# ══════════════════════════════════════════════════════════════════════
def spectrum_waterfall(result, save=None, spacing=0.8):
    T_arr  = result["T"]
    lam_nm = result["lam_nm"]
    spec   = result["spectra"]
    wp     = result["params"]["wp"]
    ws     = result["params"]["ws"]
    fwhm   = result["params"]["fwhm_nm"]
    N_T    = len(T_arr)

    fig, ax = st.subplots(figsize=(6.5, 0.42 * N_T + 2.2))

    gmax = spec.max()
    for iT in range(N_T):
        y = spec[iT] / gmax
        ax.fill_between(lam_nm, iT * spacing, iT * spacing + y,
                        alpha=0.35, color=st.ACCENT_CORAL)
        ax.plot(lam_nm, iT * spacing + y, color=st.ACCENT_RED, lw=0.6)
        ax.text(lam_nm[0] - (lam_nm[-1] - lam_nm[0]) * 0.01,
                iT * spacing + 0.05,
                f"{T_arr[iT]:.1f}°C",
                fontsize=10, va="bottom", ha="right")

    ax.set_xlabel("Signal wavelength (nm)")
    ax.set_xlim(lam_nm[0], lam_nm[-1])
    ax.set_ylim(-0.1, N_T * spacing + 0.6)
    ax.set_yticks([])
    ax.set_title(_title(
        f"$w_p$={wp:.0f}, $w_s$={ws:.0f} µm  "
        f"·  QPM @ {cfg.T_QPM}°C  ·  FWHM = {fwhm:.2f} nm"
    ))
    st.despine(ax, sides=("top", "right", "left"))
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


def spectrum_heatmap(result, save=None, log=False):
    T_arr  = result["T"]
    lam_nm = result["lam_nm"]
    spec   = result["spectra"]
    fwhm   = result["params"]["fwhm_nm"]
    wp     = result["params"]["wp"]
    ws     = result["params"]["ws"]

    Z = spec / spec.max()
    if log:
        z_min = Z[Z > 0].min() * 0.1
        norm  = mcolors.LogNorm(vmin=z_min, vmax=1.0)
        Z = np.where(Z > 0, Z, z_min)
    else:
        norm = None

    fig, ax = st.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(lam_nm, T_arr, Z,
                        cmap=st.CMAP_SEQ, shading="auto", norm=norm)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\int |C_{00}|^2 |\alpha|^2 d\lambda_i$  (normalised)")
    ax.set_xlabel("Signal wavelength (nm)")
    ax.set_ylabel("Crystal temperature (°C)")
    ax.set_title(_title(
        f"$w_p$={wp:.0f}, $w_s$={ws:.0f} µm  "
        f"·  QPM @ {cfg.T_QPM}°C  ·  FWHM = {fwhm:.2f} nm"
    ))
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  BRIGHTNESS vs T
# ══════════════════════════════════════════════════════════════════════
def brightness_vs_T(result, save=None, normalize=True, log=False,
                    t_ref=None):
    T_arr      = result["T"]
    lam_nm     = result["lam_nm"]
    spectra    = result["spectra"]
    brightness = result["brightness"]
    filters    = result["filters"]
    bws_nm     = np.atleast_1d(result["filter_bw_nm"])
    f_center   = result["filter_center_nm"]
    f_shape    = result["filter_shape"]
    fwhm       = result["params"]["fwhm_nm"]
    wp         = result["params"]["wp"]
    ws         = result["params"]["ws"]

    N_F = brightness.shape[0]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(N_F - 1, 1)) for i in range(N_F)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7.2),
        gridspec_kw=dict(height_ratios=[1.4, 1.0]),
        constrained_layout=True)
    fig.set_facecolor(st.FACE)

    # ── Top panel ────────────────────────────────────────────────
    B = brightness.copy()
    if normalize:
        B = B / np.maximum(B.max(axis=1, keepdims=True), 1e-300)
        y_label = "Normalized Brightness"
    else:
        y_label = "Brightness (arb. units)"
    if log:
        ax_top.set_yscale("log")

    for iF, bw in enumerate(bws_nm):
        if f_shape == "none":
            label = "no filter (both arms)"
        elif f_shape == "file":
            import os
            path = result.get("filter_file")
            name = os.path.basename(str(path)) if path else "(file)"
            label = f"file: {name}  (both arms)"
        else:
            kind = "FWHM" if f_shape == "gauss" else "BW"
            label = f"{f_shape}  {kind} = {bw:.3g} nm  (both arms)"
        ax_top.plot(T_arr, B[iF], color=colors[iF], lw=1.8, label=label)

    ax_top.axvline(cfg.T_QPM, color=st.MUTED, ls=":", lw=1.0, alpha=0.8,
                   label=f"T_QPM = {cfg.T_QPM}°C")
    ax_top.set_xlabel("Crystal temperature (°C)")
    ax_top.set_ylabel(y_label)
    ax_top.set_xlim(T_arr[0], T_arr[-1])
    ax_top.grid(True)
    ax_top.legend(loc="best")
    ax_top.set_title(_title(
        f"$w_p$={wp:.0f}, $w_s$={ws:.0f} µm  ·  pump FWHM = {fwhm:.2f} nm"
    ))

    # ── Bottom panel: reference spectrum + filter overlays ───────
    iT_peak = int(np.argmax(brightness[0]))
    if t_ref is None:
        iT_ref = iT_peak
    else:
        iT_ref = int(np.argmin(np.abs(T_arr - t_ref)))
    T_ref = T_arr[iT_ref]

    spec_ref = spectra[iT_ref]
    spec_norm = spec_ref / spec_ref.max()
    ax_bot.plot(lam_nm, spec_norm, color=st.ACCENT_BLUE, lw=1.6,
                label=f"signal spectrum @ T = {T_ref:.2f}°C")
    ax_bot.axvline(f_center, color=st.INK, ls="--", lw=0.8, alpha=0.6)

    for iF, bw in enumerate(bws_nm):
        if f_shape == "none":
            continue
        ax_bot.fill_between(lam_nm, 0.0, filters[iF],
                            color=colors[iF], alpha=0.18)
        ax_bot.plot(lam_nm, filters[iF], color=colors[iF], lw=1.0, alpha=0.9)

    ax_bot.set_xlabel("Signal wavelength (nm)")
    ax_bot.set_ylabel("Spectrum (norm.)  /  filter T")
    ax_bot.set_xlim(lam_nm[0], lam_nm[-1])
    ax_bot.set_ylim(0.0, 1.08)
    ax_bot.grid(True)
    ax_bot.legend(loc="upper right")

    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  HOM dip
# ══════════════════════════════════════════════════════════════════════
def hom_best_dip(result, fit=None, save=None):
    """Plot the deepest HOM dip; if `fit` (popt, perr) is supplied, overlay it."""
    T_arr  = result["T"]
    V0     = result["V0"]
    dip    = result["dip"]
    tau_fs = result["tau_ps"] * 1e3

    iT_best = int(np.argmax(V0))
    T_best  = float(T_arr[iT_best])
    y       = dip[iT_best] / 2.0

    # Filter tag if active
    tag = _filter_tag(result["params"], result.get("has_filter"), prefix="\n")

    fig, ax = st.subplots(figsize=(7, 5))

    if fit is not None:
        from scipy.optimize import curve_fit       # noqa: F401  (caller did the fit)
        popt, perr = fit
        c, a, x0, sig = popt
        sig = abs(sig)
        FWHM     = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig
        FWHM_err = 2.0 * np.sqrt(2.0 * np.log(2.0)) * perr[3]
        vis      = a / c
        vis_err  = vis * np.sqrt((perr[1]/a)**2 + (perr[0]/c)**2)

        x_line = np.linspace(tau_fs.min(), tau_fs.max(), 1000)
        y_line = c - a * np.exp(-(x_line - x0)**2 / (2.0 * sig**2))

        ax.plot(x_line, y_line, color=st.ACCENT_PURP, lw=1.6, zorder=2,
                label=fr"fit: V = {vis:.3f} $\pm$ {vis_err:.1g}")
        half = c - a / 2.0
        ax.hlines(half, x0 - FWHM/2.0, x0 + FWHM/2.0,
                  color=st.INK, lw=1)
        ax.annotate(fr"FWHM = {FWHM:.3g} $\pm$ {FWHM_err:.1g} fs",
                    xy=(x0 + FWHM/2.0 + 100, half+.035), xytext=(8, -14), textcoords="offset points",
                    ha="left", va="top", color=st.ACCENT_PURP)
        ax.axhline(c, color=st.MUTED, lw=0.6, ls=":")

    ax.plot(tau_fs, y, ".", color=st.ACCENT_BLUE, alpha=0.25,
            zorder=1, label="SPDChoco simulation")

    ax.set_xlabel("Time delay τ (fs)")
    ax.set_ylabel("Coincidence rate (norm.)")
    ax.set_title(_title(f"Best dip @ T = {T_best:.2f}°C  (T_QPM = {cfg.T_QPM}°C)") + tag)
    ax.grid(True, axis="y")
    ax.legend(loc="lower left")
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig


def hom_visibility_vs_T(result, save=None):
    tag = _filter_tag(result["params"], result.get("has_filter"), prefix="   ").lstrip()

    fig, ax = st.subplots(figsize=(7, 4.5))
    ax.plot(result["T"], result["V0"] * 100.0,
            color=st.ACCENT_PURP, lw=2.0, label="SPDChoco simulation")
    ax.set_xlabel("Crystal temperature (°C)")
    ax.set_ylabel("Visibility V(0) (%)")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title(_title(tag.lstrip()))
    fig.tight_layout()
    st.sign(fig)
    if save:
        fig.savefig(save)
    return fig
