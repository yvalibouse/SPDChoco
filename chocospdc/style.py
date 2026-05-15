"""
chocospdc.style — central plotting style for the whole library.

A single source of truth so every figure looks like it belongs to the
same family.  Edit the constants below to retune the whole project.
"""

from contextlib import contextmanager
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Colours ──────────────────────────────────────────────────────────
FACE        = "#ede2d8"          # figure background (warm cream)
INK         = "#222222"          # near-black for primary text/lines
MUTED       = "#666666"          # secondary text / grid

# Categorical palette — keep first 7 colours stable; extend if needed
PALETTE = [
    "#4954af",   # indigo
    "#ce3c64",   # raspberry
    "#4d9388",   # teal
    "#e1b23c",   # ochre
    "#9b59b6",   # violet
    "#D25752",   # coral
    "#428ED9",   # sky blue
]

# Named accents used by specific figures
ACCENT_BLUE  = PALETTE[0]
ACCENT_RED   = PALETTE[1]
ACCENT_TEAL  = PALETTE[2]
ACCENT_GOLD  = PALETTE[3]
ACCENT_PURP  = PALETTE[4]
ACCENT_CORAL = PALETTE[5]

# Colormaps
CMAP_SEQ     = "inferno"         # default sequential map
CMAP_DIV     = "RdBu_r"          # default diverging map

# Contour overlay colour on dark colormaps
CONTOUR_LINE = "white"

# Signature in the bottom-right of every figure
SIGNATURE_TEXT    = "@SPDChoco"
SIGNATURE_COLOR   = "#5A3621"     # dark brown, reads well on the cream FACE
SIGNATURE_SIZE    = 8
SIGNATURE_ENABLED = True


# ── Apply project-wide rcParams ─────────────────────────────────────
def use():
    """Set matplotlib rcParams once; call at the top of any script."""
    mpl.rcParams.update({
        "figure.facecolor":    FACE,
        "savefig.facecolor":   FACE,
        "savefig.dpi":         160,
        "axes.facecolor":      "white",
        "axes.edgecolor":      INK,
        "axes.labelcolor":     INK,
        "axes.titlesize":      13,
        "axes.labelsize":      12,
        "axes.linewidth":      1.0,
        "axes.prop_cycle":     mpl.cycler(color=PALETTE),
        "xtick.color":         INK,
        "ytick.color":         INK,
        "xtick.labelsize":     11,
        "ytick.labelsize":     11,
        "legend.frameon":      False,
        "legend.fontsize":     10,
        "lines.linewidth":     1.8,
        "font.family":         "DejaVu Sans",
        "mathtext.fontset":    "cm",
        "grid.color":          MUTED,
        "grid.alpha":          0.25,
        "grid.linewidth":      0.6,
    })


def figure(*args, **kw):
    """plt.figure with the project facecolor pre-applied."""
    kw.setdefault("facecolor", FACE)
    return plt.figure(*args, **kw)


def subplots(*args, **kw):
    """plt.subplots wrapper that applies the project facecolor."""
    fig, ax = plt.subplots(*args, **kw)
    fig.set_facecolor(FACE)
    return fig, ax


def despine(ax, sides=("top", "right")):
    """Hide a list of spines (default: top + right)."""
    for s in sides:
        ax.spines[s].set_visible(False)


def sign(fig, text=None):
    """Stamp the project signature at the bottom right of a figure.

    Called automatically by every plotter in :mod:`chocospdc.plotting`.
    Disable globally by setting ``style.SIGNATURE_ENABLED = False``.
    """
    if not SIGNATURE_ENABLED:
        return
    fig.text(0.995, 0.005,
             SIGNATURE_TEXT if text is None else text,
             ha="right", va="bottom",
             fontsize=SIGNATURE_SIZE, style="italic",
             color=SIGNATURE_COLOR, alpha=0.9)


@contextmanager
def temp_style():
    """Context manager that applies the style and restores rcParams on exit."""
    with mpl.rc_context():
        use()
        yield
