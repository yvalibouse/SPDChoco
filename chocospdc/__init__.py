"""
chocospdc — Focused SPDC source simulator.

Submodules are imported lazily so that
    python -m chocospdc.config
prints the summary without triggering a circular-import warning.

Typical usage:

    >>> from chocospdc import compute, plotting, style, config
    >>> style.use()
    >>> config.summary()
    >>> result = compute.waist_scan(n_w=30, n_lam=30)
"""

__all__ = [
    "beams", "compute", "config", "grids", "kernel",
    "plotting", "quadrature", "sellmeier", "style",
]


def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

