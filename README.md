<p align="center">
    <img width="300" height="200" alt="choco (1)" src="https://github.com/user-attachments/assets/46395fc5-8e51-4952-a1ee-22cd29252f38" />
</p>


Fast simulation of heralding efficiency, brightness, spectral purity,
HOM dips and joint spectral amplitudes for a focused SPDC photon-pair
source in a periodically-poled crystal.

All heavy computation lives in an auto-compiled C kernel with OpenMP.
The broadband Gaussian pump envelope α(λ_p), λ_i integration, and
symmetric detection filters are handled consistently throughout.

```
SPDChoco/
├── chocospdc/                  # the library
│   ├── config.py               # ← edit this for your source
│   ├── sellmeier.py            # KTP, KTP-F, BBO, PPLN, MgO:PPLN
│   ├── beams.py                # Gaussian beam helpers
│   ├── grids.py                # spectral grid builders
│   ├── quadrature.py           # Gauss–Legendre setup
│   ├── kernel.py               # auto-builds + loads libspdc
│   ├── compute.py              # high-level simulation API
│   ├── plotting.py             # figure recipes
│   └── style.py                # shared visual style (#ede2d8 cream)
├── examples/                   # runnable scripts; edit constants at top
│   ├── scan_waists.py          # 2-D waist scan
│   ├── modes.py                # mode-resolved spectra
│   ├── jsa.py                  # Joint Spectral Amplitude + Schmidt
│   ├── spectrum_vs_T.py        # waterfall + heatmap vs T
│   ├── brightness_vs_T.py      # coincidence brightness vs T
│   ├── hom_vs_T.py             # HOM dip vs T
│   ├── test_kernel.py          # regression test: method 0 vs method 1
│   └── _path.py                # tiny sys.path shim (don't edit)
├── spdc_kernel.c               # C source (OpenMP)
└── README.md
```

## Quick start

```bash
# 1. Edit your source parameters
nano chocospdc/config.py

# 2. Verify the configuration (also warns if not phase-matched)
python -m chocospdc.config

# 3. Pick a script in examples/ — parameters are constants at the top
python examples/scan_waists.py
python examples/modes.py
python examples/jsa.py
python examples/spectrum_vs_T.py
python examples/brightness_vs_T.py
python examples/hom_vs_T.py
```

Scripts work from any working directory thanks to a small `_path.py`
shim inside `examples/`.  Outputs (`plots/`, `data/`, `scan_data/`)
are created relative to the CWD, so you can keep multiple runs apart
by `cd`-ing into per-scan directories before launching a script.

The C kernel auto-compiles on first run (requires `gcc` + `libgomp`).

## Installation

```bash
# Linux
sudo apt install gcc libgomp1
# Windows (Anaconda)
conda install -c conda-forge m2w64-toolchain
# Python dependencies
pip install numpy scipy matplotlib
```

## Configuration

All physical parameters of the source live in `chocospdc/config.py`:

| Parameter   | Description                              | Example         |
|-------------|------------------------------------------|-----------------|
| `CRYSTAL`   | Crystal type                             | `CRYSTAL_KTP_F` |
| `SPDC_TYPE` | Phase-matching type                      | `TYPE_II`       |
| `T`         | Working crystal temperature [°C]         | `30`            |
| `T_QPM`     | QPM design temperature [°C]              | `20`            |
| `L`         | Crystal length [µm]                      | `30_000`        |
| `lambda0_p` | Pump wavelength [µm]                     | `0.775`         |
| `lambda0_s` | Signal wavelength [µm]                   | `1.55`          |
| `w0_p`      | Pump beam waist [µm]                     | `18`            |
| `w0_s`      | Signal collection waist [µm]             | `21`            |
| `FWHM_PUMP` | Pump spectral FWHM [µm] (intensity, > 0) | `1e-5`          |

For a narrow-linewidth CW laser, use a small `FWHM_PUMP` like `1e-6`
(1 pm) — never zero.  Set `T_QPM = T` for perfect phase matching at
degeneracy.

## Symmetric filter API

Every SPDC observable accepts the same four parameters describing a
**bandpass filter applied identically to both arms** (signal and
idler):

| Parameter            | Description                                       |
|----------------------|---------------------------------------------------|
| `filter_shape`       | `"rect"` \| `"gauss"` \| `"file"` \| `"none"`     |
| `filter_center_nm`   | Centre wavelength [nm]; `None` → λ_s₀             |
| `filter_bw_nm`       | Rect full width / Gaussian FWHM [nm]              |
| `filter_file`        | Path to text file for `filter_shape="file"`       |

Supported across all six functions: `waist_scan`, `pair_singles`,
`modes_spectrum`, `jsa`, `hom_vs_T`, `brightness_vs_T`.

### Why filtering needs to be aware of detection topology

Photon-pair observables differ in *which photons are detected*, and
the filter only matters where the photon actually hits a detector:

| Quantity | What's detected | Filter weight |
|---|---|---|
| **S₂** (coincidences) | both detectors fire | `f(λ_s) · f(λ_i)` |
| **S₁_s** (signal singles) | signal-arm detector only | `f(λ_s)` only |
| **S₁_i** (idler singles) | idler-arm detector only | `f(λ_i)` only |

Each accumulator inside the C kernel receives the correct weight, so
heralding η = √(η_s · η_i) reflects what an experimentalist sees with
real bandpass filters in front of each detector.

### Default behaviour

`filter_shape="none"` (the default everywhere) leaves the kernel in
its unfiltered mode.  The spectral integration window then acts as
the implicit rect filter — the same trick the legacy code used via
hard-coded windows.  Existing scripts give bit-identical results.

For physical filter modelling, set a real `filter_shape` and make
sure the integration window is comfortably wider than the filter
bandwidth.

### Custom filter from a text file

Pass `filter_shape="file"` and `filter_file="path/to/filter.txt"`.
The file must be two columns: **wavelength in nm**, **transmission in
percent**.  Whitespace- or comma-separated, `#` starts a comment.
The file is sorted, transmissions clipped to [0, 100] %, and linearly
interpolated to the spectral grid.  Wavelengths outside the file's
range get zero transmission.

```text
# my_filter.txt — measured transmission of a 1 nm bandpass
# lambda_nm   T_percent
1545.0        0.5
1546.0        2.1
1547.0       18.4
1548.0       72.9
1549.0       98.7
1550.0       99.4
1551.0       98.3
1552.0       70.2
1553.0       16.9
1554.0        1.8
1555.0        0.4
```

### Examples

```python
# 2-D waist scan through 1-nm bandpass filters on both arms
result = compute.waist_scan(n_w=30, filter_shape="rect", filter_bw_nm=1.0)

# Mode-resolved spectrum with the actual filter you'll use in lab
result = compute.modes_spectrum(wp=18, ws=21,
                                filter_shape="file",
                                filter_file="my_filter.txt")

# JSA "as measured" through 0.5-nm rectangular filters
result = compute.jsa(wp=18, ws=21, half_range=0.005,
                     filter_shape="rect", filter_bw_nm=0.5)

# HOM visibility vs T through 0.3-nm Gaussian filters
result = compute.hom_vs_T(T_range=(30, 50), dt=0.5,
                          filter_shape="gauss", filter_bw_nm=0.3)
```

Every example script in `examples/` exposes a `FILTER_SHAPE /
FILTER_CENTER_NM / FILTER_BW_NM / FILTER_FILE` block at the top, so
turning on a filter is a one-line edit.

## Library API

```python
from chocospdc import compute, plotting, style, config

style.use()           # apply the project visual style once

# 2-D waist scan
result = compute.waist_scan(n_w=30, method=1)
plotting.waist_scan(result, save="plots/scan.png")

# Mode-resolved spectra
result = compute.modes_spectrum(wp=18, ws=21, p_max=10)
print(f"η = {result['eta']:.4f}")

# Joint spectral amplitude + Schmidt
result = compute.jsa(wp=18, ws=21, n_lam=200)
print(f"K = {result['K']:.2f}")

# C₀₀ heatmap vs (T, λ_s)
result = compute.c00_heatmap(
    T_range=(20, 80), dt=0.2,
    lam_range=(1.54, 1.56), n_lam=400,
    wp=18, ws=21)

# Brightness vs T with one or several filters
result = compute.brightness_vs_T(
    T_range=(30, 60), dt=0.05,
    filter_center_nm=1550, filter_bw_nm=[1.0, 4.0, 10.0],
    filter_shape="rect")

# HOM dip vs T
result = compute.hom_vs_T(T_range=(22, 50), dt=1.0)
```

Every `compute.*` function returns a dict with the raw arrays, the
parameters used, and the wall-clock time.  Every `plotting.*` function
takes a result dict (and optional `save="plots/foo.png"`) and returns
the Figure for further tweaking.

## C kernel entry points

The compiled kernel (`spdc_kernel.c`) exposes five functions, all
parallelised via OpenMP:

| Function                  | Used by                              | Description                                     |
|---------------------------|--------------------------------------|-------------------------------------------------|
| `scan_waists`             | `compute.waist_scan`                 | 2-D waist scan with per-arm filter weights      |
| `compute_single`          | `compute.pair_singles`               | Single waist pair (S₂, S₁_s, S₁_i)              |
| `compute_c00_heatmap`     | `compute.c00_heatmap`                | \|C₀₀\|² at a flat array of spectral pts        |
| `compute_modes_spectrum`  | `compute.modes_spectrum`             | Mode-resolved P_{0,p}, P_{p,0} (bare + f_i)     |
| `compute_c00_integrated`  | (available)                          | \|C₀₀\|² with λ_i sum folded in                 |

Two singles methods coexist:
- **Method 0** — LG mode sum (fast when P_MAX is small).
- **Method 1** — Double-z analytic trace (Hille–Hardy, exact in the LG
  sum, O(N_Z²)).  Default for `waist_scan` and `pair_singles`.

`examples/test_kernel.py` checks that the two methods agree to ~1e-5
on S₂, S₁_s and S₁_i, and reports timing.

## Visual style

Every figure uses the shared style in `chocospdc/style.py`:

- Figure facecolor: `#ede2d8` (warm cream)
- Categorical palette: indigo, raspberry, teal, ochre, violet, coral, sky
- Sequential colormap: `inferno` with white contour overlays
- Despined line plots, math-mode labels, single source-of-truth `rcParams`
- `@SPDChoco` watermark in italic dark brown at the bottom right

Edit `style.py` to retune the whole project at once (turn off the
signature with `style.SIGNATURE_ENABLED = False`).

## Adding a new computation

1. Add a high-level function to `chocospdc/compute.py` returning a dict.
2. Add a matching plotter to `chocospdc/plotting.py`.
3. Create a thin script in `examples/` with parameters as constants and
   a `main()` that just calls compute + plot.

The first two lines of the new script should be:

```python
import _path  # noqa: F401  — make chocospdc importable from anywhere
from chocospdc import compute, plotting, style, config
```



## Disclaimer
*The author is a physicist and does not claim expertise in computer science. After its initial development, this library was refactored with the assistance of AI to ensure consistency in notation and overall architecture.*
