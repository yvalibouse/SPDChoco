<img width="1536" height="1024" alt="choco (1)" src="https://github.com/user-attachments/assets/46395fc5-8e51-4952-a1ee-22cd29252f38" />


Fast simulation of heralding efficiency, brightness, spectral purity,
HOM dips and joint spectral amplitudes for a focused SPDC photon-pair
source in a periodically-poled crystal.

All heavy computation lives in an auto-compiled C kernel with OpenMP.
The broadband Gaussian pump envelope α(λ_p) and λ_i integration are
handled consistently in every script.

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
├── spdc_kernel.c               # C source (OpenMP)
├── scan_waists.py              # 2-D waist scan
├── modes.py                    # mode-resolved spectra
├── jsa.py                      # Joint Spectral Amplitude + Schmidt
├── spectrum_vs_T.py            # waterfall + heatmap vs T
├── brightness_vs_T.py          # brightness vs T with a tunable filter
├── hom_vs_T.py                 # HOM dip vs T
├── test_kernel.py              # regression test: method 0 vs method 1
└── README.md
```

## Quick start

```bash
# 1. Edit your source parameters
nano chocospdc/config.py

# 2. Verify the configuration (also warns if not phase-matched)
python -m chocospdc.config

# 3. Pick a script — parameters are constants at the top of each file
python scan_waists.py
python modes.py
python jsa.py
python spectrum_vs_T.py
python brightness_vs_T.py
python hom_vs_T.py
```

The C kernel auto-compiles on first run (requires `gcc` and `libgomp`).

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
| `scan_waists`             | `compute.waist_scan`                 | 2-D waist scan (η, B maps)                      |
| `compute_single`          | `compute.pair_singles`               | Single waist pair (S₂, S₁_s, S₁_i)              |
| `compute_c00_heatmap`     | `compute.c00_heatmap`                | \|C₀₀\|² at a flat array of spectral pts        |
| `compute_modes_spectrum`  | `compute.modes_spectrum`             | Mode-resolved P_{0,p_i}, P_{p_s,0} with λ_i sum |
| `compute_c00_integrated`  | (available)                          | \|C₀₀\|² with λ_i sum folded in                 |

Two singles methods coexist in the kernel:
- **Method 0** — LG mode sum (original, fast when P_MAX is small).
- **Method 1** — Double-z analytic trace (Hille–Hardy, exact in the LG
  sum).  Cost scales as N_Z² but is independent of P_MAX.  This is the
  default in `compute.waist_scan` and `compute.pair_singles`.

`test_kernel.py` checks that the two methods agree to ~1e-5 on
S₂, S₁_s and S₁_i, plus the symmetric η.

## Visual style

Every figure uses the shared style in `chocospdc/style.py`:

- Figure facecolor: `#ede2d8` (warm cream)
- Categorical palette: indigo, raspberry, teal, ochre, violet, coral, sky
- Sequential colormap: `inferno` with white contour overlays
- Despined line plots, math-mode labels, single source-of-truth `rcParams`

Edit `style.py` to retune the whole project at once.
