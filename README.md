# Transition-Clock / TIP CAMB Boltzmann Run

This package contains the first reproducible CAMB run for the Duality-of-Time / Temporal Inertia Principle transition-clock branch.

## What this does

It runs stock CAMB and outputs:

- `cmb_cls_*.csv`: \(C_\ell^{TT}\), \(C_\ell^{TE}\), \(C_\ell^{EE}\), \(C_L^{\phi\phi}\)
- `matter_power_*.csv`: \(P(k,z)\)
- `fsigma8_*.csv`: \(f\sigma_8(z)\)
- `bao_distances_*.csv`: \(D_M/r_d\), \(D_H/r_d\), \(D_V/r_d\)
- `background_*.csv`: \(H(z)\), \(E(z)\), \(q(z)\)
- comparison figures in `figures/`

## Important scientific status

This is a **CAMB/PPF effective-background implementation**, not a native modification of the CLASS/CAMB perturbation hierarchy.

The clock-dressed background \(E_{\rm TIP}(a)\) is mapped into an effective tabulated \(w(a)\) and run through `camb.dark_energy.DarkEnergyPPF`. This produces real Boltzmann spectra, but it should be described in the paper as an implementation benchmark / smoke test, not as the final native clock-field Boltzmann likelihood.

The native 10/10 version should later modify CLASS, hi_class, EFTCAMB, or CAMB so that the clock constraint and perturbation variables are evolved directly.

## Recommended repository location

Put this directory inside the existing repository:

```text
The-Duality-of-Time/
  boltzmann/
    README.md
    requirements.txt
    configs/
    tip_boltzmann/
    results/
    figures/
    paper/
```

Do **not** create a separate repository unless the Boltzmann code grows into a large independent package. The current repository README says the repo is script-first and intended to produce exact PDF-report artifacts, so this belongs in the same repository under `boltzmann/`.

## Reproduce

```bash
cd boltzmann
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python tip_boltzmann/run_tip_camb.py \
  --config configs/tip_report_bestfit.json \
  --config configs/lcdm_report_baseline.json \
  --outdir results

python tip_boltzmann/plot_results.py \
  --results results \
  --figures figures
```

## Models included

### `tip_report_bestfit`

Uses the first-pass benchmark parameters:

```text
n0 = 0.593
zt = 1.842
beta = 11.604
Omega_m = 0.084
H0 = 65.545 km/s/Mpc
```

### `lcdm_report_baseline`

Uses the benchmark LCDM baseline:

```text
Omega_m = 0.275
H0 = 67.218 km/s/Mpc
```

Both use the same primordial defaults unless edited in the JSON configs:

```text
ombh2 = 0.02237
As = 2.10e-9
ns = 0.965
tau = 0.0544
```

## File interpretation

CAMB reports CMB spectra as raw \(C_\ell\) because `raw_cl=True` is used. Matter power uses `k_h_Mpc` and units \(({\rm Mpc}/h)^3\). Lensing potential is raw \(C_L^{\phi\phi}\).

## Current conclusion from this run

The run completes and produces the requested observables. The TIP effective-PPF mapping strongly changes early-time physics and yields a much smaller CAMB drag sound horizon than the benchmark proxy value. That is a warning sign: a paper should present this as the first Boltzmann implementation benchmark, not as a final validation.
