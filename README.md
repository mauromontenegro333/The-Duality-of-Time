# Transition-clock cosmology benchmark pipeline

This repository is the non-Boltzmann numerical stage for the manuscript **The Duality of Time**.
It implements the transition-clock background, DESI DR2 BAO benchmark comparison, a broad CMB acoustic-scale diagnostic, optional SN q0 summary checks, GR-like growth on the modified background, redshift-drift forecasts, and a PDF report.

Boltzmann/CLASS/CAMB code is intentionally not included yet.

## What is included

- `src/tcc/background.py`: transition-clock and LCDM background functions.
- `src/tcc/distances.py`: `D_M`, `D_H`, `D_V`, BAO ratios, CMB acoustic-scale proxy, redshift drift.
- `src/tcc/growth.py`: dependency-light RK4 solver for GR-like linear growth on the benchmark background.
- `src/tcc/likelihood.py`: compact BAO+CMB-proxy+q0-summary likelihood.
- `data/`: compact benchmark data transcribed from the manuscript.
- `scripts/finalize_benchmark_outputs.py`: regenerates the frozen benchmark tables and plots used in the report.
- `scripts/make_report.py`: builds `results/transition_clock_benchmark_report.pdf`.
- `scripts/run_benchmark.py`: experimental low-evaluation search/sampling driver.

## Reproduce the attached report

Use Python with site imports disabled and the repository plus site-packages on `PYTHONPATH`. The BLAS thread limits avoid hangs in minimal container environments.

```bash
cd transition_clock_cosmology
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=$PWD:$PWD/src:/opt/pyvenv/lib64/python3.13/site-packages \
python -S scripts/finalize_benchmark_outputs.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=$PWD:$PWD/src:/opt/pyvenv/lib64/python3.13/site-packages \
python -S scripts/make_report.py
```

The PDF will be written to:

```text
results/transition_clock_benchmark_report.pdf
```

## Scientific status

This is a benchmark, not a final public likelihood paper. The current result is deliberately conservative:

- The transition-clock branch fits the benchmark DESI BAO table reasonably at the compact-distance level.
- It misses the broad CMB acoustic-scale proxy in the current simple logistic transition.
- LCDM fits the BAO+CMB-proxy benchmark better, as expected.
- The age-corrected SN q0 summary strongly penalizes LCDM, but the transition-clock branch is still not a full successful fit.

The correct manuscript use is as a reproducible diagnostic and target-setting section, not as an observational validation claim.

## Data limitations

- DESI DR2 BAO is represented by the manuscript table values and within-bin DM-DH correlations.
- Cross-bin DESI covariance matrices are not included.
- The CMB term is `ell_A = pi D_M(z*)/r_d` with a broad uncertainty, not a Planck likelihood.
- SN information is included only through q0 summaries.

For publication, replace these with official likelihoods/covariances and then do the Boltzmann implementation.
