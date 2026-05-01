#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/src:/opt/pyvenv/lib64/python3.13/site-packages"
python -S scripts/finalize_benchmark_outputs.py
python -S scripts/make_report.py
