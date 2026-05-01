# Where to put the Boltzmann implementation

## Decision

Use the existing repository:

```text
mauromontenegro333/The-Duality-of-Time
```

Add this package as:

```text
The-Duality-of-Time/boltzmann/
```

## Why not a new repository

This code is directly tied to the manuscript and its benchmark parameters. It should be versioned with the paper so a referee can reproduce the numerical claims from the same commit.

Create a separate repository only later if the solver becomes a general public code independent of the Duality-of-Time paper.

## Paper placement

Add the paper snippet from:

```text
paper/boltzmann_implementation_section.tex
```

immediately after the current section:

```text
Implementation checklist for the later likelihood paper
```

and before:

```text
Discussion and conclusion
```

The full CSV outputs and scripts should stay in GitHub/Zenodo as supplementary material, not inside the paper body.

## Suggested commit layout

```text
boltzmann/
  README.md
  requirements.txt
  configs/
    tip_report_bestfit.json
    lcdm_report_baseline.json
  tip_boltzmann/
    run_tip_camb.py
    plot_results.py
  results/
    *.csv
    *.json
  figures/
    *.png
  paper/
    boltzmann_implementation_section.tex
    main-2_with_boltzmann_insert.tex
.github/workflows/
  boltzmann-smoke-test.yml
```

## Suggested commit message

```text
Add CAMB Boltzmann benchmark for transition-clock model
```

## Suggested release tag

```text
v0.2.0-boltzmann-benchmark
```
