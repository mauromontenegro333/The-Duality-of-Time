# CAMB Boltzmann Run Summary

Run date: generated in this ChatGPT session.

## Implementation actually run

- Code: CAMB 1.6.6
- TIP mode: stock-CAMB `DarkEnergyPPF` with tabulated effective \(w(a)\)
- Baseline: flat \(\Lambda\)CDM
- Multipoles produced: \(2 \le \ell \le 800\)
- Matter power range: \(10^{-4} \le k/(h\,{\rm Mpc}^{-1}) \le 2.0\)

## Key derived numbers

| Model | H0 | Omega_m | r_d CAMB [Mpc] | z_star | 100 theta_star | Age [Gyr] | z_eq |
|---|---:|---:|---:|---:|---:|---:|---:|
| TIP effective PPF | 65.545 | 0.084 | 88.13 | 1101.76 | 0.6661 | 13.231 | 861.81 |
| LCDM baseline | 67.218 | 0.275 | 152.17 | 1088.31 | 1.0201 | 14.341 | 2969.71 |

## What the run says

The code runs and produces all requested observables:
\(C_\ell^{TT}\), \(C_\ell^{TE}\), \(C_\ell^{EE}\), \(C_L^{\phi\phi}\), \(P(k,z)\), \(f\sigma_8(z)\), \(D_M/r_d\), and \(D_H/r_d\).

The TIP effective-PPF mapping produces a CAMB drag sound horizon of 88.13 Mpc, far below the earlier benchmark proxy \(r_d \simeq 142.91\) Mpc. That is the most important warning from this run. It means this should go into the paper as a Boltzmann implementation benchmark / failure diagnostic, not as a validation claim.

The very small TIP CAMB \(\sigma_8\) and \(f\sigma_8\) values also show that this effective mapping is not yet a viable final perturbation implementation.

## Placement decision

Put this in the existing repository, not a new repository:

```text
mauromontenegro333/The-Duality-of-Time/boltzmann/
```

Add the LaTeX section from:

```text
paper/boltzmann_implementation_section.tex
```

after the current section `Implementation checklist for the later likelihood paper` and before `Discussion and conclusion`.
