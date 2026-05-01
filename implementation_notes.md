# Implementation notes

The benchmark uses the normalized transition-clock background

```text
E(a) = [Omega_m a^-3 + Omega_r a^-4]^[1/(2(1+n(a)))]
       / [Omega_m + Omega_r]^[1/(2(1+n(1)))]
```

with

```text
n(z) = n0 / [1 + ((1+z)/(1+zt))^beta].
```

The BAO likelihood uses `D_M/r_d`, `D_H/r_d`, and `D_V/r_d`. For anisotropic DESI rows it uses the listed within-bin `D_M-D_H` correlations. No cross-bin covariances are included in this benchmark.

Growth assumes the conservative minimal branch `mu=gamma=Sigma=1`, so growth is altered only through `E(a)`. This is exactly the non-Boltzmann first step; the perturbation constraint and full CMB spectra are later work.
