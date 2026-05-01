from __future__ import annotations

import numpy as np

from .constants import C_KM_S


def _integrate_trapz_x(xs, ys):
    return float(np.trapezoid(ys, xs))


def comoving_distance(z, H0, e_func, n_grid=1600):
    """Flat-universe transverse comoving distance D_M in Mpc."""
    z = float(z)
    if z <= 0:
        return 0.0
    if z <= 10.0:
        xs = np.linspace(0.0, z, n_grid)
        integrand = 1.0 / e_func(xs)
        integral = _integrate_trapz_x(xs, integrand)
    else:
        # Log grid is much more accurate for CMB-distance diagnostics.
        xmax = np.log1p(z)
        xs = np.linspace(0.0, xmax, max(n_grid, 5000))
        zz = np.expm1(xs)
        integrand = np.exp(xs) / e_func(zz)
        integral = _integrate_trapz_x(xs, integrand)
    return C_KM_S / H0 * integral


def hubble_distance(z, H0, e_func):
    return C_KM_S / (H0 * float(e_func(float(z))))


def volume_distance(z, H0, e_func):
    dm = comoving_distance(z, H0, e_func)
    dh = hubble_distance(z, H0, e_func)
    return (float(z) * dm * dm * dh) ** (1.0 / 3.0)


def bao_observables(z, H0, rd, e_func):
    dm = comoving_distance(z, H0, e_func)
    dh = hubble_distance(z, H0, e_func)
    dv = (float(z) * dm * dm * dh) ** (1.0 / 3.0) if z > 0 else 0.0
    return {
        "DM_over_rd": dm / rd,
        "DH_over_rd": dh / rd,
        "DV_over_rd": dv / rd,
        "DM_over_DH": dm / dh if dh > 0 else np.nan,
    }


def acoustic_scale_proxy(z_star, H0, rd, e_func):
    """ell_A proxy = pi D_M(z*)/r_d. This is not a full CMB likelihood."""
    dm = comoving_distance(z_star, H0, e_func, n_grid=10000)
    return np.pi * dm / rd


def redshift_drift_velocity(z, H0, e_func, delta_t_years=20.0):
    """Velocity shift in cm/s over observer baseline delta_t_years."""
    seconds_per_year = 365.25 * 24.0 * 3600.0
    mpc_km = 3.0856775814913673e19
    h0_s = H0 / mpc_km
    c_cm_s = C_KM_S * 1.0e5
    z = np.asarray(z, dtype=float)
    factor = c_cm_s * h0_s * delta_t_years * seconds_per_year
    return factor * (1.0 - e_func(z) / (1.0 + z))
