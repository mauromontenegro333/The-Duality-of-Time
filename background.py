from __future__ import annotations

import numpy as np

from .constants import OMEGA_R_DEFAULT


def n_logistic_z(z, n0, zt, beta):
    """Transition-clock index n(z) = n0/[1 + ((1+z)/(1+zt))^beta].

    This is the code-ready redshift form used in the current manuscript. It is
    equivalent to the scale-factor form n(a)=n0/[1+(a_t/a)^beta] with
    a_t=1/(1+zt).
    """
    z = np.asarray(z, dtype=float)
    return n0 / (1.0 + ((1.0 + z) / (1.0 + zt)) ** beta)


def n_logistic_a(a, n0, zt, beta):
    a = np.asarray(a, dtype=float)
    z = 1.0 / a - 1.0
    return n_logistic_z(z, n0, zt, beta)


def e_einstein_sector(a, omega_m, omega_r=OMEGA_R_DEFAULT):
    """Matter+radiation Einstein-sector expansion before clock dressing."""
    a = np.asarray(a, dtype=float)
    return np.sqrt(omega_m * a ** -3 + omega_r * a ** -4)


def e_tip_z(z, n0, zt, beta, omega_m, omega_r=OMEGA_R_DEFAULT):
    """Normalized transition-clock H(z)/H0.

    E(a) = [Omega_m a^-3 + Omega_r a^-4]^[1/(2(1+n(a)))] / same at a=1.

    There is no independent Lambda or dark-energy density in this branch. The
    denominator enforces E(0)=1 for likelihood comparisons.
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    n = n_logistic_a(a, n0, zt, beta)
    n_today = n_logistic_a(1.0, n0, zt, beta)
    base = omega_m * a ** -3 + omega_r * a ** -4
    base_today = omega_m + omega_r
    return base ** (1.0 / (2.0 * (1.0 + n))) / base_today ** (1.0 / (2.0 * (1.0 + n_today)))


def e_lcdm_z(z, omega_m, omega_r=OMEGA_R_DEFAULT):
    z = np.asarray(z, dtype=float)
    omega_lambda = 1.0 - omega_m - omega_r
    return np.sqrt(omega_m * (1.0 + z) ** 3 + omega_r * (1.0 + z) ** 4 + omega_lambda)


def dlnE_dlna_numeric(model_e_func, z, dz=1e-4):
    """Numerically estimate d ln E / d ln a at a given redshift."""
    z = float(z)
    a = 1.0 / (1.0 + z)
    da = max(1e-5, abs(a) * dz)
    a1 = max(1e-8, a - da)
    a2 = min(1.0 + 1e-6, a + da)
    z1 = 1.0 / a1 - 1.0
    z2 = 1.0 / a2 - 1.0
    e1 = float(model_e_func(z1))
    e2 = float(model_e_func(z2))
    return (np.log(e2) - np.log(e1)) / (np.log(a2) - np.log(a1))


def q_parameter(model_e_func, z):
    """q(z) = -1 - d ln H/d ln a."""
    return -1.0 - dlnE_dlna_numeric(model_e_func, z)


def effective_w_from_e(model_e_func, z):
    """Background-equivalent effective w(z), useful only as a diagnostic."""
    return -1.0 - (2.0 / 3.0) * dlnE_dlna_numeric(model_e_func, z)
