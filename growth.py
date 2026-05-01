from __future__ import annotations

import numpy as np


def growth_solution(e_func, omega_m, z_eval, sigma8_0=0.811, a_initial=1e-3, dx=0.01):
    """Solve GR-like linear growth on a supplied background using fixed-step RK4.

    The equation is
      D_xx + [2 + dlnE/dlna] D_x - 3/2 Omega_m a^-3/E(a)^2 D = 0,
    with x = ln a. The solution is normalized to D(a=1)=1.

    This implementation avoids heavy solver dependencies so it runs in minimal
    reproducibility environments.
    """
    z_eval = np.asarray(z_eval, dtype=float)
    x0 = float(np.log(a_initial))
    x1 = 0.0
    nstep = int(np.ceil((x1 - x0) / dx))
    xs = np.linspace(x0, x1, nstep + 1)
    h = xs[1] - xs[0]

    def E_of_x(x):
        a = np.exp(x)
        z = 1.0 / a - 1.0
        return float(e_func(z))

    def dlnE_dx(x):
        hh = 1e-4
        return (np.log(E_of_x(x + hh)) - np.log(E_of_x(x - hh))) / (2.0 * hh)

    def rhs(x, y):
        D, V = y
        a = np.exp(x)
        E = E_of_x(x)
        source = 1.5 * omega_m * a ** -3 / (E * E)
        return np.array([V, -(2.0 + dlnE_dx(x)) * V + source * D], dtype=float)

    y = np.zeros((len(xs), 2), dtype=float)
    y[0] = [a_initial, a_initial]
    for i in range(len(xs) - 1):
        x = xs[i]
        yi = y[i]
        k1 = rhs(x, yi)
        k2 = rhs(x + 0.5 * h, yi + 0.5 * h * k1)
        k3 = rhs(x + 0.5 * h, yi + 0.5 * h * k2)
        k4 = rhs(x + h, yi + h * k3)
        y[i + 1] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    D_today = y[-1, 0]
    x_eval = np.log(1.0 / (1.0 + z_eval))
    D_interp = np.interp(x_eval, xs, y[:, 0]) / D_today
    V_interp = np.interp(x_eval, xs, y[:, 1])
    D_raw = np.interp(x_eval, xs, y[:, 0])
    f = V_interp / D_raw
    fs8 = f * sigma8_0 * D_interp
    return {"z": z_eval, "D": D_interp, "f": f, "fsigma8": fs8, "EG": omega_m / f}
