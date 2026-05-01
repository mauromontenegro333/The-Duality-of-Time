from __future__ import annotations

from dataclasses import dataclass

from .background import e_lcdm_z, e_tip_z, q_parameter
from .constants import OMEGA_R_DEFAULT


@dataclass(frozen=True)
class TipParams:
    n0: float
    zt: float
    beta: float
    omega_m: float
    H0: float
    rd: float
    omega_r: float = OMEGA_R_DEFAULT

    def e_func(self):
        return lambda z: e_tip_z(z, self.n0, self.zt, self.beta, self.omega_m, self.omega_r)

    def q0(self):
        return q_parameter(self.e_func(), 0.0)

    def as_dict(self):
        return {
            "n0": self.n0,
            "zt": self.zt,
            "beta": self.beta,
            "omega_m": self.omega_m,
            "H0": self.H0,
            "rd": self.rd,
            "omega_r": self.omega_r,
        }


@dataclass(frozen=True)
class LCDMParams:
    omega_m: float
    H0: float
    rd: float
    omega_r: float = OMEGA_R_DEFAULT

    def e_func(self):
        return lambda z: e_lcdm_z(z, self.omega_m, self.omega_r)

    def q0(self):
        return q_parameter(self.e_func(), 0.0)

    def as_dict(self):
        return {
            "omega_m": self.omega_m,
            "H0": self.H0,
            "rd": self.rd,
            "omega_r": self.omega_r,
        }
