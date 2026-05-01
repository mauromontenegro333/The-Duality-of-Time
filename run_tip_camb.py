#!/usr/bin/env python3
"""
Transition-clock / TIP CAMB runner.

This is a stock-CAMB Boltzmann run using an effective-background mapping of the
clock-dressed H(a) into a tabulated PPF dark-energy equation of state w(a).

Important:
    This is not a native CLASS/CAMB modification of the clock constraint.
    It is the reproducible CAMB/PPF implementation layer that produces the
    requested Boltzmann observables:
      C_l^TT, C_l^TE, C_l^EE, C_L^phiphi, P(k,z), f sigma_8(z),
      D_M/r_d and D_H/r_d.

The native next step is to modify CLASS/hi_class/EFTCAMB at the background and
perturbation-equation level. This script is deliberately conservative and
transparent about the mapping used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import camb
from camb import dark_energy, model


C_LIGHT_KM_S = 299792.458


@dataclass
class ModelConfig:
    name: str
    kind: str  # "lcdm" or "tip_effective_ppf"
    H0: float
    Omega_m: float
    ombh2: float = 0.02237
    tau: float = 0.0544
    As: float = 2.10e-9
    ns: float = 0.965
    sigma8_norm_for_tables: float = 0.811
    n0: float = 0.593
    zt: float = 1.842
    beta: float = 11.604
    Omega_r_for_tip_background: float = 9.2e-5
    lmax: int = 800
    kmax: float = 2.0
    npoints_pk: int = 220
    amin_w_table: float = 1e-5
    n_w_table: int = 500
    lens_potential_accuracy: int = 0


DESI_BAO_Z = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
FSIGMA8_Z = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])


def n_of_a(a: np.ndarray | float, n0: float, zt: float, beta: float) -> np.ndarray | float:
    a = np.asarray(a)
    return n0 / (1.0 + (1.0 / (a * (1.0 + zt))) ** beta)


def tip_E_of_a(
    a: np.ndarray | float,
    n0: float,
    zt: float,
    beta: float,
    Omega_m: float,
    Omega_r: float,
) -> np.ndarray | float:
    """
    Normalized clock-dressed background used in the manuscript addendum Eq. (C54):

        E_TIP(a) = [Omega_m a^-3 + Omega_r a^-4]^{1/[2(1+n(a))]}
                   / [Omega_m + Omega_r]^{1/[2(1+n(1))]}

    This is the background that is mapped into CAMB as an effective PPF fluid.
    """
    a = np.asarray(a)
    n = n_of_a(a, n0, zt, beta)
    n1 = float(n_of_a(1.0, n0, zt, beta))
    norm = (Omega_m + Omega_r) ** (1.0 / (2.0 * (1.0 + n1)))
    x = Omega_m * a ** (-3.0) + Omega_r * a ** (-4.0)
    return x ** (1.0 / (2.0 * (1.0 + n))) / norm


def effective_de_table(cfg: ModelConfig) -> pd.DataFrame:
    a = np.geomspace(cfg.amin_w_table, 1.0, cfg.n_w_table)
    E = tip_E_of_a(
        a,
        cfg.n0,
        cfg.zt,
        cfg.beta,
        cfg.Omega_m,
        cfg.Omega_r_for_tip_background,
    )
    rho_de = E**2 - cfg.Omega_m * a**(-3.0) - cfg.Omega_r_for_tip_background * a**(-4.0)
    if np.any(rho_de <= 0):
        bad = np.where(rho_de <= 0)[0][:10].tolist()
        raise ValueError(f"effective rho_de <= 0 at table indices {bad}; cannot log-differentiate.")
    dlnrho_dln_a = np.gradient(np.log(rho_de), np.log(a), edge_order=2)
    w = -1.0 - dlnrho_dln_a / 3.0
    return pd.DataFrame(
        {
            "a": a,
            "z": 1.0 / a - 1.0,
            "n_a": n_of_a(a, cfg.n0, cfg.zt, cfg.beta),
            "E_tip": E,
            "rho_de_eff_over_rhoc0": rho_de,
            "w_eff_ppf": w,
        }
    )


def build_camb_params(cfg: ModelConfig) -> camb.CAMBparams:
    h = cfg.H0 / 100.0
    ommh2 = cfg.Omega_m * h * h
    omch2 = ommh2 - cfg.ombh2
    if omch2 <= 0:
        raise ValueError(
            f"Omega_m h^2={ommh2:.6g} is not larger than ombh2={cfg.ombh2:.6g}; "
            "choose a smaller ombh2 or larger Omega_m/H0."
        )

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cfg.H0, ombh2=cfg.ombh2, omch2=omch2, tau=cfg.tau)
    pars.InitPower.set_params(As=cfg.As, ns=cfg.ns)
    pars.NonLinear = model.NonLinear_none
    pars.Want_CMB = True
    pars.Want_CMB_lensing = True
    pars.set_for_lmax(cfg.lmax, lens_potential_accuracy=cfg.lens_potential_accuracy)
    pars.set_matter_power(redshifts=FSIGMA8_Z.tolist(), kmax=cfg.kmax)

    if cfg.kind == "tip_effective_ppf":
        tab = effective_de_table(cfg)
        de = dark_energy.DarkEnergyPPF()
        de.set_w_a_table(tab["a"].values, tab["w_eff_ppf"].values)
        pars.DarkEnergy = de
    elif cfg.kind == "lcdm":
        pass
    else:
        raise ValueError(f"Unknown model kind: {cfg.kind}")

    return pars


def run_model(cfg: ModelConfig, outdir: Path) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    pars = build_camb_params(cfg)

    results = camb.get_results(pars)
    derived = results.get_derived_params()

    # CMB Cls: raw Cl. CMB in muK^2; lensing potential raw C_L^phiphi.
    cmb = results.get_cmb_power_spectra(
        pars,
        lmax=cfg.lmax,
        spectra=["total", "unlensed_scalar", "lensed_scalar"],
        CMB_unit="muK",
        raw_cl=True,
    )
    lens = results.get_lens_potential_cls(lmax=cfg.lmax, CMB_unit="muK", raw_cl=True)

    ell = np.arange(cfg.lmax + 1)
    cls_df = pd.DataFrame(
        {
            "ell": ell,
            "TT_total_muK2": cmb["total"][:, 0],
            "EE_total_muK2": cmb["total"][:, 1],
            "BB_total_muK2": cmb["total"][:, 2],
            "TE_total_muK2": cmb["total"][:, 3],
            "TT_unlensed_scalar_muK2": cmb["unlensed_scalar"][:, 0],
            "EE_unlensed_scalar_muK2": cmb["unlensed_scalar"][:, 1],
            "TE_unlensed_scalar_muK2": cmb["unlensed_scalar"][:, 3],
            "cl_phiphi": lens[:, 0],
            "cl_phit_muK": lens[:, 1],
            "cl_phie_muK": lens[:, 2],
        }
    )
    cls_df.to_csv(outdir / f"cmb_cls_{cfg.name}.csv", index=False)

    # Matter power P(k,z), k in h/Mpc, P in (Mpc/h)^3.
    kh, z_pk, pk = results.get_matter_power_spectrum(
        minkh=1.0e-4, maxkh=cfg.kmax, npoints=cfg.npoints_pk
    )
    pk_df = pd.DataFrame({"k_h_Mpc": kh})
    for zi, row in zip(z_pk, pk):
        pk_df[f"P_delta_delta_z{zi:g}"] = row
    pk_df.to_csv(outdir / f"matter_power_{cfg.name}.csv", index=False)

    # sigma8/fsigma8 arrays follow CAMB's sorted redshift order.
    sigma8 = np.array(results.get_sigma8())
    fsigma8 = np.array(results.get_fsigma8())
    fs8_df = pd.DataFrame(
        {
            "z": z_pk,
            "sigma8_z": sigma8,
            "fsigma8": fsigma8,
            "f_growth_rate": fsigma8 / sigma8,
        }
    ).sort_values("z")
    fs8_df.to_csv(outdir / f"fsigma8_{cfg.name}.csv", index=False)

    # BAO distances.
    rdrag = float(derived["rdrag"])
    bao_rows = []
    for z in DESI_BAO_Z:
        DA = float(results.angular_diameter_distance(z))  # Mpc
        DM = (1.0 + z) * DA
        H_z = float(results.hubble_parameter(z))  # km/s/Mpc
        DH = C_LIGHT_KM_S / H_z
        DV = (z * DM * DM * DH) ** (1.0 / 3.0)
        bao_rows.append(
            {
                "z": z,
                "H_z_km_s_Mpc": H_z,
                "D_M_Mpc": DM,
                "D_H_Mpc": DH,
                "D_V_Mpc": DV,
                "r_d_Mpc_CAMB": rdrag,
                "D_M_over_r_d": DM / rdrag,
                "D_H_over_r_d": DH / rdrag,
                "D_V_over_r_d": DV / rdrag,
            }
        )
    pd.DataFrame(bao_rows).to_csv(outdir / f"bao_distances_{cfg.name}.csv", index=False)

    # H(z) and q(z) grid from CAMB distances/H. q is numerical derivative: q=-1+(1+z)dlnH/dz.
    zgrid = np.linspace(0, 5, 501)
    Hz = np.array([results.hubble_parameter(float(z)) for z in zgrid])
    E = Hz / cfg.H0
    q = -1.0 + (1.0 + zgrid) * np.gradient(np.log(Hz), zgrid, edge_order=2)
    bg_df = pd.DataFrame({"z": zgrid, "H_z_km_s_Mpc": Hz, "E_z": E, "q_z": q})
    bg_df.to_csv(outdir / f"background_{cfg.name}.csv", index=False)

    if cfg.kind == "tip_effective_ppf":
        tab = effective_de_table(cfg)
        tab.to_csv(outdir / f"effective_w_table_{cfg.name}.csv", index=False)

    # Derived JSON with model config and code provenance.
    provenance = {
        "code": "CAMB",
        "camb_version": camb.__version__,
        "implementation": cfg.kind,
        "mapping_warning": (
            "TIP is run through a stock-CAMB PPF effective-background mapping. "
            "This is a Boltzmann run, but not a native clock-constraint Boltzmann modification."
            if cfg.kind == "tip_effective_ppf"
            else "Standard flat LCDM baseline."
        ),
        "config": asdict(cfg),
        "derived": {k: float(v) for k, v in derived.items()},
        "Omega_m_h2": cfg.Omega_m * (cfg.H0 / 100.0) ** 2,
        "Omega_c_h2": cfg.Omega_m * (cfg.H0 / 100.0) ** 2 - cfg.ombh2,
        "outputs": {
            "cmb_cls": f"cmb_cls_{cfg.name}.csv",
            "matter_power": f"matter_power_{cfg.name}.csv",
            "fsigma8": f"fsigma8_{cfg.name}.csv",
            "bao_distances": f"bao_distances_{cfg.name}.csv",
            "background": f"background_{cfg.name}.csv",
        },
    }
    if cfg.kind == "tip_effective_ppf":
        wtab = effective_de_table(cfg)
        provenance["effective_w_summary"] = {
            "w_min": float(wtab["w_eff_ppf"].min()),
            "w_max": float(wtab["w_eff_ppf"].max()),
            "rho_de_eff_min": float(wtab["rho_de_eff_over_rhoc0"].min()),
            "rho_de_eff_at_a1": float(wtab["rho_de_eff_over_rhoc0"].iloc[-1]),
        }

    with open(outdir / f"derived_{cfg.name}.json", "w") as f:
        json.dump(provenance, f, indent=2)

    return provenance


def load_config(path: Path) -> ModelConfig:
    data = json.loads(path.read_text())
    return ModelConfig(**data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True, help="Path to model JSON config. Can repeat.")
    ap.add_argument("--outdir", default="results", help="Output directory.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    summary = []
    for cpath in args.config:
        cfg = load_config(Path(cpath))
        print(f"[RUN] {cfg.name} ({cfg.kind})")
        summary.append(run_model(cfg, outdir))
        print(f"[OK] {cfg.name}")

    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] wrote {outdir}")


if __name__ == "__main__":
    main()
