#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_ell(df):
    return df[df["ell"] >= 2].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--tip-name", default="tip_report_bestfit")
    ap.add_argument("--lcdm-name", default="lcdm_report_baseline")
    args = ap.parse_args()

    r = Path(args.results)
    fdir = Path(args.figures)
    fdir.mkdir(parents=True, exist_ok=True)

    tip_cls = safe_ell(pd.read_csv(r / f"cmb_cls_{args.tip_name}.csv"))
    lcdm_cls = safe_ell(pd.read_csv(r / f"cmb_cls_{args.lcdm_name}.csv"))

    for col, ylabel, fname in [
        ("TT_total_muK2", r"$C_\ell^{TT}$ [$\mu K^2$]", "cmb_TT_raw_cl.png"),
        ("EE_total_muK2", r"$C_\ell^{EE}$ [$\mu K^2$]", "cmb_EE_raw_cl.png"),
        ("TE_total_muK2", r"$C_\ell^{TE}$ [$\mu K^2$]", "cmb_TE_raw_cl.png"),
        ("cl_phiphi", r"$C_L^{\phi\phi}$", "cmb_lensing_phiphi_raw_cl.png"),
    ]:
        plt.figure(figsize=(7.5, 5))
        # For TE, sign can cross; use linear scale.
        if col == "TE_total_muK2":
            plt.plot(tip_cls["ell"], tip_cls[col], label="TIP effective PPF")
            plt.plot(lcdm_cls["ell"], lcdm_cls[col], label="LCDM")
            plt.axhline(0, lw=0.8)
            plt.ylabel(ylabel)
        else:
            plt.loglog(tip_cls["ell"], np.abs(tip_cls[col]), label="TIP effective PPF")
            plt.loglog(lcdm_cls["ell"], np.abs(lcdm_cls[col]), label="LCDM")
            plt.ylabel("|" + ylabel + "|")
        plt.xlabel(r"$\ell$")
        plt.title(fname.replace("_", " ").replace(".png", ""))
        plt.legend()
        plt.tight_layout()
        plt.savefig(fdir / fname, dpi=200)
        plt.close()

    tip_pk = pd.read_csv(r / f"matter_power_{args.tip_name}.csv")
    lcdm_pk = pd.read_csv(r / f"matter_power_{args.lcdm_name}.csv")
    # Choose z=0 column robustly.
    tip_col = [c for c in tip_pk.columns if c.endswith("_z0") or c.endswith("_z0.0")]
    lcdm_col = [c for c in lcdm_pk.columns if c.endswith("_z0") or c.endswith("_z0.0")]
    if not tip_col:
        tip_col = [c for c in tip_pk.columns if "z0" in c][-1:]
    if not lcdm_col:
        lcdm_col = [c for c in lcdm_pk.columns if "z0" in c][-1:]
    plt.figure(figsize=(7.5, 5))
    plt.loglog(tip_pk["k_h_Mpc"], tip_pk[tip_col[0]], label="TIP effective PPF")
    plt.loglog(lcdm_pk["k_h_Mpc"], lcdm_pk[lcdm_col[0]], label="LCDM")
    plt.xlabel(r"$k$ [$h\,{\rm Mpc}^{-1}$]")
    plt.ylabel(r"$P(k,z=0)$ [$(\mathrm{Mpc}/h)^3$]")
    plt.title("Matter power spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / "matter_power_z0.png", dpi=200)
    plt.close()

    tip_fs = pd.read_csv(r / f"fsigma8_{args.tip_name}.csv")
    lcdm_fs = pd.read_csv(r / f"fsigma8_{args.lcdm_name}.csv")
    plt.figure(figsize=(7.5, 5))
    plt.plot(tip_fs["z"], tip_fs["fsigma8"], marker="o", label="TIP effective PPF")
    plt.plot(lcdm_fs["z"], lcdm_fs["fsigma8"], marker="o", label="LCDM")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.title("Growth observable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / "fsigma8.png", dpi=200)
    plt.close()

    tip_bg = pd.read_csv(r / f"background_{args.tip_name}.csv")
    lcdm_bg = pd.read_csv(r / f"background_{args.lcdm_name}.csv")
    plt.figure(figsize=(7.5, 5))
    plt.plot(tip_bg["z"], tip_bg["E_z"], label="TIP effective PPF")
    plt.plot(lcdm_bg["z"], lcdm_bg["E_z"], label="LCDM")
    plt.xlabel("z")
    plt.ylabel(r"$E(z)=H(z)/H_0$")
    plt.title("Expansion history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / "E_z.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.5, 5))
    plt.plot(tip_bg["z"], tip_bg["q_z"], label="TIP effective PPF")
    plt.plot(lcdm_bg["z"], lcdm_bg["q_z"], label="LCDM")
    plt.axhline(0, lw=0.8)
    plt.xlabel("z")
    plt.ylabel(r"$q(z)$")
    plt.title("Acceleration history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / "q_z.png", dpi=200)
    plt.close()

    tip_bao = pd.read_csv(r / f"bao_distances_{args.tip_name}.csv")
    lcdm_bao = pd.read_csv(r / f"bao_distances_{args.lcdm_name}.csv")
    plt.figure(figsize=(7.5, 5))
    plt.plot(tip_bao["z"], tip_bao["D_M_over_r_d"], marker="o", label=r"TIP $D_M/r_d$")
    plt.plot(lcdm_bao["z"], lcdm_bao["D_M_over_r_d"], marker="o", label=r"LCDM $D_M/r_d$")
    plt.plot(tip_bao["z"], tip_bao["D_H_over_r_d"], marker="s", label=r"TIP $D_H/r_d$")
    plt.plot(lcdm_bao["z"], lcdm_bao["D_H_over_r_d"], marker="s", label=r"LCDM $D_H/r_d$")
    plt.xlabel("z")
    plt.ylabel("BAO distance ratio")
    plt.title("BAO outputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fdir / "bao_distance_ratios.png", dpi=200)
    plt.close()

    print(f"[OK] wrote figures to {fdir}")


if __name__ == "__main__":
    main()
