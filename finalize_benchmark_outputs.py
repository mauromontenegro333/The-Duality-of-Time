#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from tcc.distances import bao_observables, acoustic_scale_proxy
from tcc.io import write_json, write_csv
from tcc.likelihood import BenchmarkLikelihood, tip_from_array, lcdm_from_array, tip_log_posterior
from tcc.background import q_parameter, effective_w_from_e, n_logistic_z
from tcc.distances import redshift_drift_velocity
from tcc.growth import growth_solution
from tcc.plotting import line_plot, corner_plot
from scripts.run_benchmark import run_mh, summarize_samples, as_float_dict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

# Best points obtained from the stable benchmark search in this container.
TIP_X = np.array([0.5930802540673863, 1.8415239471456921, 11.604, 0.08383930483780594, 65.54483953675593, 142.90682674871198])
LCDM_X = np.array([0.27532499947352035, 67.21838157375264, 153.55090780768253])
TIP_NAMES = ["n0", "zt", "beta", "omega_m", "H0", "rd"]


def _model_rows(tip, lcdm):
    e_tip = tip.e_func()
    e_lcdm = lcdm.e_func()
    z_points = np.array([0, 0.5, 1, 2, 3, 4], dtype=float)
    tip_growth = growth_solution(e_tip, tip.omega_m, z_points, sigma8_0=0.811)
    lcdm_growth = growth_solution(e_lcdm, lcdm.omega_m, z_points, sigma8_0=0.811)
    rows = []
    for idx, z in enumerate(z_points):
        rows.append({
            "model": "TIP",
            "z": z,
            "E": float(e_tip(z)),
            "q": float(q_parameter(e_tip, z)),
            "n": float(n_logistic_z(z, tip.n0, tip.zt, tip.beta)),
            "w_eff_diagnostic": float(effective_w_from_e(e_tip, z)),
            "drift_cm_s_20yr": float(redshift_drift_velocity(z, tip.H0, e_tip, 20.0)),
            "D_growth": float(tip_growth["D"][idx]),
            "f_growth": float(tip_growth["f"][idx]),
            "fsigma8": float(tip_growth["fsigma8"][idx]),
            "EG": float(tip_growth["EG"][idx]),
        })
        rows.append({
            "model": "LCDM",
            "z": z,
            "E": float(e_lcdm(z)),
            "q": float(q_parameter(e_lcdm, z)),
            "n": 0.0,
            "w_eff_diagnostic": float(effective_w_from_e(e_lcdm, z)),
            "drift_cm_s_20yr": float(redshift_drift_velocity(z, lcdm.H0, e_lcdm, 20.0)),
            "D_growth": float(lcdm_growth["D"][idx]),
            "f_growth": float(lcdm_growth["f"][idx]),
            "fsigma8": float(lcdm_growth["fsigma8"][idx]),
            "EG": float(lcdm_growth["EG"][idx]),
        })
    return rows


def _make_plots(tip, lcdm, chain):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    z = np.linspace(0, 5, 301)
    e_tip = tip.e_func()
    e_lcdm = lcdm.e_func()
    line_plot(PLOTS_DIR / "E_of_z.png", [
        {"x": z, "y": e_tip(z), "label": "transition clock"},
        {"x": z, "y": e_lcdm(z), "label": "flat LCDM", "dash": True},
    ], "z", "E(z)=H/H0", "Best-fit expansion history")
    q_tip = np.array([q_parameter(e_tip, zz) for zz in z])
    q_lcdm = np.array([q_parameter(e_lcdm, zz) for zz in z])
    line_plot(PLOTS_DIR / "q_of_z.png", [
        {"x": z, "y": q_tip, "label": "transition clock"},
        {"x": z, "y": q_lcdm, "label": "flat LCDM", "dash": True},
        {"x": z, "y": np.zeros_like(z), "label": "q=0", "dash": True},
    ], "z", "q(z)", "Acceleration history")
    n_vals = n_logistic_z(z, tip.n0, tip.zt, tip.beta)
    line_plot(PLOTS_DIR / "n_of_z.png", [
        {"x": z, "y": n_vals, "label": "n(z)"},
        {"x": z, "y": np.full_like(z, 0.5), "label": "acceleration threshold", "dash": True},
    ], "z", "n(z)", "Clock-index transition")
    drift_tip = redshift_drift_velocity(z, tip.H0, e_tip, 20.0)
    drift_lcdm = redshift_drift_velocity(z, lcdm.H0, e_lcdm, 20.0)
    line_plot(PLOTS_DIR / "redshift_drift.png", [
        {"x": z, "y": drift_tip, "label": "transition clock"},
        {"x": z, "y": drift_lcdm, "label": "flat LCDM", "dash": True},
        {"x": z, "y": np.zeros_like(z), "label": "zero", "dash": True},
    ], "z", "Delta v over 20 yr [cm/s]", "Redshift drift prediction")
    zg = np.linspace(0, 4, 81)
    gt = growth_solution(e_tip, tip.omega_m, zg)
    gl = growth_solution(e_lcdm, lcdm.omega_m, zg)
    line_plot(PLOTS_DIR / "fsigma8.png", [
        {"x": zg, "y": gt["fsigma8"], "label": "transition clock"},
        {"x": zg, "y": gl["fsigma8"], "label": "flat LCDM", "dash": True},
    ], "z", "f sigma8", "Growth prediction")
    corner_plot(PLOTS_DIR / "tip_corner.png", chain, TIP_NAMES, max_points=1000)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    like_bao_cmb = BenchmarkLikelihood(DATA_DIR, use_cmb_proxy=True, use_sn_q0=None)
    like_corrected = BenchmarkLikelihood(DATA_DIR, use_cmb_proxy=True, use_sn_q0="corrected_combined")
    like_uncorrected = BenchmarkLikelihood(DATA_DIR, use_cmb_proxy=True, use_sn_q0="uncorrected_combined")
    tip = tip_from_array(TIP_X)
    lcdm = lcdm_from_array(LCDM_X)

    print("[finalize] starting diagnostic chain", flush=True)
    # Short diagnostic chain only. It is not a publication-grade posterior.
    samples, lps, acc = run_mh(lambda x: tip_log_posterior(x, like_corrected), TIP_X, np.array([0.02, 0.15, 0.25, 0.008, 0.35, 0.7]), nsteps=50)
    chain = samples[10:]
    sample_rows = summarize_samples(chain, TIP_NAMES)
    print("[finalize] chain complete", flush=True)

    n_data_bao_cmb = 14
    n_data_with_q0 = 15
    summary = {
        "data_note": "Benchmark only: DESI DR2 rows are transcribed from the manuscript and treated with block-diagonal within-bin covariances. The CMB term is a broad acoustic-scale proxy. SN information is used only as q0 summary constraints. This is not a public official likelihood run.",
        "best_fit_source": "Stable low-evaluation search in the attached repository. Boltzmann/CMB spectra intentionally deferred.",
        "tip": {
            "best_fit": tip.as_dict(),
            "q0": float(tip.q0()),
            "ell_A_proxy": float(acoustic_scale_proxy(like_bao_cmb.cmb_proxy["z_star"], tip.H0, tip.rd, tip.e_func())),
            "chi2_bao_cmb": like_bao_cmb.components(tip),
            "chi2_with_sn_age_corrected_q0": like_corrected.components(tip),
            "chi2_with_sn_uncorrected_q0": like_uncorrected.components(tip),
            "AIC_bao_cmb": float(like_bao_cmb.components(tip)["total"] + 2 * 6),
            "BIC_bao_cmb": float(like_bao_cmb.components(tip)["total"] + 6 * math.log(n_data_bao_cmb)),
            "AIC_with_age_q0": float(like_corrected.components(tip)["total"] + 2 * 6),
            "BIC_with_age_q0": float(like_corrected.components(tip)["total"] + 6 * math.log(n_data_with_q0)),
            "diagnostic_chain_acceptance": float(acc),
        },
        "lcdm": {
            "best_fit": lcdm.as_dict(),
            "q0": float(lcdm.q0()),
            "ell_A_proxy": float(acoustic_scale_proxy(like_bao_cmb.cmb_proxy["z_star"], lcdm.H0, lcdm.rd, lcdm.e_func())),
            "chi2_bao_cmb": like_bao_cmb.components(lcdm),
            "chi2_with_sn_age_corrected_q0": like_corrected.components(lcdm),
            "chi2_with_sn_uncorrected_q0": like_uncorrected.components(lcdm),
            "AIC_bao_cmb": float(like_bao_cmb.components(lcdm)["total"] + 2 * 3),
            "BIC_bao_cmb": float(like_bao_cmb.components(lcdm)["total"] + 3 * math.log(n_data_bao_cmb)),
            "AIC_with_age_q0": float(like_corrected.components(lcdm)["total"] + 2 * 3),
            "BIC_with_age_q0": float(like_corrected.components(lcdm)["total"] + 3 * math.log(n_data_with_q0)),
        },
    }
    print("[finalize] writing summary", flush=True)
    write_json(RESULTS_DIR / "benchmark_summary.json", summary)
    write_csv(RESULTS_DIR / "tip_posterior_summary.csv", as_float_dict(sample_rows), ["parameter", "p16", "median", "p84", "minus", "plus"])

    print("[finalize] computing prediction table", flush=True)
    pred_rows = _model_rows(tip, lcdm)
    write_csv(RESULTS_DIR / "predictions_table.csv", as_float_dict(pred_rows), ["model", "z", "E", "q", "n", "w_eff_diagnostic", "drift_cm_s_20yr", "D_growth", "f_growth", "fsigma8", "EG"])

    print("[finalize] computing BAO residuals", flush=True)
    bao_rows = []
    for r in like_bao_cmb.bao_rows:
        z = float(r["z"])
        obs = r["observable"]
        value = float(r["value"])
        sigma = float(r["sigma"])
        pred_tip = bao_observables(z, tip.H0, tip.rd, tip.e_func())[obs]
        pred_lcdm = bao_observables(z, lcdm.H0, lcdm.rd, lcdm.e_func())[obs]
        bao_rows.append({
            "tracer": r["tracer"], "z": z, "observable": obs, "value": value, "sigma": sigma,
            "tip": pred_tip, "tip_pull": (pred_tip - value) / sigma,
            "lcdm": pred_lcdm, "lcdm_pull": (pred_lcdm - value) / sigma,
        })
    write_csv(RESULTS_DIR / "bao_residuals.csv", as_float_dict(bao_rows), ["tracer", "z", "observable", "value", "sigma", "tip", "tip_pull", "lcdm", "lcdm_pull"])
    np.savez(RESULTS_DIR / "tip_chain.npz", samples=chain, names=np.array(TIP_NAMES), logpost=lps[10:])
    print("[finalize] making plots", flush=True)
    _make_plots(tip, lcdm, chain)
    print("[finalize] plots complete", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
