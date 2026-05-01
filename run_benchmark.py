#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from tcc.background import q_parameter, effective_w_from_e, n_logistic_z
from tcc.distances import bao_observables, acoustic_scale_proxy, redshift_drift_velocity
from tcc.growth import growth_solution
from tcc.io import write_json, write_csv
from tcc.likelihood import (
    BenchmarkLikelihood,
    TIP_BOUNDS,
    LCDM_BOUNDS,
    tip_from_array,
    lcdm_from_array,
    tip_log_posterior,
)
from tcc.plotting import line_plot, corner_plot


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def _bounds_list(bounds, names):
    return np.array([bounds[n] for n in names], dtype=float)


def _random_search(objective, bounds, n, seed):
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    best_x = None
    best_y = math.inf
    for i in range(n):
        x = lo + rng.random(len(lo)) * (hi - lo)
        y = float(objective(x))
        if math.isfinite(y) and y < best_y:
            best_x = x.copy()
            best_y = y
    return best_x, best_y


def _coordinate_refine(objective, x0, bounds, step_frac=0.08, cycles=6):
    x = np.array(x0, dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    steps = (hi - lo) * step_frac
    best = float(objective(x))
    for _ in range(cycles):
        improved = False
        for j in range(len(x)):
            for sign in (1.0, -1.0):
                trial = x.copy()
                trial[j] = np.clip(trial[j] + sign * steps[j], lo[j], hi[j])
                y = float(objective(trial))
                if math.isfinite(y) and y < best:
                    x = trial
                    best = y
                    improved = True
        if not improved:
            steps *= 0.55
        if np.all(steps < (hi - lo) * 1e-5):
            break
    return x, best


def fit_tip(like, random_n=8000):
    names = ["n0", "zt", "beta", "omega_m", "H0", "rd"]
    bounds = _bounds_list(TIP_BOUNDS, names)

    def obj(x):
        return like.chi2_total(tip_from_array(x))

    x0, y0 = _random_search(obj, bounds, random_n, seed=123)
    x, y = _coordinate_refine(obj, x0, bounds, step_frac=0.08, cycles=6)
    return names, x, y


def fit_lcdm(like, random_n=4000):
    names = ["omega_m", "H0", "rd"]
    bounds = _bounds_list(LCDM_BOUNDS, names)

    def obj(x):
        return like.chi2_total(lcdm_from_array(x))

    x0, y0 = _random_search(obj, bounds, random_n, seed=456)
    x, y = _coordinate_refine(obj, x0, bounds, step_frac=0.08, cycles=6)
    return names, x, y


def run_mh(logpost, x0, step, nsteps, seed=789):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    lp = float(logpost(x))
    samples = []
    lps = []
    accepted = 0
    for i in range(nsteps):
        prop = x + rng.normal(0.0, step, size=len(x))
        lpp = float(logpost(prop))
        if math.isfinite(lpp) and math.log(rng.random()) < lpp - lp:
            x = prop
            lp = lpp
            accepted += 1
        samples.append(x.copy())
        lps.append(lp)
    return np.array(samples), np.array(lps), accepted / nsteps


def summarize_samples(samples, names):
    rows = []
    for i, name in enumerate(names):
        vals = samples[:, i]
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        rows.append({"parameter": name, "p16": q16, "median": q50, "p84": q84, "minus": q50 - q16, "plus": q84 - q50})
    return rows


def as_float_dict(rows):
    return [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in rows]


def make_predictions(tip, lcdm):
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


def make_plots(tip, lcdm, samples=None, sample_names=None):
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
    if samples is not None and sample_names is not None:
        corner_plot(PLOTS_DIR / "tip_corner.png", samples, sample_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn-q0", default="corrected_combined", choices=["none", "corrected_combined", "uncorrected_combined", "BAO_CMB_Pantheon_plus_corrected", "BAO_CMB_DES5Y_corrected"])
    ap.add_argument("--no-cmb", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    like = BenchmarkLikelihood(DATA_DIR, use_cmb_proxy=not args.no_cmb, use_sn_q0=None if args.sn_q0 == "none" else args.sn_q0)

    random_tip = 50 if args.quick else 200
    random_lcdm = 30 if args.quick else 100
    print("[run] fitting transition-clock model", flush=True)
    tip_names, tip_x, tip_chi2 = fit_tip(like, random_n=random_tip)
    print("[run] transition-clock fit complete", flush=True)
    print("[run] fitting LCDM baseline", flush=True)
    lcdm_names, lcdm_x, lcdm_chi2 = fit_lcdm(like, random_n=random_lcdm)
    print("[run] LCDM fit complete", flush=True)
    tip = tip_from_array(tip_x)
    lcdm = lcdm_from_array(lcdm_x)

    steps = np.array([0.025, 0.22, 0.35, 0.012, 0.55, 1.1])
    nsteps = 100 if args.quick else 200
    print("[run] sampling transition-clock posterior", flush=True)
    samples, lps, acc = run_mh(lambda x: tip_log_posterior(x, like), tip_x, steps, nsteps=nsteps)
    print("[run] sampling complete", flush=True)
    burn = nsteps // 3
    chain = samples[burn:]
    sample_rows = summarize_samples(chain, tip_names)

    print("[run] computing predictions and plots", flush=True)
    pred_rows = make_predictions(tip, lcdm)
    make_plots(tip, lcdm, chain, tip_names)
    print("[run] predictions and plots complete", flush=True)

    bao_rows = []
    for r in like.bao_rows:
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

    n_data = len(like.bao_rows) + (1 if like.use_cmb_proxy else 0) + (1 if like.q0_prior else 0)
    summary = {
        "run_name": "BAO + broad CMB acoustic proxy" + (" + q0 summary" if like.q0_prior else ""),
        "data_note": "DESI uses manuscript Table I with block-diagonal within-bin correlations; CMB and SN are summary/proxy constraints, not official likelihoods.",
        "q0_prior": like.q0_prior,
        "n_data": n_data,
        "tip": {
            "best_fit": tip.as_dict(),
            "chi2_components": like.components(tip),
            "k_params": len(tip_names),
            "AIC": float(tip_chi2 + 2 * len(tip_names)),
            "BIC": float(tip_chi2 + len(tip_names) * math.log(n_data)),
            "ell_A_proxy": float(acoustic_scale_proxy(like.cmb_proxy["z_star"], tip.H0, tip.rd, tip.e_func())),
            "q0": float(tip.q0()),
            "mcmc_acceptance": float(acc),
        },
        "lcdm": {
            "best_fit": lcdm.as_dict(),
            "chi2_components": like.components(lcdm),
            "k_params": len(lcdm_names),
            "AIC": float(lcdm_chi2 + 2 * len(lcdm_names)),
            "BIC": float(lcdm_chi2 + len(lcdm_names) * math.log(n_data)),
            "ell_A_proxy": float(acoustic_scale_proxy(like.cmb_proxy["z_star"], lcdm.H0, lcdm.rd, lcdm.e_func())),
            "q0": float(lcdm.q0()),
        },
    }
    write_json(RESULTS_DIR / "benchmark_summary.json", summary)
    write_csv(RESULTS_DIR / "tip_posterior_summary.csv", as_float_dict(sample_rows), ["parameter", "p16", "median", "p84", "minus", "plus"])
    write_csv(RESULTS_DIR / "predictions_table.csv", as_float_dict(pred_rows), ["model", "z", "E", "q", "n", "w_eff_diagnostic", "drift_cm_s_20yr", "D_growth", "f_growth", "fsigma8", "EG"])
    write_csv(RESULTS_DIR / "bao_residuals.csv", as_float_dict(bao_rows), ["tracer", "z", "observable", "value", "sigma", "tip", "tip_pull", "lcdm", "lcdm_pull"])
    np.savez(RESULTS_DIR / "tip_chain.npz", samples=chain, names=np.array(tip_names), logpost=lps[burn:])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
