from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .distances import acoustic_scale_proxy, bao_observables
from .io import read_csv_dicts, read_json
from .models import TipParams, LCDMParams


TIP_BOUNDS = {
    "n0": (0.5001, 1.5),
    "zt": (0.05, 8.0),
    "beta": (3.0, 12.0),
    "omega_m": (0.05, 0.60),
    "H0": (55.0, 80.0),
    "rd": (120.0, 170.0),
}

LCDM_BOUNDS = {
    "omega_m": (0.05, 0.60),
    "H0": (55.0, 80.0),
    "rd": (120.0, 170.0),
}


class BenchmarkLikelihood:
    """Compact benchmark likelihood.

    Uses the manuscript's DESI table and optional broad CMB acoustic-scale and
    q0 summary constraints. This is deliberately separated from official public
    likelihoods so the final Boltzmann/data paper can swap in official files.
    """

    def __init__(self, data_dir, use_cmb_proxy=True, use_sn_q0="corrected_combined"):
        self.data_dir = Path(data_dir)
        self.bao_rows = read_csv_dicts(self.data_dir / "desi_dr2_bao_table.csv")
        self.corr_rows = read_csv_dicts(self.data_dir / "desi_dr2_bao_correlations.csv")
        self.corr = {r["group"]: float(r["corr_DM_DH"]) for r in self.corr_rows}
        self.cmb_proxy = read_json(self.data_dir / "cmb_proxy.json")
        self.sn_rows = read_csv_dicts(self.data_dir / "sn_q0_summaries.csv")
        self.use_cmb_proxy = use_cmb_proxy
        self.use_sn_q0 = use_sn_q0
        self.q0_prior = self._make_q0_prior(use_sn_q0)

    def _make_q0_prior(self, mode):
        if mode in (None, "none", False):
            return None
        if mode == "corrected_combined":
            vals = []
            for r in self.sn_rows:
                if r["corrected"].strip().lower() == "yes":
                    vals.append((float(r["q0"]), float(r["q0_sigma"])))
            weights = np.array([1.0 / s ** 2 for _, s in vals])
            mean = float(np.sum([v * w for (v, _), w in zip(vals, weights)]) / np.sum(weights))
            sigma = float(np.sqrt(1.0 / np.sum(weights)))
            return mean, sigma, "SN age-corrected q0 combined summary"
        if mode == "uncorrected_combined":
            vals = []
            for r in self.sn_rows:
                if r["corrected"].strip().lower() == "no" and "BAO_CMB_" in r["source"]:
                    vals.append((float(r["q0"]), float(r["q0_sigma"])))
            weights = np.array([1.0 / s ** 2 for _, s in vals])
            mean = float(np.sum([v * w for (v, _), w in zip(vals, weights)]) / np.sum(weights))
            sigma = float(np.sqrt(1.0 / np.sum(weights)))
            return mean, sigma, "SN uncorrected q0 combined summary"
        # Match by source name.
        for r in self.sn_rows:
            if r["source"] == mode:
                return float(r["q0"]), float(r["q0_sigma"]), mode
        raise ValueError(f"Unknown q0-prior mode: {mode}")

    def chi2_bao(self, params):
        e = params.e_func()
        H0 = params.H0
        rd = params.rd
        chi2 = 0.0

        # Singleton isotropic rows.
        for r in self.bao_rows:
            if r["observable"] == "DV_over_rd":
                pred = bao_observables(float(r["z"]), H0, rd, e)["DV_over_rd"]
                chi2 += ((pred - float(r["value"])) / float(r["sigma"])) ** 2

        # Anisotropic DM-DH 2x2 blocks.
        groups = sorted({r["corr_group"] for r in self.bao_rows if r["corr_group"]})
        for group in groups:
            rows = [r for r in self.bao_rows if r["corr_group"] == group]
            rows_by_obs = {r["observable"]: r for r in rows}
            z = float(rows[0]["z"])
            pred = bao_observables(z, H0, rd, e)
            obs = np.array([
                float(rows_by_obs["DM_over_rd"]["value"]),
                float(rows_by_obs["DH_over_rd"]["value"]),
            ])
            pp = np.array([pred["DM_over_rd"], pred["DH_over_rd"]])
            s_dm = float(rows_by_obs["DM_over_rd"]["sigma"])
            s_dh = float(rows_by_obs["DH_over_rd"]["sigma"])
            rho = self.corr[group]
            # Analytic 2x2 inverse avoids heavy linear-algebra calls in tight loops.
            d0 = float(pp[0] - obs[0])
            d1 = float(pp[1] - obs[1])
            cov00 = s_dm ** 2
            cov11 = s_dh ** 2
            cov01 = rho * s_dm * s_dh
            det = cov00 * cov11 - cov01 * cov01
            chi2 += float((cov11 * d0 * d0 - 2.0 * cov01 * d0 * d1 + cov00 * d1 * d1) / det)
        return chi2

    def chi2_cmb_proxy(self, params):
        if not self.use_cmb_proxy:
            return 0.0
        e = params.e_func()
        ell_a = acoustic_scale_proxy(self.cmb_proxy["z_star"], params.H0, params.rd, e)
        return ((ell_a - self.cmb_proxy["ell_A_target"]) / self.cmb_proxy["ell_A_sigma"]) ** 2

    def chi2_q0(self, params):
        if self.q0_prior is None:
            return 0.0
        mean, sigma, _ = self.q0_prior
        return ((params.q0() - mean) / sigma) ** 2

    def chi2_total(self, params):
        c = self.chi2_bao(params)
        c += self.chi2_cmb_proxy(params)
        c += self.chi2_q0(params)
        return float(c)

    def components(self, params):
        out = {
            "bao": float(self.chi2_bao(params)),
            "cmb_proxy": float(self.chi2_cmb_proxy(params)),
            "q0_summary": float(self.chi2_q0(params)),
        }
        out["total"] = sum(out.values())
        return out


def in_bounds(values, names, bounds):
    for v, name in zip(values, names):
        lo, hi = bounds[name]
        if not (lo <= v <= hi):
            return False
    return True


def tip_from_array(x):
    return TipParams(n0=x[0], zt=x[1], beta=x[2], omega_m=x[3], H0=x[4], rd=x[5])


def lcdm_from_array(x):
    return LCDMParams(omega_m=x[0], H0=x[1], rd=x[2])


def tip_log_prior(x):
    names = ["n0", "zt", "beta", "omega_m", "H0", "rd"]
    return 0.0 if in_bounds(x, names, TIP_BOUNDS) else -math.inf


def lcdm_log_prior(x):
    names = ["omega_m", "H0", "rd"]
    return 0.0 if in_bounds(x, names, LCDM_BOUNDS) else -math.inf


def tip_log_posterior(x, like):
    lp = tip_log_prior(x)
    if not math.isfinite(lp):
        return -math.inf
    return lp - 0.5 * like.chi2_total(tip_from_array(x))


def lcdm_log_posterior(x, like):
    lp = lcdm_log_prior(x)
    if not math.isfinite(lp):
        return -math.inf
    return lp - 0.5 * like.chi2_total(lcdm_from_array(x))
