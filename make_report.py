#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
OUT = RESULTS / "transition_clock_benchmark_report.pdf"
TOP_OUT = Path("/mnt/data/Transition_Clock_Benchmark_Report.pdf")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=3):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 10:
        return f"{x:.3f}"
    return f"{x:.{nd}f}"


def small_table(data, col_widths=None, font_size=8, header=True):
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(7.5 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    with open(RESULTS / "benchmark_summary.json") as f:
        summary = json.load(f)
    pred = read_csv(RESULTS / "predictions_table.csv")
    bao = read_csv(RESULTS / "bao_residuals.csv")
    post = read_csv(RESULTS / "tip_posterior_summary.csv")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=10.5))
    styles.add(ParagraphStyle(name="Note", parent=styles["BodyText"], fontSize=9, leading=11, backColor=colors.HexColor("#f7f7f7"), borderColor=colors.HexColor("#bbbbbb"), borderWidth=0.25, borderPadding=6))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.65 * inch)
    story = []

    story.append(Paragraph("Transition-Clock Cosmology Benchmark Pipeline", styles["TitleCenter"]))
    story.append(Paragraph("Non-Boltzmann background, BAO, CMB-proxy, SN-summary, growth, and redshift-drift results", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Scope", styles["Heading2"]))
    story.append(Paragraph(
        "This report implements the requested first numerical stage while leaving Boltzmann/CLASS/CAMB work for later. It uses the transition-clock background equations, the DESI DR2 BAO benchmark table transcribed from the manuscript, a broad CMB acoustic-scale proxy, and optional supernova q0 summaries. The outputs are reproducible code, data tables, plots, and model-comparison diagnostics.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "Important limitation: this is not the final public likelihood paper. The DESI covariance is approximated as independent redshift blocks with the listed DM-DH correlations. The CMB term is only pi D_M(z*)/r_d with a deliberately broad uncertainty, not a Planck likelihood. The SN input is q0 summary information only, not a light-curve-level SN likelihood.",
        styles["Note"],
    ))

    story.append(Paragraph("Executive result", styles["Heading2"]))
    story.append(Paragraph(
        "The first-pass transition-clock branch fits the DESI BAO distance ratios reasonably at the benchmark level, but it does not yet pass the broad CMB acoustic-distance proxy: ell_A is 284.6 versus the target 301.0. The LCDM baseline fits the BAO+CMB-proxy benchmark much better, as expected. When the age-corrected supernova q0 summary is imposed, LCDM is heavily penalized because its best-fit q0 remains strongly negative; the transition-clock model is penalized less, but still does not hit the corrected q0 target. The safe paper-language conclusion is that this benchmark is a useful reproducible diagnostic, not an observational validation yet.",
        styles["BodyText"],
    ))

    tip = summary["tip"]
    lcdm = summary["lcdm"]
    bf_rows = [
        ["Parameter", "Transition clock", "LCDM baseline"],
        ["n0", fmt(tip["best_fit"].get("n0", "-")), "-"],
        ["zt", fmt(tip["best_fit"].get("zt", "-")), "-"],
        ["beta", fmt(tip["best_fit"].get("beta", "-")), "-"],
        ["Omega_m", fmt(tip["best_fit"]["omega_m"]), fmt(lcdm["best_fit"]["omega_m"])],
        ["H0 [km/s/Mpc]", fmt(tip["best_fit"]["H0"]), fmt(lcdm["best_fit"]["H0"])],
        ["r_d [Mpc]", fmt(tip["best_fit"]["rd"]), fmt(lcdm["best_fit"]["rd"])],
        ["q0", fmt(tip["q0"]), fmt(lcdm["q0"])],
        ["ell_A proxy", fmt(tip["ell_A_proxy"]), fmt(lcdm["ell_A_proxy"])],
    ]
    story.append(small_table(bf_rows, col_widths=[2.1 * inch, 2.2 * inch, 2.2 * inch], font_size=9))
    story.append(Spacer(1, 0.15 * inch))

    comp_rows = [
        ["Run", "Model", "chi2_BAO", "chi2_CMBproxy", "chi2_q0", "chi2_total", "AIC", "BIC"],
        ["BAO+CMBproxy", "TIP", fmt(tip["chi2_bao_cmb"]["bao"]), fmt(tip["chi2_bao_cmb"]["cmb_proxy"]), "-", fmt(tip["chi2_bao_cmb"]["total"]), fmt(tip["AIC_bao_cmb"]), fmt(tip["BIC_bao_cmb"])],
        ["BAO+CMBproxy", "LCDM", fmt(lcdm["chi2_bao_cmb"]["bao"]), fmt(lcdm["chi2_bao_cmb"]["cmb_proxy"]), "-", fmt(lcdm["chi2_bao_cmb"]["total"]), fmt(lcdm["AIC_bao_cmb"]), fmt(lcdm["BIC_bao_cmb"])],
        ["+ age-corrected q0", "TIP", fmt(tip["chi2_with_sn_age_corrected_q0"]["bao"]), fmt(tip["chi2_with_sn_age_corrected_q0"]["cmb_proxy"]), fmt(tip["chi2_with_sn_age_corrected_q0"]["q0_summary"]), fmt(tip["chi2_with_sn_age_corrected_q0"]["total"]), fmt(tip["AIC_with_age_q0"]), fmt(tip["BIC_with_age_q0"])],
        ["+ age-corrected q0", "LCDM", fmt(lcdm["chi2_with_sn_age_corrected_q0"]["bao"]), fmt(lcdm["chi2_with_sn_age_corrected_q0"]["cmb_proxy"]), fmt(lcdm["chi2_with_sn_age_corrected_q0"]["q0_summary"]), fmt(lcdm["chi2_with_sn_age_corrected_q0"]["total"]), fmt(lcdm["AIC_with_age_q0"]), fmt(lcdm["BIC_with_age_q0"])],
    ]
    story.append(Paragraph("Model-comparison diagnostics", styles["Heading2"]))
    story.append(small_table(comp_rows, col_widths=[1.35 * inch, 0.75 * inch, 0.85 * inch, 0.95 * inch, 0.75 * inch, 0.85 * inch, 0.6 * inch, 0.6 * inch], font_size=7.7))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Interpretation: on the BAO+CMB-proxy benchmark, LCDM has much lower chi2, AIC, and BIC. The transition-clock branch should not be presented as observationally viable yet. The useful result is that the benchmark has identified the next target: improve the transition function or parameter priors so that BAO and the CMB acoustic distance are both satisfied before investing in Boltzmann code.", styles["Small"]))

    story.append(PageBreak())
    story.append(Paragraph("Primary diagnostic plots", styles["Heading2"]))
    for fn, caption in [
        ("E_of_z.png", "Expansion history H(z)/H0."),
        ("q_of_z.png", "Acceleration history q(z)."),
        ("n_of_z.png", "Clock-index transition n(z)."),
        ("redshift_drift.png", "Redshift-drift velocity shift over 20 years."),
        ("fsigma8.png", "Growth prediction f sigma8(z) using GR-like growth on the benchmark background."),
        ("tip_corner.png", "Short local diagnostic chain for transition-clock parameters. Not a publication-grade posterior."),
    ]:
        img_path = PLOTS / fn
        if img_path.exists():
            story.append(KeepTogether([Image(str(img_path), width=6.45 * inch, height=4.2 * inch), Paragraph(caption, styles["Small"]), Spacer(1, 0.12 * inch)]))

    story.append(PageBreak())
    story.append(Paragraph("BAO residuals", styles["Heading2"]))
    bao_data = [["Tracer", "z", "Obs", "Data", "TIP", "TIP pull", "LCDM", "LCDM pull"]]
    for r in bao:
        bao_data.append([r["tracer"], fmt(r["z"], 3), r["observable"].replace("_over_", "/"), fmt(r["value"]), fmt(r["tip"]), fmt(r["tip_pull"]), fmt(r["lcdm"]), fmt(r["lcdm_pull"])])
    story.append(small_table(bao_data, col_widths=[0.8 * inch, 0.45 * inch, 0.8 * inch, 0.65 * inch, 0.7 * inch, 0.6 * inch, 0.7 * inch, 0.65 * inch], font_size=6.9))

    story.append(PageBreak())
    story.append(Paragraph("Forecast-style derived predictions", styles["Heading2"]))
    pred_data = [["Model", "z", "E", "q", "n", "drift cm/s", "D", "f", "f sigma8", "EG"]]
    for r in pred:
        pred_data.append([r["model"], fmt(r["z"], 2), fmt(r["E"]), fmt(r["q"]), fmt(r["n"]), fmt(r["drift_cm_s_20yr"]), fmt(r["D_growth"]), fmt(r["f_growth"]), fmt(r["fsigma8"]), fmt(r["EG"])] )
    story.append(small_table(pred_data, col_widths=[0.65 * inch, 0.35 * inch, 0.6 * inch, 0.55 * inch, 0.55 * inch, 0.85 * inch, 0.55 * inch, 0.55 * inch, 0.75 * inch, 0.55 * inch], font_size=7.1))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Posterior-diagnostic summary", styles["Heading2"]))
    post_data = [["Parameter", "16%", "median", "84%", "-", "+"]]
    for r in post:
        post_data.append([r["parameter"], fmt(r["p16"]), fmt(r["median"]), fmt(r["p84"]), fmt(r["minus"]), fmt(r["plus"])] )
    story.append(small_table(post_data, col_widths=[1.0 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch, 0.7 * inch], font_size=8))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Recommended next actions", styles["Heading2"]))
    next_text = "1. Do not put this into the paper as a claimed successful fit. Put it in as a reproducible benchmark and failure/target diagnostic. 2. Replace the broad CMB proxy with a real CMB distance prior only if Boltzmann code is still deferred; otherwise go straight to CLASS/CAMB later. 3. Replace the DESI block-diagonal approximation with official covariance matrices when web/data access is available. 4. Try a more flexible n(a), or a separate early normalization, because the current simple logistic branch fits BAO but misses the acoustic scale. 5. Keep the code and report in GitHub/Zenodo as supplementary material once pushed."
    story.append(Paragraph(next_text, styles["BodyText"]))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    TOP_OUT.write_bytes(OUT.read_bytes())
    print(str(OUT))
    print(str(TOP_OUT))


if __name__ == "__main__":
    main()
