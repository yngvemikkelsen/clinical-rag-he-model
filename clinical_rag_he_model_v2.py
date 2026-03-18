"""
================================================================================
Early economic evaluation of retrieval-layer correction in clinical RAG:
a decision-uncertainty framework
================================================================================
Mikkelsen Y. Early economic evaluation of retrieval-layer correction in
clinical RAG: a decision-uncertainty framework. Value in Health. 2026 [in press].

Companion repositories:
  Benchmark study:        github.com/yngvemikkelsen/clinical-rag-benchmark
  clinical-embedding-fix: github.com/yngvemikkelsen/clinical-embedding-fix
  DiReCT alpha notebook:  see /notebooks/direct_alpha_experiment.ipynb

Model structure
---------------
Decision tree, two primary arms:
  A — No correction (baseline)
  B — Corpus-only ZCA whitening (primary intervention)

Whitening is modelled across two corpus branches:
  Heterogeneous corpus (p = p_degraded * p_heterogeneous):  ΔMRR = +0.221
  Homogeneous corpus  (p = p_degraded * (1-p_heterogeneous)): ΔMRR = -0.05

Surrogate link (α)
  Base case: α = 1.111 (DiReCT empirical, ClinicalBERT PDD-level)
  Conservative: α = 0.36 (Tao et al. 2020, CDSS transported estimate)
  PSA: Trunc N(1.111, 0.37) bounds [0.36, 2.54]

Perspective: Norwegian healthcare system (NoMA reference case)
Discount rate: 4% (5-year horizon); 3.0% and 3.5% sensitivity available
Currency: EUR (NOK/EUR = 11.50)

Usage
-----
Run in any Python 3.10+ environment. No GPU required.
    pip install numpy pandas matplotlib scipy

For Colab: runtime → change runtime type → CPU is sufficient.

Outputs
-------
  - Base-case results table (printed)
  - Per-1000-query scale-independent results
  - Threshold analyses (α*, harm*, adoption*)
  - PSA cost-effectiveness plane and CEAC
  - Tornado diagram
  - Scenario table

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.stats import truncnorm, beta as beta_dist, gamma as gamma_dist
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────

N_SIM      = 10_000     # PSA iterations
HORIZON    = 5          # years
DISC_RATE  = 0.04       # NoMA reference case (4%)
N_QUERIES  = 133_000    # annual query volume — medium hospital base case
WTP        = 60_000     # EUR/QALY (Norwegian reference)
EUR_PER_NOK = 1/11.50

# ── Discount factor ───────────────────────────────────────────────────────────

def discount_factor(r=DISC_RATE, n=HORIZON):
    return sum(1/(1+r)**t for t in range(1, n+1))

D = discount_factor()

# ── Base-case parameters ──────────────────────────────────────────────────────

PARAMS = {
    # Effectiveness — whitening
    "delta_mrr_heterogeneous":  0.221,   # ΔMRR@10, degraded + heterogeneous corpus
    "delta_mrr_homogeneous":   -0.050,   # ΔMRR@10, degraded + homogeneous corpus (harm)

    # Deployment prevalence
    "p_degraded":               0.45,    # P(model shows degraded geometry)
    "p_heterogeneous":          0.67,    # P(corpus is heterogeneous | degraded)

    # Surrogate link — DiReCT empirical base case
    "alpha":                    1.111,   # Δ diagnosis-label retrieval acc per unit ΔMRR

    # Clinical chain
    "p_adopt":                  0.60,    # clinician adoption (constructed)
    "p_ae":                     0.12,    # P(ADE | diagnostic error)
    "qaly_loss":                0.06,    # QALY loss per ADE (constructed)

    # Costs
    "cost_ae_system":           12_000,  # EUR — healthcare system per ADE
    "cost_ae_societal":         66_564,  # EUR — NPE avg payout (societal perspective)
    "cost_w_impl":              800,     # EUR — whitening one-time implementation
    "cost_w_annual":            150,     # EUR — annual corpus refit

    # Deployment scale
    "n_queries":                N_QUERIES,
}


# ── Core model functions ───────────────────────────────────────────────────────

def annual_ades_averted(p, n=None):
    """
    Net annual ADEs averted by whitening.

    Parameters
    ----------
    p : dict   model parameters
    n : int    annual query volume (overrides p["n_queries"] if given)

    Returns
    -------
    float      net ADEs averted per year (positive = benefit)
    """
    n = n or p["n_queries"]
    p1 = p["p_degraded"] * p["p_heterogeneous"]        # heterogeneous branch
    p2 = p["p_degraded"] * (1 - p["p_heterogeneous"])  # homogeneous branch (harm)

    benefit = p["alpha"] * p["delta_mrr_heterogeneous"] * p["p_adopt"] * p1
    harm    = p["alpha"] * p["delta_mrr_homogeneous"]   * p["p_adopt"] * p2

    return (benefit + harm) * p["p_ae"] * n


def technology_cost_5yr(p, d=D):
    """5-year discounted technology cost of whitening."""
    return p["cost_w_impl"] + p["cost_w_annual"] * d


def incremental_results(p, n=None, perspective="system", d=D):
    """
    Compute incremental cost and QALY for whitening vs no correction.

    Returns dict with keys: ade_yr, saving_5yr, ic, iq, icer
    """
    n = n or p["n_queries"]
    cost_ae = p["cost_ae_system"] if perspective == "system" else p["cost_ae_societal"]

    ade_yr  = annual_ades_averted(p, n)
    saving  = ade_yr * cost_ae * d
    tech    = technology_cost_5yr(p, d)
    ic      = tech - saving
    iq      = ade_yr * p["qaly_loss"] * d
    icer    = ic / iq if iq != 0 else np.nan

    return {"ade_yr": ade_yr, "saving_5yr": saving, "ic": ic, "iq": iq, "icer": icer}


def classify(ic, iq):
    """Four-quadrant ICER classification."""
    if ic < 0 and iq > 0: return "Dominant"
    if ic > 0 and iq > 0: return "ICER"
    if ic > 0 and iq < 0: return "Dominated"
    return "Extended dominance"


# ── Base-case results ──────────────────────────────────────────────────────────

print("=" * 65)
print("BASE-CASE RESULTS (α=1.111, N=133,000/yr, system perspective)")
print("=" * 65)

r = incremental_results(PARAMS)
print(f"  Net ADEs averted per year:    {r['ade_yr']:>10.2f}")
print(f"  5yr healthcare savings:       €{r['saving_5yr']:>12,.0f}")
print(f"  Technology cost (5yr):        €{technology_cost_5yr(PARAMS):>12,.0f}")
print(f"  Incremental cost (ic_BvA):    €{r['ic']:>12,.0f}")
print(f"  QALY gain (5yr):              {r['iq']:>10.2f}")
print(f"  Classification:               {classify(r['ic'], r['iq'])}")
print()

# Conservative scenario (Tao et al. α)
p_cons = {**PARAMS, "alpha": 0.36}
r_cons = incremental_results(p_cons)
print("Conservative scenario (α=0.36, Tao et al. 2020):")
print(f"  Net ADEs averted per year:    {r_cons['ade_yr']:>10.2f}")
print(f"  5yr healthcare savings:       €{r_cons['saving_5yr']:>12,.0f}")
print(f"  Incremental cost:             €{r_cons['ic']:>12,.0f}")
print(f"  QALY gain (5yr):              {r_cons['iq']:>10.2f}")
print(f"  Classification:               {classify(r_cons['ic'], r_cons['iq'])}")


# ── Scale-independent results (per 1,000 queries/year) ───────────────────────

print()
print("=" * 65)
print("SCALE-INDEPENDENT RESULTS (per 1,000 queries/year, 5yr)")
print("=" * 65)

headers = ["Scenario", "ADEs/yr", "Savings 5yr (€)", "QALYs 5yr"]
rows = []
for label, alpha in [("Base (α=1.111)", 1.111), ("Conservative (α=0.36)", 0.36),
                     ("Stress-test (α=0.15)", 0.15)]:
    p_ = {**PARAMS, "alpha": alpha, "n_queries": 1_000}
    r_ = incremental_results(p_)
    rows.append([label, f"{r_['ade_yr']:.3f}", f"€{r_['saving_5yr']:,.0f}",
                 f"{r_['iq']:.4f}"])
df_scale = pd.DataFrame(rows, columns=headers)
print(df_scale.to_string(index=False))


# ── Breakeven / minimum N* ────────────────────────────────────────────────────

print()
print("=" * 65)
print("MINIMUM ANNUAL QUERY VOLUME N* (whitening recovers €800 cost)")
print("=" * 65)
print(f"  {'α':<22} {'Harm=-0.05':>12}  {'Harm=-0.20':>12}  {'Harm=-0.45':>12}")
print(f"  {'':-<60}")

for alpha in [0.05, 0.15, 0.36, 1.111, 2.54]:
    row = [f"α={alpha}"]
    for harm in [-0.05, -0.20, -0.45]:
        p_ = {**PARAMS, "alpha": alpha, "delta_mrr_homogeneous": harm, "n_queries": 1}
        p1 = p_["p_degraded"] * p_["p_heterogeneous"]
        p2 = p_["p_degraded"] * (1 - p_["p_heterogeneous"])
        save_per_q = (alpha*(p_["delta_mrr_heterogeneous"]*p1 + harm*p2)
                      * p_["p_adopt"] * p_["p_ae"] * p_["cost_ae_system"] * D)
        tech = technology_cost_5yr(PARAMS)
        n_star = tech / save_per_q if save_per_q > 0 else float("inf")
        row.append(f"{n_star:.0f}" if n_star < 1e6 else "∞")
    print(f"  {row[0]:<22} {row[1]:>12}  {row[2]:>12}  {row[3]:>12}")


# ── Threshold analyses ────────────────────────────────────────────────────────

print()
print("=" * 65)
print("THRESHOLD ANALYSES (N=133,000/yr, base-case α=1.111)")
print("=" * 65)

# α threshold (analytical)
p1 = PARAMS["p_degraded"] * PARAMS["p_heterogeneous"]
p2 = PARAMS["p_degraded"] * (1 - PARAMS["p_heterogeneous"])
tech = technology_cost_5yr(PARAMS)

A = ((PARAMS["delta_mrr_heterogeneous"]*p1 + PARAMS["delta_mrr_homogeneous"]*p2)
     * PARAMS["p_adopt"] * PARAMS["p_ae"] * PARAMS["n_queries"]
     * PARAMS["cost_ae_system"] * D)
alpha_thresh = tech / A

# harm threshold
benefit_per_unit_harm = PARAMS["alpha"] * p2 * PARAMS["p_adopt"] * PARAMS["p_ae"] * PARAMS["n_queries"] * PARAMS["cost_ae_system"] * D
benefit_heterogeneous = PARAMS["alpha"] * PARAMS["delta_mrr_heterogeneous"] * PARAMS["p_adopt"] * p1 * PARAMS["p_ae"] * PARAMS["n_queries"] * PARAMS["cost_ae_system"] * D
harm_thresh = (tech - benefit_heterogeneous) / benefit_per_unit_harm

# adoption threshold
adopt_coeff = PARAMS["alpha"] * (PARAMS["delta_mrr_heterogeneous"]*p1 + PARAMS["delta_mrr_homogeneous"]*p2) * PARAMS["p_ae"] * PARAMS["n_queries"] * PARAMS["cost_ae_system"] * D
adopt_thresh = tech / adopt_coeff

print(f"  Surrogate link α*:            {alpha_thresh:.5f}  (base={PARAMS['alpha']}, margin={PARAMS['alpha']-alpha_thresh:.3f})")
print(f"  Harm magnitude |ΔMRR|*:       {abs(harm_thresh):.3f}    (base=0.05, margin={abs(harm_thresh)-0.05:.3f})")
print(f"  Adoption p_adopt*:            {adopt_thresh:.5f}  (base={PARAMS['p_adopt']}, margin={PARAMS['p_adopt']-adopt_thresh:.3f})")


# ── Probabilistic Sensitivity Analysis (PSA) ──────────────────────────────────

print()
print("=" * 65)
print(f"PROBABILISTIC SENSITIVITY ANALYSIS ({N_SIM:,} iterations)")
print("=" * 65)

def sample_params(n=N_SIM):
    """Sample PSA parameter distributions."""

    def trunc_normal(mu, sigma, lo, hi, size):
        a, b = (lo - mu)/sigma, (hi - mu)/sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)

    return {
        # Surrogate link — DiReCT empirical, bounds = [0.36, 2.54]
        "alpha":                    trunc_normal(1.111, 0.37, 0.36, 2.54, n),
        # Harm branch — bounds = [-0.20, 0.0]
        "delta_mrr_homogeneous":    trunc_normal(-0.05, 0.04, -0.20, 0.00, n),
        # Degradation/corpus prevalence
        "p_degraded":               beta_dist.rvs(5, 6, size=n),
        "p_heterogeneous":          beta_dist.rvs(8, 4, size=n),
        # Clinical chain
        "p_adopt":                  beta_dist.rvs(9, 6, size=n),
        "p_ae":                     beta_dist.rvs(6, 44, size=n),
        "qaly_loss":                gamma_dist.rvs(1.2, scale=0.05, size=n),
        # Costs
        "cost_ae_system":           gamma_dist.rvs(2.0, scale=6_000, size=n),
        "cost_w_impl":              gamma_dist.rvs(2.0, scale=400, size=n),
        "cost_w_annual":            gamma_dist.rvs(1.5, scale=100, size=n),
    }

samples = sample_params()

# Vectorised model evaluation
p1_s = samples["p_degraded"] * samples["p_heterogeneous"]
p2_s = samples["p_degraded"] * (1 - samples["p_heterogeneous"])

benefit_s = samples["alpha"] * PARAMS["delta_mrr_heterogeneous"] * samples["p_adopt"] * p1_s
harm_s    = samples["alpha"] * samples["delta_mrr_homogeneous"]   * samples["p_adopt"] * p2_s
ade_s     = (benefit_s + harm_s) * samples["p_ae"] * PARAMS["n_queries"]

saving_s  = ade_s * samples["cost_ae_system"] * D
tech_s    = samples["cost_w_impl"] + samples["cost_w_annual"] * D
ic_s      = tech_s - saving_s
iq_s      = ade_s * samples["qaly_loss"] * D

dominant = np.mean((ic_s < 0) & (iq_s > 0))
print(f"  Dominant (cost-saving + QALY-gaining): {dominant*100:.1f}%")
print(f"  Mean ic_BvA:   €{np.mean(ic_s):>12,.0f}  (SD €{np.std(ic_s):,.0f})")
print(f"  Mean ΔQALY:    {np.mean(iq_s):>10.2f}  (SD {np.std(iq_s):.2f})")
print()
print("  NOTE: PSA excludes α=0 by construction (lower bound=0.36).")
print("  99.7% dominance is conditional on α being substantively positive.")
print("  It does not address P(α>0) in a live clinical workflow.")


# ── Discount rate sensitivity ─────────────────────────────────────────────────

print()
print("=" * 65)
print("DISCOUNT RATE SENSITIVITY (N=133,000/yr, α=1.111)")
print("=" * 65)
print(f"  {'Rate':<12} {'ic_BvA':>14}  {'ΔQALYs':>10}  {'Classification'}")
print(f"  {'':-<52}")
for r_disc, label in [(0.030, "3.0% (NICE)"), (0.035, "3.5% (ISPOR)"), (0.040, "4.0% (NoMA base)")]:
    d_ = discount_factor(r_disc)
    r_ = incremental_results(PARAMS, d=d_)
    print(f"  {label:<12} €{r_['ic']:>12,.0f}  {r_['iq']:>10.2f}  {classify(r_['ic'], r_['iq'])}")


# ── Scenario table ────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SCENARIO TABLE (system perspective unless noted)")
print("=" * 65)

scenarios = [
    ("Small clinic  (N=18k,   α=1.111)", {**PARAMS, "n_queries": 18_000}),
    ("Medium hosp   (N=133k,  α=1.111)", PARAMS),
    ("Large system  (N=1M,    α=1.111)", {**PARAMS, "n_queries": 1_000_000}),
    ("Conservative  (N=133k,  α=0.36)",  {**PARAMS, "alpha": 0.36}),
    ("Stress-test   (N=133k,  α=0.15)",  {**PARAMS, "alpha": 0.15}),
    ("Societal NPE  (N=133k,  α=1.111)", {**PARAMS, "cost_ae_system": 66_564}),
]

print(f"  {'Scenario':<38} {'ic_BvA':>14}  {'ΔQALYs':>8}  {'Class'}")
print(f"  {'':-<70}")
for label, p_ in scenarios:
    r_ = incremental_results(p_)
    print(f"  {label:<38} €{r_['ic']:>12,.0f}  {r_['iq']:>8.2f}  {classify(r_['ic'], r_['iq'])}")


# ── Figures ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "Early economic evaluation of ZCA whitening for clinical RAG\n"
    "Mikkelsen Y. Value in Health. 2026.",
    fontsize=13, fontweight="bold", y=0.98
)

BLUE   = "#2E4E7A"
ORANGE = "#C0392B"
GREY   = "#95A5A6"
GREEN  = "#27AE60"

# ── Panel A: Cost-effectiveness plane ─────────────────────────────────────────
ax = axes[0, 0]
# PSA cloud (subsample 2000 for speed)
idx = np.random.choice(N_SIM, 2000, replace=False)
sc = ax.scatter(iq_s[idx], ic_s[idx]/1e6,
                alpha=0.25, s=6, c=np.where(ic_s[idx] < 0, BLUE, GREY))
# Base case
r_bc = incremental_results(PARAMS)
ax.scatter(r_bc["iq"], r_bc["ic"]/1e6, color=ORANGE, s=100, zorder=5,
           marker="D", label="Base case (α=1.111)")
# Conservative
r_cv = incremental_results(p_cons)
ax.scatter(r_cv["iq"], r_cv["ic"]/1e6, color=GREEN, s=80, zorder=5,
           marker="s", label="Conservative (α=0.36)")
# WTP line
iq_range = np.linspace(0, max(iq_s)*1.1, 100)
ax.plot(iq_range, WTP * iq_range / 1e6, "k--", lw=1, alpha=0.5,
        label=f"WTP €{WTP:,}/QALY")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Incremental QALYs (5yr)")
ax.set_ylabel("Incremental Cost (€M, 5yr)")
ax.set_title("A. Cost-effectiveness plane (B vs A)")
ax.legend(fontsize=8)
patch_dom = mpatches.Patch(color=BLUE, alpha=0.5, label="Dominant (PSA)")
patch_non = mpatches.Patch(color=GREY, alpha=0.5, label="Non-dominant (PSA)")
ax.legend(handles=[patch_dom, patch_non,
                   plt.scatter([], [], color=ORANGE, marker="D", s=60),
                   plt.scatter([], [], color=GREEN, marker="s", s=50),
                   plt.Line2D([0], [0], color="k", ls="--", lw=1)],
          labels=["Dominant (PSA)", "Non-dominant (PSA)",
                  "Base case α=1.111", "Conservative α=0.36",
                  f"WTP €{WTP:,}/QALY"],
          fontsize=7.5, loc="upper left")

# ── Panel B: CEAC ─────────────────────────────────────────────────────────────
ax = axes[0, 1]
wtp_vals = np.linspace(0, 100_000, 200)
ceac = [np.mean((ic_s < wtp * iq_s)) for wtp in wtp_vals]
ax.plot(wtp_vals/1000, ceac, color=BLUE, lw=2)
ax.axvline(WTP/1000, color="k", ls="--", lw=1, label=f"WTP = €{WTP//1000}k/QALY")
ax.axhline(0.997, color=GREY, ls=":", lw=1, label="99.7% (PSA result)")
ax.set_xlabel("Willingness-to-pay threshold (€k/QALY)")
ax.set_ylabel("Probability cost-effective")
ax.set_title("B. Cost-effectiveness acceptability curve")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.text(5, 0.92,
        "CAUTION: PSA excludes α=0 by\nconstruction. Results are conditional\non α being substantively positive.",
        fontsize=7, color=ORANGE, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# ── Panel C: Tornado diagram ───────────────────────────────────────────────────
ax = axes[1, 0]

tornado_params = [
    ("Surrogate link α",         "alpha",                      0.36,   2.54),
    ("ΔMRR whitening",           "delta_mrr_heterogeneous",    0.157,  0.272),
    ("Harm magnitude |ΔMRR|",    "delta_mrr_homogeneous",      -0.20,  -0.01),
    ("P(model degraded)",        "p_degraded",                 0.20,   0.70),
    ("P(corpus heterogeneous)",  "p_heterogeneous",            0.45,   0.85),
    ("Cost per ADE (€)",         "cost_ae_system",             5_000,  30_000),
    ("Clinician adoption",       "p_adopt",                    0.30,   0.80),
    ("P(ADE | diag. error)",     "p_ae",                       0.06,   0.17),
    ("QALY loss per ADE",        "qaly_loss",                  0.03,   0.12),
    ("Whitening tech cost (€)",  "cost_w_impl",                300,    2_000),
]

ic_base = incremental_results(PARAMS)["ic"]
bars = []
for label, key, lo, hi in tornado_params:
    p_lo = {**PARAMS, key: lo}
    p_hi = {**PARAMS, key: hi}
    ic_lo = incremental_results(p_lo)["ic"]
    ic_hi = incremental_results(p_hi)["ic"]
    bars.append((label, min(ic_lo, ic_hi), max(ic_lo, ic_hi)))

bars.sort(key=lambda x: x[2]-x[1], reverse=True)
labels = [b[0] for b in bars]
lefts  = [b[1]/1e6 for b in bars]
widths = [(b[2]-b[1])/1e6 for b in bars]

y_pos = range(len(bars))
ax.barh(y_pos, widths, left=lefts, color=BLUE, alpha=0.75, height=0.6)
ax.axvline(ic_base/1e6, color="k", lw=1.2, ls="--", label="Base case")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Incremental cost (€M, 5yr)")
ax.set_title("C. Tornado diagram (one-way SA, B vs A)")
ax.legend(fontsize=8)

# ── Panel D: α sensitivity ────────────────────────────────────────────────────
ax = axes[1, 1]
alphas = np.linspace(0.05, 3.0, 200)
ics    = [incremental_results({**PARAMS, "alpha": a})["ic"]/1e6 for a in alphas]
iqs    = [incremental_results({**PARAMS, "alpha": a})["iq"] for a in alphas]

ax2 = ax.twinx()
ax.plot(alphas, ics, color=BLUE, lw=2, label="ic_BvA (€M)")
ax2.plot(alphas, iqs, color=GREEN, lw=2, ls="--", label="ΔQALYs")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(PARAMS["alpha"], color=ORANGE, lw=1.5, ls=":", label=f"Base α={PARAMS['alpha']}")
ax.axvline(0.36, color=GREY, lw=1.5, ls=":", label="Conservative α=0.36")
ax.fill_betweenx([min(ics)*1.05, 0], 0.36, 2.54,
                  alpha=0.08, color=BLUE, label="DiReCT 95% CI")
ax.set_xlabel("Surrogate link α")
ax.set_ylabel("Incremental cost (€M, 5yr)", color=BLUE)
ax2.set_ylabel("QALY gain (5yr)", color=GREEN)
ax.set_title("D. α sensitivity (N=133k, system perspective)")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7.5, loc="upper right")

plt.tight_layout()
plt.savefig("he_model_results_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved: he_model_results_v2.png")

print()
print("=" * 65)
print("Run complete.")
print("Cite as: Mikkelsen Y. Value in Health. 2026 [in press].")
print("Code: github.com/yngvemikkelsen/clinical-rag-he-model")
print("=" * 65)
