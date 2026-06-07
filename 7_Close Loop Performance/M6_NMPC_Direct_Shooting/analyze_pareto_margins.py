#!/usr/bin/env python3
"""
analyze_pareto_margins.py
=========================

Fair Pareto comparison (R1.4 / plan item P0.5) between RAMC and the
static-margin baseline. Deliverables, matching the P0.5 specification:

  1. Equal-energy comparison: for each RAMC point, find the margin with the
     closest total energy (m_E = argmin_m |E_margin(m) - E_RAMC|), then
     compare CDH, peak violation, and CVaR_0.9.
  2. Equal-peak comparison: for each RAMC point, find the margin with the
     closest peak violation, then compare energy and CDH.
  3. Equal-CDH comparison: for each RAMC point, find the margin with the
     closest cold degree-hours, then compare energy and peak (and CVaR).
     [Added to satisfy the priority-table "minimum acceptable implementation":
      compare at equal energy, equal peak violation, AND equal CDH.]
  4. Domination summary: for each RAMC point, in each (energy, comfort-metric)
     plane, report whether any static margin dominates it (uses no more
     energy AND is no worse on the comfort metric, with at least one strict).
     This is the explicit, computed answer to the P0.5 question
     "is RAMC non-dominated for some metric/scenario?".
  5. Pareto-front figure in the (energy, CVaR_0.9) plane, per scenario.

Data sources:
  - Margin grid:  results_NMPC_margin_grid_*/matrix/all_results.json
  - RAMC + A0:    results_NMPC_ablation_*/matrix/all_results.json
    A0 Fidelity (margin 0.0) is read from the ablation run and used as the
    m=0.0 point of the margin curve (verified bitwise-identical to a
    comfort_margin_C=0.0 run), so the curve starts at the true no-margin
    baseline. Any RAMC_lambda_* models present in the ablation directory are
    auto-detected and compared.

Outputs (written to <margin-dir>/pareto_analysis/ by default):
  - equal_energy_comparison.csv
  - equal_peak_comparison.csv
  - equal_cdh_comparison.csv
  - domination_summary.csv
  - pareto_front.png / pareto_front.pdf

Usage (run from the M6_NMPC_Direct_Shooting directory):
  python analyze_pareto_margins.py \
      --margin-dir   results_NMPC_margin_grid_20260520_104342 \
      --ablation-dir results_NMPC_ablation_20260519_114954
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    "total_energy_kWh", "deg_hours_cold_occ",
    "cvar90_cold_occ_C", "cvar95_cold_occ_C",
    "peak_cold_violation_occ_C", "hours_cold_outside_occ",
]

# Comfort metrics for the domination check: curve-key -> (ramc-key, label).
# All are "lower is better", as is energy.
COMFORT_PLANES = [
    ("CDH", "deg_hours_cold_occ", "CDH"),
    ("peak", "peak_cold_violation_occ_C", "peak"),
    ("cvar90", "cvar90_cold_occ_C", "CVaR90"),
]


def load_results(results_dir: Path):
    """Load all_results.json -> list of successful experiment dicts."""
    path = results_dir / "matrix" / "all_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    with open(path) as f:
        data = json.load(f)
    return [r for r in data["results"] if r.get("success")]


def aggregate(results):
    """Return {(model, scenario): {metric: {mean, std, vals, seeds}}}."""
    groups = defaultdict(list)
    for r in results:
        groups[(r["model"], r["scenario"])].append(r)
    agg = {}
    for key, exps in groups.items():
        exps = sorted(exps, key=lambda e: e["seed"])
        d = {}
        for m in METRICS:
            vals = [e["metrics"][m] for e in exps]
            d[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "vals": vals,
                "seeds": [e["seed"] for e in exps],
            }
        agg[key] = d
    return agg


def margin_value(model_name: str) -> float:
    """'Fidelity_margin_0.3' -> 0.3"""
    return float(model_name.split("_margin_")[1])


def lambda_str(ramc_name: str) -> str:
    """'RAMC_lambda_0.0015_rollout' -> '0.0015'"""
    return ramc_name.split("_lambda_")[1].split("_")[0]


def main():
    parser = argparse.ArgumentParser(description="Fair Pareto margin analysis (P0.5 / R1.4)")
    parser.add_argument("--margin-dir", required=True,
                        help="results_NMPC_margin_grid_* directory")
    parser.add_argument("--ablation-dir", required=True,
                        help="results_NMPC_ablation_* directory (RAMC + A0)")
    parser.add_argument("--out", default=None,
                        help="output dir (default: <margin-dir>/pareto_analysis)")
    args = parser.parse_args()

    margin_dir = Path(args.margin_dir)
    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out) if args.out else (margin_dir / "pareto_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    margin_agg = aggregate(load_results(margin_dir))
    ablation_agg = aggregate(load_results(ablation_dir))

    scenarios = sorted(set(s for (_, s) in margin_agg.keys()))

    margin_models = sorted(
        set(m for (m, _) in margin_agg.keys() if "_margin_" in m),
        key=margin_value,
    )
    A0_NAME = "Fidelity_Baseline_rollout"
    ramc_models = sorted(
        set(m for (m, _) in ablation_agg.keys() if m.startswith("RAMC_lambda_"))
    )

    print(f"Margin levels found: {[margin_value(m) for m in margin_models]}")
    print(f"RAMC models found:   {ramc_models}")
    if A0_NAME not in [m for (m, _) in ablation_agg.keys()]:
        print(f"WARNING: {A0_NAME} not found in ablation dir; "
              f"margin curve will not include the m=0.0 point.")

    def margin_curve(scenario):
        """Margin operating points for a scenario, incl. m=0.0 from A0."""
        pts = []
        if (A0_NAME, scenario) in ablation_agg:
            a0 = ablation_agg[(A0_NAME, scenario)]
            pts.append({
                "margin": 0.0, "model": "Fidelity_margin_0.0",
                "E": a0["total_energy_kWh"]["mean"],
                "CDH": a0["deg_hours_cold_occ"]["mean"],
                "peak": a0["peak_cold_violation_occ_C"]["mean"],
                "cvar90": a0["cvar90_cold_occ_C"]["mean"],
            })
        for mm in margin_models:
            if (mm, scenario) not in margin_agg:
                continue
            d = margin_agg[(mm, scenario)]
            pts.append({
                "margin": margin_value(mm), "model": mm,
                "E": d["total_energy_kWh"]["mean"],
                "CDH": d["deg_hours_cold_occ"]["mean"],
                "peak": d["peak_cold_violation_occ_C"]["mean"],
                "cvar90": d["cvar90_cold_occ_C"]["mean"],
            })
        pts.sort(key=lambda p: p["margin"])
        return pts

    # -- Comparison tables --
    eq_energy_rows, eq_peak_rows, eq_cdh_rows, domination_rows = [], [], [], []

    for scenario in scenarios:
        curve = margin_curve(scenario)
        if not curve:
            continue
        for ramc in ramc_models:
            if (ramc, scenario) not in ablation_agg:
                continue
            rd = ablation_agg[(ramc, scenario)]
            E_r = rd["total_energy_kWh"]["mean"]
            CDH_r = rd["deg_hours_cold_occ"]["mean"]
            peak_r = rd["peak_cold_violation_occ_C"]["mean"]
            cvar_r = rd["cvar90_cold_occ_C"]["mean"]
            ramc_vals = {"CDH": CDH_r, "peak": peak_r, "cvar90": cvar_r}

            # -- Equal-energy --
            mE = min(curve, key=lambda p: abs(p["E"] - E_r))
            edge_E = (mE["margin"] == curve[-1]["margin"] and E_r > mE["E"]) \
                or (mE["margin"] == curve[0]["margin"] and E_r < mE["E"])
            eq_energy_rows.append({
                "scenario": scenario, "ramc": ramc, "ramc_lambda": lambda_str(ramc),
                "ramc_E": E_r, "matched_margin_C": mE["margin"], "margin_E": mE["E"],
                "energy_gap_kWh": E_r - mE["E"], "matched_at_grid_edge": edge_E,
                "ramc_CDH": CDH_r, "margin_CDH": mE["CDH"], "dCDH": CDH_r - mE["CDH"],
                "ramc_peak": peak_r, "margin_peak": mE["peak"], "dpeak": peak_r - mE["peak"],
                "ramc_cvar90": cvar_r, "margin_cvar90": mE["cvar90"],
                "dcvar90": cvar_r - mE["cvar90"],
            })

            # -- Equal-peak --
            mP = min(curve, key=lambda p: abs(p["peak"] - peak_r))
            edge_P = (mP["margin"] == curve[-1]["margin"] and peak_r < mP["peak"]) \
                or (mP["margin"] == curve[0]["margin"] and peak_r > mP["peak"])
            eq_peak_rows.append({
                "scenario": scenario, "ramc": ramc, "ramc_lambda": lambda_str(ramc),
                "ramc_peak": peak_r, "matched_margin_C": mP["margin"],
                "margin_peak": mP["peak"], "peak_gap_C": peak_r - mP["peak"],
                "matched_at_grid_edge": edge_P,
                "ramc_E": E_r, "margin_E": mP["E"], "dE": E_r - mP["E"],
                "ramc_CDH": CDH_r, "margin_CDH": mP["CDH"], "dCDH": CDH_r - mP["CDH"],
            })

            # -- Equal-CDH (priority-table requirement) --
            mC = min(curve, key=lambda p: abs(p["CDH"] - CDH_r))
            edge_C = (mC["margin"] == curve[-1]["margin"] and CDH_r < mC["CDH"]) \
                or (mC["margin"] == curve[0]["margin"] and CDH_r > mC["CDH"])
            eq_cdh_rows.append({
                "scenario": scenario, "ramc": ramc, "ramc_lambda": lambda_str(ramc),
                "ramc_CDH": CDH_r, "matched_margin_C": mC["margin"],
                "margin_CDH": mC["CDH"], "cdh_gap": CDH_r - mC["CDH"],
                "matched_at_grid_edge": edge_C,
                "ramc_E": E_r, "margin_E": mC["E"], "dE": E_r - mC["E"],
                "ramc_peak": peak_r, "margin_peak": mC["peak"], "dpeak": peak_r - mC["peak"],
                "ramc_cvar90": cvar_r, "margin_cvar90": mC["cvar90"],
                "dcvar90": cvar_r - mC["cvar90"],
            })

            # -- Domination check across (energy, comfort) planes --
            for ckey, _, clabel in COMFORT_PLANES:
                rv = ramc_vals[ckey]
                dominators = []
                for mp in curve:
                    no_worse = (mp["E"] <= E_r) and (mp[ckey] <= rv)
                    strictly = (mp["E"] < E_r) or (mp[ckey] < rv)
                    if no_worse and strictly:
                        dominators.append(mp["margin"])
                domination_rows.append({
                    "scenario": scenario, "ramc": ramc, "ramc_lambda": lambda_str(ramc),
                    "plane": f"energy_vs_{clabel}",
                    "ramc_non_dominated": len(dominators) == 0,
                    "n_dominating_margins": len(dominators),
                    "dominating_margins_C": ";".join(f"{m:.1f}" for m in dominators),
                })

    pd.DataFrame(eq_energy_rows).to_csv(out_dir / "equal_energy_comparison.csv", index=False)
    pd.DataFrame(eq_peak_rows).to_csv(out_dir / "equal_peak_comparison.csv", index=False)
    pd.DataFrame(eq_cdh_rows).to_csv(out_dir / "equal_cdh_comparison.csv", index=False)
    pd.DataFrame(domination_rows).to_csv(out_dir / "domination_summary.csv", index=False)

    # -- Print readable summaries --
    print("\n" + "=" * 72)
    print("EQUAL-ENERGY COMPARISON  (RAMC vs the closest-energy static margin)")
    print("Negative deltas favour RAMC.")
    print("=" * 72)
    for row in eq_energy_rows:
        edge = "  [matched at grid edge -- RAMC energy outside margin range]" \
            if row["matched_at_grid_edge"] else ""
        print(f"\n  {row['scenario']} | RAMC lambda={row['ramc_lambda']}")
        print(f"    RAMC energy {row['ramc_E']:.0f} kWh ~ margin +{row['matched_margin_C']:.1f}C "
              f"(margin energy {row['margin_E']:.0f} kWh, gap {row['energy_gap_kWh']:+.0f}){edge}")
        for name, key, unit in [("CDH", "dCDH", "degC.h"),
                                ("peak", "dpeak", "degC"),
                                ("CVaR90", "dcvar90", "degC")]:
            d = row[key]
            who = "RAMC better" if d < 0 else "margin better"
            print(f"    d{name:<7s}= {d:+.4f} {unit:<6s} ({who})")

    print("\n" + "=" * 72)
    print("EQUAL-PEAK COMPARISON  (RAMC vs the closest-peak static margin)")
    print("Negative dE favours RAMC (less energy); negative dCDH favours RAMC.")
    print("=" * 72)
    for row in eq_peak_rows:
        edge = "  [matched at grid edge -- RAMC peak outside margin range]" \
            if row["matched_at_grid_edge"] else ""
        print(f"\n  {row['scenario']} | RAMC lambda={row['ramc_lambda']}")
        print(f"    RAMC peak {row['ramc_peak']:.2f}C ~ margin +{row['matched_margin_C']:.1f}C "
              f"(margin peak {row['margin_peak']:.2f}C, gap {row['peak_gap_C']:+.2f}){edge}")
        whoE = "RAMC less" if row["dE"] < 0 else "margin less"
        whoC = "RAMC better" if row["dCDH"] < 0 else "margin better"
        print(f"    dE   = {row['dE']:+.0f} kWh  ({whoE})")
        print(f"    dCDH = {row['dCDH']:+.4f} degC.h  ({whoC})")

    print("\n" + "=" * 72)
    print("EQUAL-CDH COMPARISON  (RAMC vs the closest-CDH static margin)")
    print("Negative dE favours RAMC (less energy); negative dpeak favours RAMC.")
    print("=" * 72)
    for row in eq_cdh_rows:
        edge = "  [matched at grid edge -- RAMC CDH outside margin range]" \
            if row["matched_at_grid_edge"] else ""
        print(f"\n  {row['scenario']} | RAMC lambda={row['ramc_lambda']}")
        print(f"    RAMC CDH {row['ramc_CDH']:.2f} ~ margin +{row['matched_margin_C']:.1f}C "
              f"(margin CDH {row['margin_CDH']:.2f}, gap {row['cdh_gap']:+.2f}){edge}")
        whoE = "RAMC less" if row["dE"] < 0 else "margin less"
        whoP = "RAMC better" if row["dpeak"] < 0 else "margin better"
        print(f"    dE    = {row['dE']:+.0f} kWh   ({whoE})")
        print(f"    dpeak = {row['dpeak']:+.4f} degC  ({whoP})")

    print("\n" + "=" * 72)
    print("DOMINATION SUMMARY  (does any static margin dominate RAMC?)")
    print("A margin dominates if it uses <= energy AND is <= (better/equal) on")
    print("the comfort metric, with at least one strict improvement.")
    print("=" * 72)
    for row in domination_rows:
        verdict = "RAMC NON-DOMINATED" if row["ramc_non_dominated"] \
            else f"dominated by margins {{{row['dominating_margins_C']}}}"
        print(f"  {row['scenario']:<16s} | lambda={row['ramc_lambda']:<7s} | "
              f"{row['plane']:<18s} : {verdict}")

    any_nd = [r for r in domination_rows if r["ramc_non_dominated"]]
    print("\n  " + "-" * 68)
    if any_nd:
        print(f"  RAMC is NON-DOMINATED in {len(any_nd)} of {len(domination_rows)} "
              f"(scenario x metric) planes:")
        for r in any_nd:
            print(f"    - {r['scenario']} | lambda={r['ramc_lambda']} | {r['plane']}")
    else:
        print(f"  RAMC is DOMINATED by some static margin in ALL "
              f"{len(domination_rows)} (scenario x metric) planes.")
    print("  " + "-" * 68)

    # -- Pareto-front figure (energy vs CVaR0.9) --
    ncols = len(scenarios)
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 5.2))
    if ncols == 1:
        axes = [axes]

    ramc_colors = {
        "RAMC_lambda_0.0001_rollout": "#ff7f0e",
        "RAMC_lambda_0.0005_rollout": "#9467bd",
        "RAMC_lambda_0.0015_rollout": "#d62728",
    }

    for ax, scenario in zip(axes, scenarios):
        curve = margin_curve(scenario)
        if curve:
            mEv = [p["E"] for p in curve]
            mCv = [p["cvar90"] for p in curve]
            ax.plot(mEv, mCv, "o-", color="#2ca02c", lw=1.6, ms=5, alpha=0.85,
                    label="Static margin", zorder=2)
            for p in curve:
                ax.annotate(f"+{p['margin']:.1f}", (p["E"], p["cvar90"]),
                            fontsize=6.5, ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points",
                            color="#2ca02c")
        for ramc in ramc_models:
            if (ramc, scenario) not in ablation_agg:
                continue
            rd = ablation_agg[(ramc, scenario)]
            E_r = rd["total_energy_kWh"]["mean"]
            cvar_r = rd["cvar90_cold_occ_C"]["mean"]
            ax.scatter([E_r], [cvar_r], s=150, marker="*",
                       color=ramc_colors.get(ramc, "#d62728"),
                       edgecolors="black", lw=0.6, zorder=4,
                       label=f"RAMC $\\lambda$={lambda_str(ramc)}")
        ax.set_xlabel("Total heating energy (kWh)", fontsize=11)
        ax.set_ylabel("CVaR$_{0.9}$ cold violation (\u00b0C)", fontsize=11)
        ax.set_title(scenario, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.suptitle("Fair Pareto comparison: RAMC vs static-margin baseline (P0.5 / R1.4)\n"
                 "lower-left is better", fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"pareto_front.{ext}", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved tables and figure to: {out_dir}")
    print("  - equal_energy_comparison.csv")
    print("  - equal_peak_comparison.csv")
    print("  - equal_cdh_comparison.csv")
    print("  - domination_summary.csv")
    print("  - pareto_front.png / .pdf")


if __name__ == "__main__":
    main()
