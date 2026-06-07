"""
Assignment 3 — λ-selection and statistical stability
======================================================

This script performs the PURE ANALYSIS portion of A3 (no new simulations).
It reads existing Phase 2 and Phase 3 data to produce:

  1. Utopia-point λ-selection analysis (from decomposed_evaluation_full.csv)
  2. Bootstrap 95% CIs for all paired differences (from all_results.json)
  3. Wilcoxon signed-rank tests as secondary evidence
  4. Revised Phase 3 model roster with explicit rationale
  5. Publication-ready figures: trade-off plot + forest plot

Run from 8_Assignments/ directory:
    python -m A3_lambda_selection.run_a3

Or from Spyder IPython console:
    %run A3_lambda_selection/run_a3.py

Outputs saved to: A3_lambda_selection/results/
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Imports from shared infrastructure ──────────────────────────
# Ensure shared/ is importable
THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    DECOMPOSED_EVAL_CSV,
    NMPC_ALL_RESULTS_JSON,
    A3_DIR,
    get_results_dir,
    PHASE3_SEEDS,
    COMFORT_BAND,
)
from shared.lambda_selection_rule import run_utopia_analysis
from shared.stats_utils import bootstrap_ci, wilcoxon_signed_rank


# ================================================================
# Part 1: Utopia-point λ-selection
# ================================================================

def part1_utopia_selection():
    """Run utopia-point analysis on Phase 2 decomposed evaluation."""
    print("=" * 65)
    print("  Part 1: Utopia-Point λ-Selection")
    print("=" * 65)

    result = run_utopia_analysis(DECOMPOSED_EVAL_CSV)

    print(f"\n  Primary criterion (total risk):")
    print(f"    Selected λ* = {result['selected_lambda_total_risk']:.1e}")
    print(f"    Model: {result['selected_model_total_risk']}")
    print(f"    Utopia distance: {result['utopia_distance_total_risk']:.4f}")

    print(f"\n  Secondary criterion (comfort risk only):")
    print(f"    Selected λ* = {result['selected_lambda_comfort_risk']:.1e}")
    print(f"    Model: {result['selected_model_comfort_risk']}")
    print(f"    Utopia distance: {result['utopia_distance_comfort_risk']:.4f}")

    # Print full table
    print(f"\n  {'Model':<45s} {'λ':>10s} {'Lfid':>10s} {'Rtotal':>10s} {'Dist':>8s} {'Sel':>5s}")
    print("  " + "-" * 90)
    for m in result["model_results"]:
        d_str = f"{m['utopia_distance']:.4f}" if m["utopia_distance"] is not None else "  excl"
        sel_str = "  <<<" if m["selected"] else ""
        print(f"  {m['model_name']:<45s} {m['lambda']:>10.1e} "
              f"{m['fidelity_loss']:.5f} {m['risk_total']:>10.3f} {d_str:>8s}{sel_str}")

    # Save
    out_dir = get_results_dir(A3_DIR)
    out_path = out_dir / "utopia_point_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {out_path.name}")

    return result


# ================================================================
# Part 2: Extract per-seed data and compute bootstrap CIs
# ================================================================

def load_per_seed_results() -> dict:
    """
    Load all_results.json and organise by (model, scenario) -> list of metric dicts.

    Returns dict like:
        {("Fidelity_Baseline_rollout", "nominal"): [
            {"seed": 42, "deg_hours_cold_occ": 11.29, ...},
            {"seed": 123, ...}, ...
        ]}
    """
    with open(NMPC_ALL_RESULTS_JSON) as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for exp in data["results"]:
        if not exp["success"]:
            continue
        key = (exp["model"], exp["scenario"])
        entry = {"seed": exp["seed"], **exp["metrics"]}
        grouped[key].append(entry)

    return dict(grouped)


def compute_paired_differences(grouped: dict) -> list:
    """
    For each RAMC model and scenario, compute per-seed paired differences
    against the Fidelity baseline, then bootstrap CIs.

    Returns list of result dicts.
    """
    # Identify baseline and RAMC models
    baseline_name = "Fidelity_Baseline_rollout"
    ramc_models = sorted(set(
        model for model, scenario in grouped.keys()
        if model != baseline_name
    ))
    scenarios = sorted(set(scenario for model, scenario in grouped.keys()))

    # Metrics to analyse
    metrics = [
        "deg_hours_cold_occ",
        "cvar90_cold_occ_C",
        "peak_cold_violation_occ_C",
        "total_energy_kWh",
    ]
    metric_labels = {
        "deg_hours_cold_occ": "Cold CDH",
        "cvar90_cold_occ_C": "CVaR90 cold (°C)",
        "peak_cold_violation_occ_C": "Peak cold (°C)",
        "total_energy_kWh": "Energy (kWh)",
    }

    all_paired = []

    for ramc_model in ramc_models:
        for scenario in scenarios:
            base_key = (baseline_name, scenario)
            ramc_key = (ramc_model, scenario)

            if base_key not in grouped or ramc_key not in grouped:
                continue

            base_runs = sorted(grouped[base_key], key=lambda x: x["seed"])
            ramc_runs = sorted(grouped[ramc_key], key=lambda x: x["seed"])

            # Match by seed
            base_by_seed = {r["seed"]: r for r in base_runs}
            ramc_by_seed = {r["seed"]: r for r in ramc_runs}
            common_seeds = sorted(set(base_by_seed) & set(ramc_by_seed))

            if len(common_seeds) < 2:
                continue

            for metric in metrics:
                base_vals = np.array([base_by_seed[s][metric] for s in common_seeds])
                ramc_vals = np.array([ramc_by_seed[s][metric] for s in common_seeds])
                deltas = ramc_vals - base_vals  # negative = RAMC better for cost metrics

                mean_d, ci_lo, ci_hi = bootstrap_ci(deltas, n_resamples=10_000)
                w_stat, w_p, n_eff = wilcoxon_signed_rank(deltas)

                all_paired.append({
                    "model": ramc_model,
                    "scenario": scenario,
                    "metric": metric,
                    "metric_label": metric_labels[metric],
                    "n_pairs": len(common_seeds),
                    "seeds": common_seeds,
                    "mean_delta": float(mean_d),
                    "std_delta": float(np.std(deltas, ddof=1)),
                    "median_delta": float(np.median(deltas)),
                    "ci_95_lower": float(ci_lo),
                    "ci_95_upper": float(ci_hi),
                    "ci_excludes_zero": not (ci_lo <= 0 <= ci_hi),
                    "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
                    "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
                    "sign_negative": int(np.sum(deltas < 0)),
                    "sign_positive": int(np.sum(deltas > 0)),
                    "per_seed_deltas": deltas.tolist(),
                })

    return all_paired


def part2_statistical_analysis():
    """Extract per-seed data, compute bootstrap CIs and Wilcoxon tests."""
    print("\n" + "=" * 65)
    print("  Part 2: Bootstrap CIs and Statistical Tests")
    print("=" * 65)

    grouped = load_per_seed_results()
    print(f"\n  Loaded {sum(len(v) for v in grouped.values())} experiment results")
    print(f"  Unique (model, scenario) groups: {len(grouped)}")

    paired = compute_paired_differences(grouped)

    # Print summary table for the primary model (RAMC λ=1.5e-3)
    print(f"\n  Paired differences: RAMC λ=1.5e-3 vs Fidelity Baseline")
    print(f"  (negative = RAMC better for comfort metrics, positive = more energy)")
    print(f"\n  {'Scenario':<18s} {'Metric':<20s} {'Δ mean':>9s} {'95% CI':>22s} {'Excl 0?':>8s} {'Wilc p':>8s}")
    print("  " + "-" * 87)

    for p in paired:
        if "0.0015" not in p["model"]:
            continue
        ci_str = f"[{p['ci_95_lower']:+.3f}, {p['ci_95_upper']:+.3f}]"
        excl = "YES" if p["ci_excludes_zero"] else "no"
        w_str = f"{p['wilcoxon_p']:.4f}" if p["wilcoxon_p"] is not None else "n/a"
        print(f"  {p['scenario']:<18s} {p['metric_label']:<20s} "
              f"{p['mean_delta']:>+9.3f} {ci_str:>22s} {excl:>8s} {w_str:>8s}")

    # Also print for λ=1e-4 and λ=5e-4
    for lam_str, label in [("0.0001", "1e-4"), ("0.0005", "5e-4")]:
        subset = [p for p in paired if lam_str in p["model"]]
        if subset:
            print(f"\n  Paired differences: RAMC λ={label} vs Fidelity Baseline")
            print(f"  {'Scenario':<18s} {'Metric':<20s} {'Δ mean':>9s} {'95% CI':>22s} {'Excl 0?':>8s}")
            print("  " + "-" * 79)
            for p in subset:
                ci_str = f"[{p['ci_95_lower']:+.3f}, {p['ci_95_upper']:+.3f}]"
                excl = "YES" if p["ci_excludes_zero"] else "no"
                print(f"  {p['scenario']:<18s} {p['metric_label']:<20s} "
                      f"{p['mean_delta']:>+9.3f} {ci_str:>22s} {excl:>8s}")

    # Save
    out_dir = get_results_dir(A3_DIR)
    out_path = out_dir / "paired_bootstrap_analysis.json"
    with open(out_path, "w") as f:
        json.dump(paired, f, indent=2)
    print(f"\n  Saved: {out_path.name}")

    return paired


# ================================================================
# Part 3: Pooled cross-scenario analysis
# ================================================================

def part3_pooled_analysis(paired: list):
    """
    Pool paired differences across all 3 scenarios for each model
    (n=15 per model) and compute pooled bootstrap CIs.
    """
    print("\n" + "=" * 65)
    print("  Part 3: Pooled Cross-Scenario Analysis (n=15)")
    print("=" * 65)

    models = sorted(set(p["model"] for p in paired))
    metrics = ["deg_hours_cold_occ", "cvar90_cold_occ_C",
               "peak_cold_violation_occ_C", "total_energy_kWh"]
    metric_labels = {
        "deg_hours_cold_occ": "Cold CDH",
        "cvar90_cold_occ_C": "CVaR90 cold",
        "peak_cold_violation_occ_C": "Peak cold",
        "total_energy_kWh": "Energy",
    }

    pooled_results = []

    for model in models:
        # Extract a short label
        if "0.0001" in model:
            short = "λ=1e-4"
        elif "0.0005" in model:
            short = "λ=5e-4"
        elif "0.0015" in model:
            short = "λ=1.5e-3"
        else:
            short = model[:25]

        print(f"\n  Model: {short}")
        print(f"  {'Metric':<16s} {'n':>4s} {'Δ mean':>9s} {'95% CI':>22s} {'Excl 0?':>8s} {'Wilc p':>8s}")
        print("  " + "-" * 69)

        for metric in metrics:
            # Pool all per-seed deltas across scenarios
            all_deltas = []
            for p in paired:
                if p["model"] == model and p["metric"] == metric:
                    all_deltas.extend(p["per_seed_deltas"])

            if not all_deltas:
                continue

            all_deltas = np.array(all_deltas)
            mean_d, ci_lo, ci_hi = bootstrap_ci(all_deltas, n_resamples=10_000)
            w_stat, w_p, n_eff = wilcoxon_signed_rank(all_deltas)

            ci_str = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
            excl = "YES" if not (ci_lo <= 0 <= ci_hi) else "no"
            w_str = f"{w_p:.4f}" if not np.isnan(w_p) else "n/a"

            print(f"  {metric_labels[metric]:<16s} {len(all_deltas):>4d} "
                  f"{mean_d:>+9.3f} {ci_str:>22s} {excl:>8s} {w_str:>8s}")

            pooled_results.append({
                "model": model,
                "model_short": short,
                "metric": metric,
                "n_pooled": len(all_deltas),
                "mean_delta": float(mean_d),
                "ci_95_lower": float(ci_lo),
                "ci_95_upper": float(ci_hi),
                "ci_excludes_zero": not (ci_lo <= 0 <= ci_hi),
                "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
            })

    out_dir = get_results_dir(A3_DIR)
    out_path = out_dir / "pooled_cross_scenario.json"
    with open(out_path, "w") as f:
        json.dump(pooled_results, f, indent=2)
    print(f"\n  Saved: {out_path.name}")

    return pooled_results


# ================================================================
# Part 4: Generate figures
# ================================================================

def part4_figures(utopia_result: dict, paired: list, pooled: list):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available — skipping figures")
        return

    out_dir = get_results_dir(A3_DIR, "figures")

    # ── Figure 1: Utopia-point trade-off plot ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    fid_norm = np.array(utopia_result["fid_normalised"])
    risk_norm = np.array(utopia_result["risk_normalised"])
    lams = utopia_result["filtered_lambdas"]
    names = utopia_result["filtered_names"]
    distances = np.array(utopia_result["distances"])
    best_idx = int(np.argmin(distances))

    # Color by distance
    sc = ax.scatter(fid_norm, risk_norm, c=distances, cmap="RdYlGn_r",
                    s=80, edgecolors="k", linewidths=0.5, zorder=5)
    ax.scatter(fid_norm[best_idx], risk_norm[best_idx],
               s=200, facecolors="none", edgecolors="red", linewidths=2.5,
               zorder=6, label=f"Selected: λ*={lams[best_idx]:.1e}")

    # Utopia point
    ax.scatter(0, 0, marker="*", s=300, c="gold", edgecolors="k",
               linewidths=0.8, zorder=7, label="Utopia (0, 0)")

    # Annotate key points
    for i, (fx, fy, lam) in enumerate(zip(fid_norm, risk_norm, lams)):
        if lam in [0.0, 5e-4, 1.5e-3, 5e-3] or i == best_idx:
            label = "Fidelity" if lam == 0.0 else f"λ={lam:.1e}"
            ax.annotate(label, (fx, fy), textcoords="offset points",
                        xytext=(8, 6), fontsize=7, alpha=0.85)

    plt.colorbar(sc, ax=ax, label="Utopia distance")
    ax.set_xlabel("Normalised fidelity loss ->", fontsize=11)
    ax.set_ylabel("Normalised risk (total) ->", fontsize=11)
    ax.set_title("Fidelity–Risk Trade-off with Utopia-Point Selection", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"utopia_tradeoff.{ext}", dpi=200)
    plt.close(fig)
    print(f"\n  Saved: utopia_tradeoff.png/pdf")

    # ── Figure 2: Forest plot of paired CIs (λ=1.5e-3 only, comfort metrics) ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    scenarios = ["nominal", "cold_snap", "forecast_error"]
    scenario_labels = {"nominal": "Nominal", "cold_snap": "Cold Snap",
                       "forecast_error": "Forecast Error"}
    comfort_metrics = ["deg_hours_cold_occ", "cvar90_cold_occ_C",
                       "peak_cold_violation_occ_C"]
    metric_labels = {
        "deg_hours_cold_occ": "Δ Cold CDH",
        "cvar90_cold_occ_C": "Δ CVaR90 cold (°C)",
        "peak_cold_violation_occ_C": "Δ Peak cold (°C)",
    }

    for ax, metric in zip(axes, comfort_metrics):
        y_pos = []
        means = []
        ci_los = []
        ci_his = []
        labels = []

        for i, scen in enumerate(scenarios):
            match = [p for p in paired
                     if "0.0015" in p["model"]
                     and p["scenario"] == scen
                     and p["metric"] == metric]
            if match:
                p = match[0]
                y_pos.append(i)
                means.append(p["mean_delta"])
                ci_los.append(p["ci_95_lower"])
                ci_his.append(p["ci_95_upper"])
                labels.append(scenario_labels[scen])

        y_pos = np.array(y_pos)
        means = np.array(means)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)

        # Plot CI bars
        ax.barh(y_pos, means, height=0.4, color="steelblue", alpha=0.6, zorder=3)
        for y, lo, hi in zip(y_pos, ci_los, ci_his):
            ax.plot([lo, hi], [y, y], color="black", linewidth=2, zorder=4)
            ax.plot([lo, lo], [y - 0.1, y + 0.1], color="black", linewidth=1.5)
            ax.plot([hi, hi], [y - 0.1, y + 0.1], color="black", linewidth=1.5)

        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(metric_labels[metric], fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(metric_labels[metric], fontsize=11)

    fig.suptitle("RAMC λ=1.5×10⁻³ vs Fidelity: Paired Δ with 95% Bootstrap CI",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"forest_plot_bootstrap_ci.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved: forest_plot_bootstrap_ci.png/pdf")


# ================================================================
# Part 5: Narrative text for manuscript
# ================================================================

def part5_narrative(utopia_result: dict, paired: list, pooled: list):
    """Generate the revised narrative text for the manuscript."""
    out_dir = get_results_dir(A3_DIR)

    selected_lam = utopia_result["selected_lambda_total_risk"]
    selected_model = utopia_result["selected_model_total_risk"]
    dist = utopia_result["utopia_distance_total_risk"]

    # Find pooled CDH CI for λ=1.5e-3
    cdh_pooled = None
    for p in pooled:
        if "0.0015" in p.get("model", "") and p["metric"] == "deg_hours_cold_occ":
            cdh_pooled = p
            break

    lines = [
        "=" * 65,
        "REVISED MANUSCRIPT TEXT — λ-selection and statistical reporting",
        "=" * 65,
        "",
        "--- Section: λ-selection protocol (Methods) ---",
        "",
        f"The risk weight λ was selected prior to closed-loop evaluation",
        f"using the normalised utopia-point method applied to Phase 2",
        f"open-loop metrics. Both the fidelity loss Lfid and the total",
        f"CVaR risk Rtotal were normalised to [0, 1] via min-max scaling",
        f"across the λ-sweep (excluding the Raw MSE baseline). The model",
        f"closest to the utopia point (0, 0) in normalised (fidelity, risk)",
        f"space was selected as λ* = {selected_lam:.1e}",
        f"({selected_model}, utopia distance = {dist:.4f}).",
        f"Phase 3 closed-loop evaluation serves as independent validation",
        f"of this selection, not as a selection criterion.",
        "",
        "--- Section: Phase 3 model roster (Methods) ---",
        "",
        "Four models were evaluated in closed loop:",
        f"  - Fidelity Baseline (λ = 0): primary comparator, best open-loop fidelity.",
        f"  - RAMC λ = 1×10⁻⁴: near-baseline risk regularisation; tests whether",
        f"    even minimal RAMC improves control.",
        f"  - RAMC λ = 5×10⁻⁴: region of strongest open-loop risk reduction;",
        f"    included to test whether open-loop risk correlates with closed-loop safety.",
        f"  - RAMC λ = 1.5×10⁻³: selected by the utopia-point rule; represents",
        f"    the recommended operating point.",
        "",
        "--- Section: Statistical reporting (Results) ---",
        "",
    ]

    if cdh_pooled:
        lines.append(
            f"Pooled across all three scenarios (n = {cdh_pooled['n_pooled']}), "
            f"RAMC λ = 1.5×10⁻³ reduced cold degree-hours by "
            f"{abs(cdh_pooled['mean_delta']):.2f} deg-h relative to the Fidelity "
            f"Baseline (95% bootstrap CI: "
            f"[{cdh_pooled['ci_95_lower']:+.2f}, {cdh_pooled['ci_95_upper']:+.2f}])."
        )
        if cdh_pooled["ci_excludes_zero"]:
            lines.append("The confidence interval excludes zero, indicating a")
            lines.append("statistically reliable improvement.")
        else:
            lines.append("The confidence interval includes zero; the effect is")
            lines.append("suggestive but not statistically conclusive at n = 15.")

    lines.extend([
        "",
        "--- PHRASES TO DELETE from current manuscript ---",
        "",
        '  DELETE: "confirmed in preliminary runs"',
        '  DELETE: "empirically best closed-loop performer"',
        '  DELETE: any implication that λ was chosen by looking at Phase 3 results',
        "",
        "--- REPLACEMENT LANGUAGE ---",
        "",
        f'  REPLACE WITH: "selected by the normalised utopia-point method',
        f'  applied to Phase 2 open-loop fidelity and risk metrics"',
    ])

    text = "\n".join(lines)
    print(f"\n{text}")

    out_path = out_dir / "revised_narrative.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  Saved: {out_path.name}")


# ================================================================
# Part 6: Summary CSV for easy reference
# ================================================================

def part6_summary_csv(utopia_result: dict, paired: list, pooled: list):
    """Save a compact CSV summary of paired CIs."""
    out_dir = get_results_dir(A3_DIR)
    out_path = out_dir / "paired_ci_summary.csv"

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "scenario", "metric", "n_pairs",
            "mean_delta", "ci_95_lower", "ci_95_upper",
            "ci_excludes_zero", "wilcoxon_p",
        ])
        for p in paired:
            writer.writerow([
                p["model"], p["scenario"], p["metric_label"], p["n_pairs"],
                f"{p['mean_delta']:.4f}",
                f"{p['ci_95_lower']:.4f}",
                f"{p['ci_95_upper']:.4f}",
                p["ci_excludes_zero"],
                f"{p['wilcoxon_p']:.4f}" if p["wilcoxon_p"] is not None else "n/a",
            ])
    print(f"\n  Saved: {out_path.name}")

    # Also save pooled
    out_path2 = out_dir / "pooled_ci_summary.csv"
    with open(out_path2, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "metric", "n_pooled",
            "mean_delta", "ci_95_lower", "ci_95_upper",
            "ci_excludes_zero", "wilcoxon_p",
        ])
        for p in pooled:
            writer.writerow([
                p["model_short"], p["metric"], p["n_pooled"],
                f"{p['mean_delta']:.4f}",
                f"{p['ci_95_lower']:.4f}",
                f"{p['ci_95_upper']:.4f}",
                p["ci_excludes_zero"],
                f"{p['wilcoxon_p']:.4f}" if p["wilcoxon_p"] is not None else "n/a",
            ])
    print(f"  Saved: {out_path2.name}")


# ================================================================
# Main
# ================================================================

def main():
    print("\n" + "#" * 65)
    print("  Assignment 3 — λ-Selection and Statistical Stability")
    print("  Pure analysis of existing Phase 2 + Phase 3 data")
    print("#" * 65)

    # Part 1: λ selection
    utopia_result = part1_utopia_selection()

    # Part 2: Bootstrap CIs
    paired = part2_statistical_analysis()

    # Part 3: Pooled analysis
    pooled = part3_pooled_analysis(paired)

    # Part 4: Figures
    part4_figures(utopia_result, paired, pooled)

    # Part 5: Narrative
    part5_narrative(utopia_result, paired, pooled)

    # Part 6: Summary CSVs
    part6_summary_csv(utopia_result, paired, pooled)

    print("\n" + "=" * 65)
    print("  Assignment 3 Part 1 complete.")
    print(f"  All outputs in: {get_results_dir(A3_DIR)}")
    print("=" * 65)
    print("\n  NEXT STEP: Part 2 of A3 requires new NMPC simulations —")
    print("  a reduced-budget λ-sweep in closed loop with additional λ values.")
    print("  This is a separate script (run_a3_sweep.py) that takes ~4-8 hours.")


if __name__ == "__main__":
    main()
