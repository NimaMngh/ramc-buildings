"""
Assignment 6 — Energy–Comfort Pareto Analysis
===============================================

Tests H6: The comfort improvement delivered by RAMC places it on or
near a defensible energy–comfort Pareto frontier, and the energy
premium is operationally justifiable.

This script aggregates results from:
  - Phase 3 original matrix (Fidelity, RAMC λ=1e-4, 5e-4, 1.5e-3)
  - Assignment 2 (RC-NMPC benchmark, conservative margins)
  - Assignment 3 (broader λ sweep if available)

Produces:
  1. Energy–comfort Pareto frontier plot (CDH vs energy, peak vs energy)
  2. Cost translation table (monetary cost per CDH avoided)
  3. Practical interpretation paragraph data

Run from 8_Assignments/ directory:
    python A6_pareto_analysis/run_a6.py

Author: RAMC Assignment Framework
"""

import matplotlib
matplotlib.use("Agg")

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Imports from shared infrastructure ──
THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    NMPC_AGGREGATE_CSV, NMPC_ALL_RESULTS_JSON,
    A2_DIR, A3_DIR, A6_DIR,
    get_results_dir,
    W_ENERGY,
)


# =============================================================================
# Configuration
# =============================================================================

ENERGY_TARIFF_SEK = 0.9  # SEK/kWh
SEK_TO_EUR = 0.087       # approximate conversion


# =============================================================================
# Data loading
# =============================================================================

def load_phase3_results() -> list:
    """Load Phase 3 original matrix results."""
    records = []
    
    if NMPC_AGGREGATE_CSV.exists():
        df = pd.read_csv(NMPC_AGGREGATE_CSV)
        print(f"  Phase 3: loaded {len(df)} rows from aggregate CSV")
        
        for _, row in df.iterrows():
            records.append({
                "source": "Phase3",
                "model_label": row.get("model", row.get("model_name", "unknown")),
                "scenario": row.get("scenario", "unknown"),
                "cdh_mean": row.get("deg_hours_cold_occ_mean", row.get("deg_hours_cold_occ", float("nan"))),
                "cdh_std": row.get("deg_hours_cold_occ_std", 0.0),
                "peak_mean": row.get("peak_cold_violation_occ_C_mean",
                                    row.get("peak_cold_violation_occ_C", float("nan"))),
                "peak_std": row.get("peak_cold_violation_occ_C_std", 0.0),
                "energy_mean": row.get("total_energy_kWh_mean",
                                      row.get("total_energy_kWh", float("nan"))),
                "energy_std": row.get("total_energy_kWh_std", 0.0),
                "cvar90_mean": row.get("cvar90_cold_occ_C_mean",
                                      row.get("cvar90_cold_occ_C", float("nan"))),
            })
    else:
        # Try loading all_results.json
        if NMPC_ALL_RESULTS_JSON.exists():
            with open(NMPC_ALL_RESULTS_JSON) as f:
                data = json.load(f)
            
            results = data.get("results", data) if isinstance(data, dict) else data
            print(f"  Phase 3: loaded all_results.json ({len(results)} experiments)")
            
            grouped = defaultdict(lambda: defaultdict(list))
            for r in results:
                metrics = r.get("metrics", r)
                key = (r.get("model", r.get("model_name")), r.get("scenario"))
                grouped[key]["cdh"].append(metrics.get("deg_hours_cold_occ", float("nan")))
                grouped[key]["peak"].append(metrics.get("peak_cold_violation_occ_C", float("nan")))
                grouped[key]["energy"].append(metrics.get("total_energy_kWh", float("nan")))
                grouped[key]["cvar90"].append(metrics.get("cvar90_cold_occ_C", float("nan")))
            
            for (model, scenario), vals in grouped.items():
                records.append({
                    "source": "Phase3",
                    "model_label": model,
                    "scenario": scenario,
                    "cdh_mean": np.nanmean(vals["cdh"]),
                    "cdh_std": np.nanstd(vals["cdh"]),
                    "peak_mean": np.nanmean(vals["peak"]),
                    "peak_std": np.nanstd(vals["peak"]),
                    "energy_mean": np.nanmean(vals["energy"]),
                    "energy_std": np.nanstd(vals["energy"]),
                    "cvar90_mean": np.nanmean(vals["cvar90"]),
                })
        else:
            print("  Phase 3: no results found")
    
    return records


def load_a2_results() -> list:
    """Load Assignment 2 (benchmark sufficiency) results."""
    records = []
    a2_summary = A2_DIR / "results" / "benchmark_summary.json"
    
    if a2_summary.exists():
        with open(a2_summary) as f:
            data = json.load(f)
        print(f"  A2: loaded {len(data)} summary entries")
        
        for entry in data:
            records.append({
                "source": "A2",
                "model_label": entry["model_label"],
                "scenario": entry["scenario"],
                "cdh_mean": entry.get("deg_hours_cold_occ_mean", float("nan")),
                "cdh_std": entry.get("deg_hours_cold_occ_std", 0.0),
                "peak_mean": entry.get("peak_cold_violation_occ_C_mean", float("nan")),
                "peak_std": entry.get("peak_cold_violation_occ_C_std", 0.0),
                "energy_mean": entry.get("total_energy_kWh_mean", float("nan")),
                "energy_std": entry.get("total_energy_kWh_std", 0.0),
                "cvar90_mean": entry.get("cvar90_cold_occ_C_mean", float("nan")),
            })
    else:
        print(f"  A2: no results found at {a2_summary}")
    
    return records


def load_a3_results() -> list:
    """Load Assignment 3 (λ sweep) results if available."""
    records = []
    
    # Try multiple possible locations
    for candidate in [
        A3_DIR / "results" / "paired_bootstrap_analysis.json",
        A3_DIR / "results" / "pooled_cross_scenario.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            print(f"  A3: loaded {candidate.name}")
            # A3 data is primarily for the λ narrative, not Pareto
            # Include if it has per-λ energy/comfort summaries
            break
    else:
        print(f"  A3: no results found (optional for Pareto)")
    
    return records


# =============================================================================
# Pareto analysis
# =============================================================================

def compute_pareto_frontier(points: list) -> list:
    """
    Find Pareto-optimal points minimizing both CDH and energy.
    Returns indices of Pareto-optimal points.
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is <= on both and < on at least one
            if (points[j][0] <= points[i][0] and points[j][1] <= points[i][1] and
                (points[j][0] < points[i][0] or points[j][1] < points[i][1])):
                is_pareto[i] = False
                break
    
    return [i for i in range(n) if is_pareto[i]]


def run_a6():
    """Run the Pareto analysis."""
    print("\n" + "#" * 70)
    print("  Assignment 6 — Energy–Comfort Pareto Analysis")
    print("#" * 70)
    
    out_dir = get_results_dir(A6_DIR)
    fig_dir = get_results_dir(A6_DIR, "figures")
    
    # ── Load all available results ──
    print("\n[1] Loading results from all sources...")
    all_records = []
    all_records.extend(load_phase3_results())
    all_records.extend(load_a2_results())
    all_records.extend(load_a3_results())
    
    if not all_records:
        print("\n  ERROR: No results found. Run Phase 3, A2, or A3 first.")
        return
    
    print(f"\n  Total records: {len(all_records)}")
    
    # ── Build per-scenario Pareto data ──
    scenarios = sorted(set(r["scenario"] for r in all_records if r["scenario"]))
    
    all_pareto_data = []
    
    for scenario in scenarios:
        scene_records = [r for r in all_records if r["scenario"] == scenario]
        if not scene_records:
            continue
        
        print(f"\n  Scenario: {scenario} ({len(scene_records)} models)")
        print(f"  {'Model':<35s} {'CDH':>8s} {'Peak':>8s} {'Energy':>8s} {'Source':>8s}")
        print(f"  {'-'*69}")
        
        for r in sorted(scene_records, key=lambda x: x.get("cdh_mean", 999)):
            print(f"  {r['model_label']:<35s} "
                  f"{r.get('cdh_mean', float('nan')):>8.2f} "
                  f"{r.get('peak_mean', float('nan')):>8.3f} "
                  f"{r.get('energy_mean', float('nan')):>8.0f} "
                  f"{r['source']:>8s}")
            
            all_pareto_data.append({**r, "scenario": scenario})
    
    # ── Compute cost translation ──
    print("\n[2] Computing cost translation...")
    cost_table = []
    
    # Find Fidelity baseline for each scenario
    for scenario in scenarios:
        scene = [r for r in all_records if r["scenario"] == scenario]
        
        fid = [r for r in scene if "Fidelity" in r["model_label"]
               and "margin" not in r["model_label"]
               and r["source"] == "Phase3"]
        
        if not fid:
            fid = [r for r in scene if "Fidelity" in r["model_label"]
                   and "margin" not in r["model_label"]]
        
        if not fid:
            continue
        
        fid_record = fid[0]
        fid_cdh = fid_record.get("cdh_mean", float("nan"))
        fid_energy = fid_record.get("energy_mean", float("nan"))
        
        for r in scene:
            if r["model_label"] == fid_record["model_label"]:
                continue
            
            cdh = r.get("cdh_mean", float("nan"))
            energy = r.get("energy_mean", float("nan"))
            
            if np.isnan(cdh) or np.isnan(energy):
                continue
            
            delta_cdh = cdh - fid_cdh
            delta_energy = energy - fid_energy
            delta_cost_sek = delta_energy * ENERGY_TARIFF_SEK
            
            cost_per_cdh = abs(delta_cost_sek / delta_cdh) if abs(delta_cdh) > 0.01 else float("inf")
            
            cost_table.append({
                "scenario": scenario,
                "model": r["model_label"],
                "source": r["source"],
                "CDH": cdh,
                "delta_CDH": delta_cdh,
                "energy_kWh": energy,
                "delta_energy_kWh": delta_energy,
                "energy_premium_SEK": delta_cost_sek,
                "energy_premium_EUR": delta_cost_sek * SEK_TO_EUR,
                "cost_per_CDH_avoided_SEK": cost_per_cdh if delta_cdh < 0 else None,
            })
    
    if cost_table:
        print(f"\n  {'Model':<35s} {'Scenario':<15s} {'ΔCDH':>8s} {'ΔEnergy':>10s} "
              f"{'Premium':>10s} {'SEK/CDH':>10s}")
        print(f"  {'-'*90}")
        
        for ct in cost_table:
            cdh_str = f"{ct['delta_CDH']:>+8.2f}" if not np.isnan(ct["delta_CDH"]) else "    N/A"
            energy_str = f"{ct['delta_energy_kWh']:>+10.0f}"
            premium_str = f"{ct['energy_premium_SEK']:>+10.0f}"
            cost_str = (f"{ct['cost_per_CDH_avoided_SEK']:>10.0f}"
                       if ct.get("cost_per_CDH_avoided_SEK") and
                       ct["cost_per_CDH_avoided_SEK"] < 1e6
                       else "       N/A")
            
            print(f"  {ct['model']:<35s} {ct['scenario']:<15s} "
                  f"{cdh_str} {energy_str} {premium_str} {cost_str}")
    
    # ── Plot Pareto frontiers ──
    print("\n[3] Generating Pareto plots...")
    plot_pareto(all_pareto_data, scenarios, fig_dir)
    
    # ── Save ──
    with open(out_dir / "pareto_data.json", "w") as f:
        json.dump(all_pareto_data, f, indent=2, default=str)
    
    with open(out_dir / "cost_translation.json", "w") as f:
        json.dump(cost_table, f, indent=2, default=str)
    
    print(f"\n  Saved: pareto_data.json, cost_translation.json")
    print("=" * 70)


def plot_pareto(data: list, scenarios: list, fig_dir: Path):
    """Generate Pareto frontier plots."""
    
    # Color/marker scheme by model category
    style_map = {
        "RC_Exact": {"color": "gold", "marker": "*", "s": 200, "label": "RC-NMPC (exact)"},
        "Fidelity": {"color": "steelblue", "marker": "o", "s": 80, "label": "Fidelity Baseline"},
        "RAMC": {"color": "firebrick", "marker": "D", "s": 80, "label": "RAMC"},
        "margin": {"color": "green", "marker": "s", "s": 60, "label": "Conservative Margin"},
    }
    
    def get_style(label):
        if "RC_Exact" in label or "RC_NMPC" in label:
            return style_map["RC_Exact"]
        elif "margin" in label:
            return style_map["margin"]
        elif "RAMC" in label or "lambda" in label:
            return style_map["RAMC"]
        else:
            return style_map["Fidelity"]
    
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 5))
    if n_scenarios == 1:
        axes = [axes]
    
    for ax, scenario in zip(axes, scenarios):
        scene_data = [d for d in data if d["scenario"] == scenario]
        
        # Track labels for legend dedup
        legend_labels = set()
        
        for d in scene_data:
            cdh = d.get("cdh_mean", float("nan"))
            energy = d.get("energy_mean", float("nan"))
            
            if np.isnan(cdh) or np.isnan(energy):
                continue
            
            style = get_style(d["model_label"])
            label = style["label"] if style["label"] not in legend_labels else None
            if label:
                legend_labels.add(label)
            
            ax.scatter(energy, cdh,
                      c=style["color"], marker=style["marker"],
                      s=style["s"], label=label, zorder=5,
                      edgecolors="black", linewidths=0.5)
            
            # Error bars if available
            cdh_std = d.get("cdh_std", 0)
            energy_std = d.get("energy_std", 0)
            if cdh_std > 0:
                ax.errorbar(energy, cdh, yerr=cdh_std, xerr=energy_std,
                           fmt="none", ecolor=style["color"], alpha=0.4, capsize=2)
            
            # Label key points
            short_name = d["model_label"].replace("_rollout", "").replace("Baseline", "BL")
            if len(short_name) > 20:
                short_name = short_name[:18] + "…"
            ax.annotate(short_name, (energy, cdh),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=6, alpha=0.7)
        
        ax.set_xlabel("Total Energy (kWh)", fontsize=10)
        ax.set_ylabel("Cold Degree-Hours (deg-h)", fontsize=10)
        ax.set_title(f"Scenario: {scenario}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        
        # Arrow indicating "better" direction
        ax.annotate("", xy=(0.05, 0.05), xycoords="axes fraction",
                    xytext=(0.15, 0.15), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
        ax.text(0.03, 0.17, "Better", transform=ax.transAxes,
               fontsize=8, color="gray", style="italic")
    
    fig.suptitle("Energy–Comfort Pareto Frontier", fontsize=13, y=1.02)
    plt.tight_layout()
    
    fig.savefig(fig_dir / "pareto_energy_comfort.png", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "pareto_energy_comfort.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pareto_energy_comfort.png/.pdf")


if __name__ == "__main__":
    run_a6()
