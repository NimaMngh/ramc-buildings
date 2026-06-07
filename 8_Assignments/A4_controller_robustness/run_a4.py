"""
Assignment 4 — Controller-Robustness Hypothesis
=================================================

Tests H4: The ranking between RAMC and Fidelity is robust to reasonable
changes in NMPC horizon, optimizer iterations, learning rate, and move
blocking. The controller remains computationally feasible.

Design (staged):
  Stage 1 (mandatory): Horizon × Iterations sensitivity
    - H ∈ {12, 24, 36, 48} × iters ∈ {10, 25, 50}
    - 2 models (Fidelity, RAMC 1.5e-3) × 2 scenarios × 3 seeds
    - Subtracting shared default (H=24, iters=25): 6 unique settings
    - Total: 2 × 2 × 3 × 6 = 72 experiments

  Stage 2 (if time allows): LR × Block size
    - LR ∈ {0.02, 0.05, 0.10} × block ∈ {2, 4, 6}
    - Same 2 models × 2 scenarios × 3 seeds
    - Subtracting shared default: 8 unique settings
    - Total: 2 × 2 × 3 × 8 = 96 experiments

Run from 8_Assignments/ directory:
    python A4_controller_robustness/run_a4.py
    python A4_controller_robustness/run_a4.py --stage1-only

Author: RAMC Assignment Framework
"""

import matplotlib
matplotlib.use("Agg")

import sys
import json
import time
import csv
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Imports from shared infrastructure ──
THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    CHECKPOINT_FIDELITY, CHECKPOINT_RAMC_15E4,
    SCENARIO_DIR, A4_DIR,
    get_results_dir, setup_imports,
    NMPC_DEFAULTS,
    W_COLD, W_HOT, W_ENERGY, W_TERMINAL,
    STEPS_PER_EPISODE, REDUCED_SEEDS,
    STRESS_TEST_MODELS,
)
from shared.stats_utils import bootstrap_ci

setup_imports()

from closed_loop_nmpc import ClosedLoopNMPCSimulator
from shared_constants import ENERGY_COST_RATE


# =============================================================================
# Configuration
# =============================================================================

DT_SECONDS = 600
SIMULATION_STEPS = STEPS_PER_EPISODE

MODELS = {
    "Fidelity": str(CHECKPOINT_FIDELITY),
    "RAMC_1.5e-3": str(CHECKPOINT_RAMC_15E4),
}

SCENARIOS = [
    {"name": "nominal",
     "truth": str(SCENARIO_DIR / "nominal_truth.csv"),
     "forecast": str(SCENARIO_DIR / "nominal_forecast.csv")},
    {"name": "forecast_error",
     "truth": str(SCENARIO_DIR / "forecast_error_truth.csv"),
     "forecast": str(SCENARIO_DIR / "forecast_error_forecast.csv")},
]

SEEDS = REDUCED_SEEDS  # [42, 123, 456]

# Default NMPC settings (baseline)
DEFAULT_SETTINGS = {
    "horizon": 24,
    "block_size": 4,
    "adam_iters": 25,
    "learning_rate": 0.05,
}

# Stage 1: One-at-a-time sensitivity (vary one, hold others at default)
STAGE1_SETTINGS = [
    # Default (baseline — always included)
    {"horizon": 24, "block_size": 4, "adam_iters": 25, "learning_rate": 0.05},
    # Horizon sweep (iters fixed at 25)
    {"horizon": 12, "block_size": 4, "adam_iters": 25, "learning_rate": 0.05},
    {"horizon": 36, "block_size": 4, "adam_iters": 25, "learning_rate": 0.05},
    {"horizon": 48, "block_size": 4, "adam_iters": 25, "learning_rate": 0.05},
    # Iteration sweep (horizon fixed at 24)
    {"horizon": 24, "block_size": 4, "adam_iters": 10, "learning_rate": 0.05},
    {"horizon": 24, "block_size": 4, "adam_iters": 50, "learning_rate": 0.05},
]

# Stage 2: LR and block size sensitivity (one-at-a-time, default H=24, iters=25)
STAGE2_SETTINGS = [
    # LR sweep (block_size fixed at 4)
    {"horizon": 24, "block_size": 4, "adam_iters": 25, "learning_rate": 0.02},
    {"horizon": 24, "block_size": 4, "adam_iters": 25, "learning_rate": 0.10},
    # Block size sweep (LR fixed at 0.05)
    {"horizon": 24, "block_size": 2, "adam_iters": 25, "learning_rate": 0.05},
    {"horizon": 24, "block_size": 6, "adam_iters": 25, "learning_rate": 0.05},
]


def create_initial_state(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = np.array([19.5, 17.0, 18.0, 35.0, 32.0, 30.0])
    perturb = np.array([
        rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
        rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0),
    ])
    return np.clip(base + perturb, 5.0, 60.0)


def extract_metrics(results: dict) -> dict:
    return {
        "deg_hours_cold_occ": float(results["deg_hours_cold_occ"]),
        "cvar90_cold_occ_C": float(results["cvar90_cold_occ_C"]),
        "peak_cold_violation_occ_C": float(results["peak_cold_violation_occ_C"]),
        "total_energy_kWh": float(results["total_energy_kWh"]),
        "fallback_rate": float(np.mean(results["fallback_used"])),
        "median_solve_time_ms": float(np.median(results["solver_time_ms"])),
        "p95_solve_time_ms": float(np.percentile(results["solver_time_ms"], 95)),
        "max_solve_time_ms": float(np.max(results["solver_time_ms"])),
    }


def setting_label(s: dict) -> str:
    return f"H{s['horizon']}_iter{s['adam_iters']}_lr{s['learning_rate']}_bs{s['block_size']}"


# =============================================================================
# Main
# =============================================================================

def run_a4_experiments(stage1_only: bool = False):
    print("\n" + "#" * 70)
    print("  Assignment 4 — Controller-Robustness Hypothesis")
    print("  NMPC Sensitivity Study")
    print("#" * 70)
    
    raw_dir = get_results_dir(A4_DIR, "raw")
    
    # Build settings list
    settings_list = STAGE1_SETTINGS.copy()
    if not stage1_only:
        settings_list.extend(STAGE2_SETTINGS)
    
    # Deduplicate by label
    seen = set()
    unique_settings = []
    for s in settings_list:
        lab = setting_label(s)
        if lab not in seen:
            seen.add(lab)
            unique_settings.append(s)
    
    # Build experiment list
    experiments = []
    exp_id = 1
    for s in unique_settings:
        for model_name, model_path in MODELS.items():
            for scenario in SCENARIOS:
                for seed in SEEDS:
                    experiments.append({
                        "id": exp_id,
                        "model_name": model_name,
                        "model_path": model_path,
                        "scenario": scenario["name"],
                        "scenario_truth": scenario["truth"],
                        "scenario_forecast": scenario["forecast"],
                        "seed": seed,
                        "settings": s,
                        "settings_label": setting_label(s),
                    })
                    exp_id += 1
    
    total = len(experiments)
    stages = "Stage 1 only" if stage1_only else "Stage 1 + 2"
    print(f"\n  {stages}: {len(unique_settings)} settings × "
          f"{len(MODELS)} models × {len(SCENARIOS)} scenarios × "
          f"{len(SEEDS)} seeds = {total} experiments")
    
    config = {
        "assignment": "A4_controller_robustness",
        "stages": stages,
        "n_settings": len(unique_settings),
        "settings": [{"label": setting_label(s), **s} for s in unique_settings],
        "models": list(MODELS.keys()),
        "scenarios": [s["name"] for s in SCENARIOS],
        "seeds": SEEDS,
        "total_experiments": total,
        "started": datetime.now().isoformat(),
    }
    with open(raw_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run experiments
    print(f"\n  Running {total} experiments...")
    sim_cache = {}
    results_all = []
    start_total = time.time()
    
    for exp in experiments:
        result_file = raw_dir / f"result_{exp['id']:03d}.json"
        
        if result_file.exists():
            with open(result_file) as f:
                cached = json.load(f)
            if cached.get("success", False):
                results_all.append(cached)
                print(f"  EXP {exp['id']:3d}/{total} — SKIP "
                      f"{exp['model_name']} | {exp['settings_label']} | "
                      f"{exp['scenario']} | s={exp['seed']}")
                continue
        
        print(f"\n  EXP {exp['id']}/{total}: {exp['model_name']} | "
              f"{exp['settings_label']} | {exp['scenario']} | seed={exp['seed']}")
        
        start = time.time()
        s = exp["settings"]
        
        try:
            cache_key = (exp["model_name"], exp["settings_label"], exp["scenario"])
            if cache_key not in sim_cache:
                sim = ClosedLoopNMPCSimulator(
                    nn_model_path=exp["model_path"],
                    weather_truth_path=exp["scenario_truth"],
                    weather_forecast_path=exp["scenario_forecast"],
                    nmpc_horizon=s["horizon"],
                    nmpc_block_size=s["block_size"],
                    nmpc_n_iter=s["adam_iters"],
                    nmpc_lr=s["learning_rate"],
                    nmpc_grad_clip=NMPC_DEFAULTS["grad_clip"],
                    w_energy=W_ENERGY,
                    w_cold=W_COLD,
                    w_hot=W_HOT,
                    w_du=1e-3,
                    w_terminal=W_TERMINAL,
                    w_trust=0.0,
                    du_max=np.array([2.0, 0.3]),
                    energy_cost_rate=ENERGY_COST_RATE,
                    dtype=torch.float64,
                    verbose_init=(len(sim_cache) == 0),
                )
                sim_cache[cache_key] = sim
            else:
                sim = sim_cache[cache_key]
            
            sim.reset_for_new_seed()
            torch.manual_seed(exp["seed"])
            init_state = create_initial_state(exp["seed"])
            
            sim_results = sim.simulate_episode(
                initial_state=init_state,
                simulation_steps=SIMULATION_STEPS,
                verbose=False,
                log_interval=250,
            )
            
            elapsed = time.time() - start
            metrics = extract_metrics(sim_results)
            
            result = {
                "experiment_id": exp["id"],
                "model_name": exp["model_name"],
                "scenario": exp["scenario"],
                "seed": exp["seed"],
                "settings_label": exp["settings_label"],
                "settings": exp["settings"],
                "success": True,
                "elapsed_s": elapsed,
                "metrics": metrics,
            }
            
            print(f"    CDH={metrics['deg_hours_cold_occ']:.2f} | "
                  f"Peak={metrics['peak_cold_violation_occ_C']:.2f}°C | "
                  f"Solve={metrics['median_solve_time_ms']:.0f}ms | "
                  f"Time={elapsed:.0f}s")
        
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            result = {
                "experiment_id": exp["id"],
                "model_name": exp["model_name"],
                "scenario": exp["scenario"],
                "seed": exp["seed"],
                "settings_label": exp["settings_label"],
                "success": False,
                "error": str(e),
            }
        
        results_all.append(result)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        done = sum(1 for r in results_all if r.get("success", False))
        elapsed_total = time.time() - start_total
        rate = done / elapsed_total if elapsed_total > 0 else 0
        remaining = total - len(results_all)
        eta_min = remaining / rate / 60 if rate > 0 else 0
        print(f"    Progress: {len(results_all)}/{total} | ETA: {eta_min:.0f} min")
    
    # Save all
    with open(raw_dir / "all_results.json", "w") as f:
        json.dump({"config": config, "results": results_all}, f, indent=2)
    
    # Aggregate
    compute_sensitivity_summary(results_all, unique_settings)
    
    elapsed_total = time.time() - start_total
    print(f"\n  Assignment 4 complete. Total time: {elapsed_total/3600:.1f} hours")


def compute_sensitivity_summary(results: list, settings: list):
    """Compute and print the sensitivity summary table."""
    out_dir = get_results_dir(A4_DIR)
    successful = [r for r in results if r.get("success", False)]
    
    # Group by (settings_label, model_name, scenario)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in successful:
        key = (r["settings_label"], r["model_name"], r["scenario"])
        for m, v in r["metrics"].items():
            grouped[key][m].append(v)
    
    # Summary rows
    summary = []
    for s in settings:
        lab = setting_label(s)
        for scenario_name in ["nominal", "forecast_error"]:
            fid_key = (lab, "Fidelity", scenario_name)
            ramc_key = (lab, "RAMC_1.5e-3", scenario_name)
            
            fid_cdh = grouped.get(fid_key, {}).get("deg_hours_cold_occ", [])
            ramc_cdh = grouped.get(ramc_key, {}).get("deg_hours_cold_occ", [])
            fid_time = grouped.get(fid_key, {}).get("median_solve_time_ms", [])
            
            if fid_cdh and ramc_cdh:
                delta = np.mean(ramc_cdh) - np.mean(fid_cdh)
                entry = {
                    "settings": lab,
                    "scenario": scenario_name,
                    "H": s["horizon"],
                    "iters": s["adam_iters"],
                    "lr": s["learning_rate"],
                    "block_size": s["block_size"],
                    "fidelity_CDH_mean": float(np.mean(fid_cdh)),
                    "ramc_CDH_mean": float(np.mean(ramc_cdh)),
                    "delta_CDH": float(delta),
                    "ramc_advantage": bool(delta < 0),
                    "solve_time_ms_median": float(np.mean(fid_time)) if fid_time else None,
                }
                summary.append(entry)
    
    # Print
    print(f"\n  {'Settings':<30s} {'Scenario':<15s} {'Fid CDH':>8s} {'RAMC CDH':>9s} "
          f"{'ΔCDH':>8s} {'Advantage':>10s} {'Solve ms':>9s}")
    print(f"  {'-'*91}")
    
    ramc_wins = 0
    total_comparisons = 0
    
    for s in summary:
        adv_str = "RAMC" if s["ramc_advantage"] else "Fidelity"
        solve_str = f"{s['solve_time_ms_median']:.0f}" if s["solve_time_ms_median"] else "N/A"
        print(f"  {s['settings']:<30s} {s['scenario']:<15s} "
              f"{s['fidelity_CDH_mean']:>8.2f} {s['ramc_CDH_mean']:>9.2f} "
              f"{s['delta_CDH']:>+8.2f} {adv_str:>10s} {solve_str:>9s}")
        total_comparisons += 1
        if s["ramc_advantage"]:
            ramc_wins += 1
    
    print(f"\n  RAMC advantage in {ramc_wins}/{total_comparisons} settings")
    
    # Runtime feasibility
    all_times = []
    for r in successful:
        all_times.append(r["metrics"].get("median_solve_time_ms", 0))
    
    if all_times:
        print(f"\n  Runtime (all experiments):")
        print(f"    Median solve time: {np.median(all_times):.0f} ms")
        print(f"    P95 solve time:    {np.percentile(all_times, 95):.0f} ms")
        print(f"    Max solve time:    {np.max(all_times):.0f} ms")
        print(f"    10-min feasible:   {'YES' if np.percentile(all_times, 95) < 120000 else 'CHECK'}")
    
    # Save
    with open(out_dir / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    if summary:
        csv_path = out_dir / "sensitivity_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    
    print(f"  Saved: sensitivity_summary.json/csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-only", action="store_true",
                       help="Run only Stage 1 (horizon × iterations)")
    args = parser.parse_args()
    
    run_a4_experiments(stage1_only=args.stage1_only)
