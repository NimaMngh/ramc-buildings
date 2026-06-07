"""
Assignment 2 — Benchmark Sufficiency Hypothesis
=================================================

Tests H2: The reported RAMC gains are meaningful relative to stronger
baselines, not only relative to the fidelity-only model.

Experiments:
  A) Exact-model RC-NMPC benchmark (RC model as planning model in NMPC)
  B) Conservative margin baselines (Fidelity model with shifted T_min)
  C) Gap-to-exact-model metric computation

Design:
  - RC benchmark: 1 model × 3 scenarios × 3 seeds = 9 experiments
  - Conservative margins (3 levels): 3 × 3 scenarios × 3 seeds = 27 experiments
  - Total: 36 experiments
  - Expected runtime: ~12-24 hours on CPU

Run from 8_Assignments/ directory:
    python A2_benchmark_sufficiency/run_a2.py

Author: RAMC Assignment Framework
"""

import matplotlib
matplotlib.use("Agg")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import csv
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from types import MethodType

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ── Imports from shared infrastructure ──
THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    CHECKPOINT_FIDELITY, CHECKPOINT_RAMC_15E4,
    SCENARIOS, SCENARIO_DIR, A2_DIR,
    get_results_dir, setup_imports,
    NMPC_DEFAULTS, COMFORT_BAND,
    W_COLD, W_HOT, W_ENERGY, W_TERMINAL,
    STEPS_PER_EPISODE, REDUCED_SEEDS,
    T_SUPPLY_RANGE, MDOT_RANGE,
    NMPC_AGGREGATE_CSV, NMPC_ALL_RESULTS_JSON,
    STRESS_TEST_MODELS,
)
from shared.rc_nmpc_benchmark import RCPlanningModel, verify_rc_planning_model
from shared.stats_utils import bootstrap_ci

setup_imports()

from closed_loop_nmpc import ClosedLoopNMPCSimulator
from rc_ground_truth import RCGroundTruthModel
from shared_constants import (
    ENERGY_COST_RATE,
    get_occupancy_status, get_comfort_bounds,
    OCC_TARGET_C, DEADBAND_C,
)


# =============================================================================
# Configuration
# =============================================================================

DT_SECONDS = 600
SIMULATION_STEPS = STEPS_PER_EPISODE  # 1008

SEEDS = REDUCED_SEEDS  # [42, 123, 456]

# Conservative margin levels (shift T_min upward during occupied hours)
MARGIN_SHIFTS = {
    "margin_0.3": 0.3,   # T_min_occ = 20.0 -> 20.3
    "margin_0.5": 0.5,   # T_min_occ = 20.0 -> 20.5
    "margin_1.0": 1.0,   # T_min_occ = 20.0 -> 21.0
}

# NMPC settings (must match Phase 3 exactly)
NMPC_KWARGS = {
    "nmpc_horizon": NMPC_DEFAULTS["horizon"],
    "nmpc_block_size": NMPC_DEFAULTS["block_size"],
    "nmpc_n_iter": NMPC_DEFAULTS["adam_iters"],
    "nmpc_lr": NMPC_DEFAULTS["learning_rate"],
    "nmpc_grad_clip": NMPC_DEFAULTS["grad_clip"],
    "w_energy": W_ENERGY,
    "w_cold": W_COLD,
    "w_hot": W_HOT,
    "w_du": 1e-3,
    "w_terminal": W_TERMINAL,
    "w_trust": 0.0,
    "du_max": np.array([2.0, 0.3]),
    "energy_cost_rate": ENERGY_COST_RATE,
    "dtype": torch.float64,
}

# All scenarios
SCENARIO_LIST = [
    {"name": "nominal",
     "truth": str(SCENARIO_DIR / "nominal_truth.csv"),
     "forecast": str(SCENARIO_DIR / "nominal_forecast.csv")},
    {"name": "cold_snap",
     "truth": str(SCENARIO_DIR / "cold_snap_truth.csv"),
     "forecast": str(SCENARIO_DIR / "cold_snap_forecast.csv")},
    {"name": "forecast_error",
     "truth": str(SCENARIO_DIR / "forecast_error_truth.csv"),
     "forecast": str(SCENARIO_DIR / "forecast_error_forecast.csv")},
]


def create_initial_state(seed: int) -> np.ndarray:
    """Create seed-dependent initial state (MUST match run_nmpc_matrix.py)."""
    rng = np.random.RandomState(seed)
    base = np.array([19.5, 17.0, 18.0, 35.0, 32.0, 30.0])
    perturb = np.array([
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-5.0, 5.0),
        rng.uniform(-5.0, 5.0),
        rng.uniform(-5.0, 5.0),
    ])
    return np.clip(base + perturb, 5.0, 60.0)


def extract_key_metrics(results: dict) -> dict:
    """Extract metrics for A2 analysis."""
    return {
        "deg_hours_cold_occ": float(results["deg_hours_cold_occ"]),
        "cvar90_cold_occ_C": float(results["cvar90_cold_occ_C"]),
        "peak_cold_violation_occ_C": float(results["peak_cold_violation_occ_C"]),
        "total_energy_kWh": float(results["total_energy_kWh"]),
        "T_air_occ_mean_C": float(results["T_air_occ_mean_C"]),
        "T_air_occ_min_C": float(results["T_air_occ_min_C"]),
        "fallback_rate": float(np.mean(results["fallback_used"])),
        "median_solve_time_ms": float(np.median(results["solver_time_ms"])),
    }


# =============================================================================
# Factory: create simulator with RC planning model
# =============================================================================

def create_rc_nmpc_simulator(
    scenario: dict,
    verbose: bool = True,
) -> ClosedLoopNMPCSimulator:
    """
    Create a ClosedLoopNMPCSimulator that uses the RC model as the planner.
    
    Strategy: create normally with Fidelity model (to set up infrastructure),
    then replace the planning model with a differentiable RC model.
    """
    # Create with Fidelity model to set up infrastructure
    sim = ClosedLoopNMPCSimulator(
        nn_model_path=str(CHECKPOINT_FIDELITY),
        weather_truth_path=scenario["truth"],
        weather_forecast_path=scenario["forecast"],
        verbose_init=verbose,
        **NMPC_KWARGS,
    )
    
    # Replace planning model with differentiable RC model
    rc_model = RCPlanningModel()
    rc_model.eval()
    rc_model = rc_model.to(dtype=NMPC_KWARGS["dtype"])
    
    sim.model = rc_model
    sim.nmpc.model = rc_model
    
    if verbose:
        print(f"  Planning model replaced with: {rc_model}")
    
    return sim


# =============================================================================
# Factory: create simulator with conservative margin
# =============================================================================

def create_margin_simulator(
    scenario: dict,
    margin_shift: float,
    verbose: bool = True,
) -> ClosedLoopNMPCSimulator:
    """
    Create a ClosedLoopNMPCSimulator with the Fidelity model but shifted
    comfort bounds (T_min_occ increased by margin_shift).
    
    This tests whether a simple static margin achieves the same comfort
    improvement as RAMC without the training framework.
    """
    sim = ClosedLoopNMPCSimulator(
        nn_model_path=str(CHECKPOINT_FIDELITY),
        weather_truth_path=scenario["truth"],
        weather_forecast_path=scenario["forecast"],
        verbose_init=verbose,
        **NMPC_KWARGS,
    )
    
    # Monkey-patch the comfort bounds builder to shift T_min upward
    original_method = sim._build_comfort_bounds_sequence
    shift = margin_shift
    
    def shifted_comfort_bounds(self_ref, start_timestamp, horizon):
        from datetime import timedelta
        Tmin_seq = np.zeros(horizon, dtype=float)
        Tmax_seq = np.zeros(horizon, dtype=float)
        for h in range(horizon):
            ts_h = start_timestamp + timedelta(seconds=(h + 1) * DT_SECONDS)
            occ_h = get_occupancy_status(ts_h)
            tmin_h, tmax_h = get_comfort_bounds(occ_h)
            if occ_h:
                tmin_h += shift  # shift lower bound upward for occupied hours
            Tmin_seq[h] = tmin_h
            Tmax_seq[h] = tmax_h
        return Tmin_seq, Tmax_seq
    
    sim._build_comfort_bounds_sequence = MethodType(shifted_comfort_bounds, sim)
    
    if verbose:
        print(f"  Comfort margin shift: T_min_occ += {margin_shift}°C "
              f"(effective: [{20.0 + margin_shift}, 22.0]°C)")
    
    return sim


# =============================================================================
# Main experiment runner
# =============================================================================

def run_a2_experiments():
    """Run the full A2 benchmark sufficiency experiments."""
    print("\n" + "#" * 70)
    print("  Assignment 2 — Benchmark Sufficiency Hypothesis")
    print("  RC-NMPC Benchmark + Conservative Margin Baselines")
    print("#" * 70)
    
    raw_dir = get_results_dir(A2_DIR, "raw")
    
    # ── Step 0: Verify RC planning model ──
    print("\n[Step 0] Verifying RC planning model...")
    verify_ok = verify_rc_planning_model(verbose=True)
    if not verify_ok:
        print("  WARNING: RC model verification failed! Proceeding with caution.")
    
    # ── Step 1: Build experiment list ──
    experiments = []
    exp_id = 1
    
    # A) RC-NMPC benchmark
    for scenario in SCENARIO_LIST:
        for seed in SEEDS:
            experiments.append({
                "id": exp_id,
                "model_type": "RC_NMPC",
                "model_label": "RC_Exact",
                "scenario": scenario["name"],
                "scenario_truth": scenario["truth"],
                "scenario_forecast": scenario["forecast"],
                "seed": seed,
                "margin_shift": 0.0,
            })
            exp_id += 1
    
    # B) Conservative margin baselines
    for margin_label, margin_val in MARGIN_SHIFTS.items():
        for scenario in SCENARIO_LIST:
            for seed in SEEDS:
                experiments.append({
                    "id": exp_id,
                    "model_type": "Conservative_Margin",
                    "model_label": f"Fidelity_{margin_label}",
                    "scenario": scenario["name"],
                    "scenario_truth": scenario["truth"],
                    "scenario_forecast": scenario["forecast"],
                    "seed": seed,
                    "margin_shift": margin_val,
                })
                exp_id += 1
    
    total = len(experiments)
    n_rc = sum(1 for e in experiments if e["model_type"] == "RC_NMPC")
    n_margin = sum(1 for e in experiments if e["model_type"] == "Conservative_Margin")
    print(f"\n[Step 1] Experiment matrix: {n_rc} RC-NMPC + {n_margin} margin = {total} total")
    
    # ── Step 2: Save config ──
    config = {
        "assignment": "A2_benchmark_sufficiency",
        "experiments": {
            "RC_NMPC": f"{n_rc} (3 scenarios × {len(SEEDS)} seeds)",
            "Conservative_Margin": f"{n_margin} ({len(MARGIN_SHIFTS)} margins × 3 scenarios × {len(SEEDS)} seeds)",
        },
        "margin_shifts": MARGIN_SHIFTS,
        "seeds": SEEDS,
        "nmpc_settings": {k: str(v) for k, v in NMPC_KWARGS.items()},
        "started": datetime.now().isoformat(),
    }
    with open(raw_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # ── Step 3: Run experiments with resume support ──
    print(f"\n[Step 2] Running {total} experiments...")
    print(f"  Results saved to: {raw_dir}")
    
    sim_cache = {}
    results_all = []
    start_total = time.time()
    
    for exp in experiments:
        result_file = raw_dir / f"result_{exp['id']:03d}.json"
        
        # Resume
        if result_file.exists():
            with open(result_file) as f:
                cached = json.load(f)
            if cached.get("success", False):
                results_all.append(cached)
                print(f"  EXP {exp['id']:3d}/{total} — SKIP (cached) "
                      f"{exp['model_label']} | {exp['scenario']} | seed={exp['seed']}")
                continue
        
        print(f"\n{'='*65}")
        print(f"  EXP {exp['id']}/{total}: {exp['model_label']} | "
              f"{exp['scenario']} | seed={exp['seed']}")
        
        start = time.time()
        try:
            # RC experiments: fresh simulator every time (no caching)
            # to avoid autograd graph retention across seeds
            scenario_cfg = {
                "truth": exp["scenario_truth"],
                "forecast": exp["scenario_forecast"],
            }
            
            if exp["model_type"] == "RC_NMPC":
                sim = create_rc_nmpc_simulator(
                    scenario_cfg, verbose=(exp["id"] == 1))
            else:
                # Margin experiments: safe to cache (NN model, no graph issues)
                cache_key = (exp["model_label"], exp["scenario"])
                if cache_key not in sim_cache:
                    sim = create_margin_simulator(
                        scenario_cfg, exp["margin_shift"],
                        verbose=(len(sim_cache) == 0))
                    sim_cache[cache_key] = sim
                else:
                    sim = sim_cache[cache_key]
            
            sim.reset_for_new_seed()
            torch.manual_seed(exp["seed"])
            init_state = create_initial_state(exp["seed"])
            
            # RC experiments: verbose progress so we can see it's working
            is_rc = (exp["model_type"] == "RC_NMPC")
            sim_results = sim.simulate_episode(
                initial_state=init_state,
                simulation_steps=SIMULATION_STEPS,
                verbose=is_rc,
                log_interval=50 if is_rc else 200,
            )
            
            elapsed = time.time() - start
            metrics = extract_key_metrics(sim_results)
            
            result = {
                "experiment_id": exp["id"],
                "model_type": exp["model_type"],
                "model_label": exp["model_label"],
                "scenario": exp["scenario"],
                "seed": exp["seed"],
                "margin_shift": exp["margin_shift"],
                "success": True,
                "elapsed_s": elapsed,
                "metrics": metrics,
            }
            
            print(f"    CDH={metrics['deg_hours_cold_occ']:.2f} | "
                  f"Peak={metrics['peak_cold_violation_occ_C']:.2f}°C | "
                  f"Energy={metrics['total_energy_kWh']:.0f}kWh | "
                  f"Time={elapsed:.0f}s")
        
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            result = {
                "experiment_id": exp["id"],
                "model_type": exp["model_type"],
                "model_label": exp["model_label"],
                "scenario": exp["scenario"],
                "seed": exp["seed"],
                "success": False,
                "error": str(e),
            }
        
        results_all.append(result)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Progress
        done = sum(1 for r in results_all if r.get("success", False))
        elapsed_total = time.time() - start_total
        rate = done / elapsed_total if elapsed_total > 0 else 0
        remaining = total - len(results_all)
        eta_min = remaining / rate / 60 if rate > 0 else 0
        print(f"    Progress: {len(results_all)}/{total} | ETA: {eta_min:.0f} min")
    
    # ── Step 4: Aggregate and compute gap metrics ──
    print(f"\n{'='*65}")
    print("  Aggregating results and computing gap metrics...")
    
    with open(raw_dir / "all_results.json", "w") as f:
        json.dump({"config": config, "results": results_all}, f, indent=2)
    
    compute_gap_analysis(results_all)
    
    elapsed_total = time.time() - start_total
    print(f"\n{'='*65}")
    print(f"  Assignment 2 complete. Total time: {elapsed_total/3600:.1f} hours")
    print(f"  Results in: {raw_dir}")


# =============================================================================
# Gap-to-exact-model analysis
# =============================================================================

def compute_gap_analysis(results: list):
    """
    Compute gap-to-exact-model metrics and benchmark ladder.
    
    Also loads existing Phase 3 results for Fidelity and RAMC to compute
    the gap-closed metric.
    """
    out_dir = get_results_dir(A2_DIR)
    successful = [r for r in results if r.get("success", False)]
    
    # ── Load existing Phase 3 results ──
    phase3_data = {}
    if NMPC_AGGREGATE_CSV.exists():
        print("  Loading Phase 3 aggregate metrics...")
        phase3_df = pd.read_csv(NMPC_AGGREGATE_CSV)
        phase3_data = phase3_df.to_dict("records")
    
    # ── Aggregate A2 results by (model_label, scenario) ──
    grouped = defaultdict(lambda: defaultdict(list))
    for r in successful:
        key = (r["model_label"], r["scenario"])
        for metric, val in r["metrics"].items():
            grouped[key][metric].append(val)
    
    # Build summary table
    summary = []
    for (model_label, scenario), metrics_dict in sorted(grouped.items()):
        entry = {
            "model_label": model_label,
            "scenario": scenario,
            "n_seeds": len(metrics_dict.get("deg_hours_cold_occ", [])),
        }
        for metric, vals in metrics_dict.items():
            arr = np.array(vals)
            entry[f"{metric}_mean"] = float(arr.mean())
            entry[f"{metric}_std"] = float(arr.std())
        summary.append(entry)
    
    # ── Print benchmark ladder ──
    print(f"\n  {'Model':<30s} {'Scenario':<15s} {'CDH':>8s} {'Peak':>8s} {'Energy':>8s}")
    print(f"  {'-'*71}")
    
    for s in summary:
        cdh = s.get("deg_hours_cold_occ_mean", float("nan"))
        peak = s.get("peak_cold_violation_occ_C_mean", float("nan"))
        energy = s.get("total_energy_kWh_mean", float("nan"))
        print(f"  {s['model_label']:<30s} {s['scenario']:<15s} "
              f"{cdh:>8.2f} {peak:>8.3f} {energy:>8.0f}")
    
    # ── Compute gap-to-exact-model ──
    # For each scenario, compute: gap_closed = (M_fidelity - M_ramc) / (M_fidelity - M_exact)
    gap_analysis = []
    for scenario_cfg in SCENARIO_LIST:
        scenario_name = scenario_cfg["name"]
        
        exact_key = ("RC_Exact", scenario_name)
        
        if exact_key not in grouped:
            continue
        
        exact_cdh = np.mean(grouped[exact_key]["deg_hours_cold_occ"])
        exact_peak = np.mean(grouped[exact_key]["peak_cold_violation_occ_C"])
        exact_energy = np.mean(grouped[exact_key]["total_energy_kWh"])
        
        gap_entry = {
            "scenario": scenario_name,
            "exact_model_CDH": exact_cdh,
            "exact_model_peak": exact_peak,
            "exact_model_energy": exact_energy,
        }
        
        # Add margin results
        for margin_label in MARGIN_SHIFTS:
            margin_key = (f"Fidelity_{margin_label}", scenario_name)
            if margin_key in grouped:
                gap_entry[f"{margin_label}_CDH"] = np.mean(grouped[margin_key]["deg_hours_cold_occ"])
                gap_entry[f"{margin_label}_energy"] = np.mean(grouped[margin_key]["total_energy_kWh"])
        
        gap_analysis.append(gap_entry)
    
    # Save
    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(out_dir / "gap_analysis.json", "w") as f:
        json.dump(gap_analysis, f, indent=2)
    
    if summary:
        csv_path = out_dir / "benchmark_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    
    print(f"\n  Saved: benchmark_summary.json/csv, gap_analysis.json")


if __name__ == "__main__":
    run_a2_experiments()
