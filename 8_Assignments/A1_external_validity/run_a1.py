"""
Assignment 1 — RC-on-RC External Validity Stress Test
======================================================

Overnight runner for the perturbed-plant experiments.

Design (from Assignment 1 document):
  - 2 models: Fidelity Baseline + RAMC λ=1.5e-3
  - 1 scenario: forecast_error (largest existing RAMC advantage)
  - ~14 plant variants: 1 matched + 4 one-at-a-time + 8 random + 1 severe
  - 3 seeds per (model, plant) combination
  - Total: 2 × 14 × 3 = 84 experiments
  - Expected runtime: ~8-12 hours on CPU

Resume support: results are saved per-experiment as JSON. If the script
is interrupted, rerun it and it will skip completed experiments.

Run from 8_Assignments/ directory:
    python A1_external_validity/run_a1.py

Or from Spyder:
    %cd <project_root>/8_Assignments
    %run A1_external_validity/run_a1.py
"""

import matplotlib
matplotlib.use("Agg")

import sys
import json
import time
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
    SCENARIO_DIR, A1_DIR, get_results_dir,
    STRESS_TEST_MODELS,
    PHASE3_SEEDS, REDUCED_SEEDS,
    NMPC_DEFAULTS, COMFORT_BAND,
    W_COLD, W_HOT, W_ENERGY, W_TERMINAL,
    STEPS_PER_EPISODE, DT_MINUTES,
    T_SUPPLY_RANGE, MDOT_RANGE,
    setup_imports,
)
from shared.perturbed_rc_plant import (
    generate_all_plant_variants,
    validate_plant_variant,
    save_plant_variants,
    BASELINE_PARAMS,
)
from shared.stats_utils import bootstrap_ci

# Set up imports to access existing modules
setup_imports()

from closed_loop_nmpc import ClosedLoopNMPCSimulator
from rc_ground_truth import RCGroundTruthModel

# Import shared_constants for energy cost rate
from shared_constants import ENERGY_COST_RATE


# =============================================================================
# Configuration
# =============================================================================

DT_SECONDS = 600
SIMULATION_STEPS = STEPS_PER_EPISODE  # 1008 (7 days)

# Models to test (focused pair from Assignment 1 document)
MODELS = {
    "Fidelity": {
        "path": str(CHECKPOINT_FIDELITY),
        "lambda": 0.0,
    },
    "RAMC_1.5e-3": {
        "path": str(CHECKPOINT_RAMC_15E4),
        "lambda": 0.0015,
    },
}

# Scenario: forecast_error only (per Assignment 1 document)
SCENARIO = {
    "name": "forecast_error",
    "truth": str(SCENARIO_DIR / "forecast_error_truth.csv"),
    "forecast": str(SCENARIO_DIR / "forecast_error_forecast.csv"),
}

# Seeds (reduced set for A1: 3 seeds)
SEEDS = REDUCED_SEEDS  # [42, 123, 456]

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
    """Extract the metrics needed for A1 analysis."""
    return {
        "deg_hours_cold_occ": float(results["deg_hours_cold_occ"]),
        "cvar90_cold_occ_C": float(results["cvar90_cold_occ_C"]),
        "peak_cold_violation_occ_C": float(results["peak_cold_violation_occ_C"]),
        "total_energy_kWh": float(results["total_energy_kWh"]),
        "T_air_occ_mean_C": float(results["T_air_occ_mean_C"]),
        "T_air_occ_min_C": float(results["T_air_occ_min_C"]),
        "fallback_rate": float(np.mean(results["fallback_used"])),
    }


# =============================================================================
# Main experiment runner
# =============================================================================

def run_a1_experiments():
    """Run the full A1 perturbed-plant stress test."""
    print("\n" + "#" * 70)
    print("  Assignment 1 — RC-on-RC External Validity Stress Test")
    print("  Perturbed-Plant Experiments")
    print("#" * 70)

    # ── Output directories ──
    raw_dir = get_results_dir(A1_DIR, "raw")
    plants_dir = get_results_dir(A1_DIR, "perturbed_plants")

    # ── Step 1: Generate and validate plant variants ──
    print("\n[Step 1] Generating plant variants...")
    variants = generate_all_plant_variants(n_random=8, random_seed=42)

    print(f"  Generated {len(variants)} variants. Validating...")
    valid_variants = []
    for v in variants:
        ok = validate_plant_variant(v["params"], v["label"])
        status = "OK" if ok else "REJECTED"
        print(f"    {v['label']:<25s} {status}")
        if ok:
            valid_variants.append(v)

    save_plant_variants(valid_variants, plants_dir / "plant_variants.json")
    print(f"  {len(valid_variants)} valid plant variants")

    # ── Step 2: Build experiment list ──
    experiments = []
    exp_id = 1
    for model_name, model_cfg in MODELS.items():
        for variant in valid_variants:
            for seed in SEEDS:
                experiments.append({
                    "id": exp_id,
                    "model_name": model_name,
                    "model_path": model_cfg["path"],
                    "model_lambda": model_cfg["lambda"],
                    "plant_label": variant["label"],
                    "plant_category": variant["category"],
                    "plant_params": variant["params"],
                    "seed": seed,
                })
                exp_id += 1

    total = len(experiments)
    print(f"\n[Step 2] Experiment matrix: {len(MODELS)} models × "
          f"{len(valid_variants)} plants × {len(SEEDS)} seeds = {total}")

    # ── Step 3: Save config ──
    config = {
        "assignment": "A1_external_validity",
        "description": "Perturbed-plant stress test for RC-on-RC external validity",
        "models": {k: {"lambda": v["lambda"]} for k, v in MODELS.items()},
        "scenario": SCENARIO["name"],
        "n_plant_variants": len(valid_variants),
        "plant_categories": {
            "matched": sum(1 for v in valid_variants if v["category"] == "matched"),
            "one_at_a_time": sum(1 for v in valid_variants if v["category"] == "one_at_a_time"),
            "random_ensemble": sum(1 for v in valid_variants if v["category"] == "random_ensemble"),
            "combined_severe": sum(1 for v in valid_variants if v["category"] == "combined_severe"),
        },
        "seeds": SEEDS,
        "total_experiments": total,
        "nmpc_settings": {k: str(v) for k, v in NMPC_KWARGS.items()},
        "started": datetime.now().isoformat(),
    }
    with open(raw_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Step 4: Run experiments with resume support ──
    print(f"\n[Step 3] Running {total} experiments...")
    print(f"  Results saved to: {raw_dir}")
    print(f"  Resume: rerun script to skip completed experiments\n")

    # Simulator cache: keyed by (model_name, plant_label)
    sim_cache = {}
    results_all = []
    start_total = time.time()

    for exp in experiments:
        result_file = raw_dir / f"result_{exp['id']:03d}.json"

        # Resume: skip if result exists
        if result_file.exists():
            with open(result_file) as f:
                cached = json.load(f)
            if cached.get("success", False):
                results_all.append(cached)
                print(f"  EXP {exp['id']:3d}/{total} — SKIP (cached) "
                      f"{exp['model_name']} | {exp['plant_label']} | seed={exp['seed']}")
                continue

        print(f"\n{'='*65}")
        print(f"  EXP {exp['id']}/{total}: {exp['model_name']} | "
              f"{exp['plant_label']} ({exp['plant_category']}) | seed={exp['seed']}")

        start = time.time()
        try:
            # Get or create simulator
            cache_key = (exp["model_name"], exp["plant_label"])
            if cache_key not in sim_cache:
                print(f"    Creating simulator...")
                sim = ClosedLoopNMPCSimulator(
                    nn_model_path=exp["model_path"],
                    weather_truth_path=SCENARIO["truth"],
                    weather_forecast_path=SCENARIO["forecast"],
                    verbose_init=(len(sim_cache) == 0),
                    **NMPC_KWARGS,
                )
                # CRITICAL: replace plant with perturbed version
                sim.ground_truth = RCGroundTruthModel(
                    params=exp["plant_params"],
                    dt_seconds=DT_SECONDS,
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
                log_interval=200,
            )

            elapsed = time.time() - start
            metrics = extract_key_metrics(sim_results)

            result = {
                "experiment_id": exp["id"],
                "model_name": exp["model_name"],
                "model_lambda": exp["model_lambda"],
                "plant_label": exp["plant_label"],
                "plant_category": exp["plant_category"],
                "seed": exp["seed"],
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
                "model_name": exp["model_name"],
                "plant_label": exp["plant_label"],
                "plant_category": exp["plant_category"],
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
        print(f"    Progress: {len(results_all)}/{total} | "
              f"ETA: {eta_min:.0f} min")

    # ── Step 5: Aggregate and save ──
    print(f"\n{'='*65}")
    print("  Aggregating results...")

    # Save all results
    with open(raw_dir / "all_results.json", "w") as f:
        json.dump({"config": config, "results": results_all}, f, indent=2)

    # Compute delta metrics
    compute_delta_analysis(results_all, valid_variants)

    elapsed_total = time.time() - start_total
    print(f"\n{'='*65}")
    print(f"  Assignment 1 complete. Total time: {elapsed_total/3600:.1f} hours")
    print(f"  Results in: {raw_dir}")
    print(f"{'='*65}")


def compute_delta_analysis(results: list, variants: list):
    """
    Compute Δ-metrics (RAMC minus Fidelity) for each plant variant.
    This is the core A1 deliverable.
    """
    out_dir = get_results_dir(A1_DIR)
    successful = [r for r in results if r.get("success", False)]

    # Group by (plant_label, seed) -> {model_name: metrics}
    grouped = defaultdict(dict)
    for r in successful:
        key = (r["plant_label"], r["seed"])
        grouped[key][r["model_name"]] = r["metrics"]

    # Compute paired deltas per plant variant
    metrics_to_compare = [
        "deg_hours_cold_occ", "cvar90_cold_occ_C",
        "peak_cold_violation_occ_C", "total_energy_kWh",
    ]

    plant_summary = []
    for variant in variants:
        label = variant["label"]
        category = variant["category"]

        deltas = {m: [] for m in metrics_to_compare}
        for seed in SEEDS:
            key = (label, seed)
            if key not in grouped:
                continue
            g = grouped[key]
            if "Fidelity" not in g or "RAMC_1.5e-3" not in g:
                continue
            for m in metrics_to_compare:
                d = g["RAMC_1.5e-3"][m] - g["Fidelity"][m]
                deltas[m].append(d)

        if not deltas["deg_hours_cold_occ"]:
            continue

        entry = {
            "plant_label": label,
            "plant_category": category,
            "n_seeds": len(deltas["deg_hours_cold_occ"]),
        }

        for m in metrics_to_compare:
            arr = np.array(deltas[m])
            mean_d, ci_lo, ci_hi = bootstrap_ci(arr, n_resamples=10_000)
            entry[f"delta_{m}_mean"] = float(mean_d)
            entry[f"delta_{m}_median"] = float(np.median(arr))
            entry[f"delta_{m}_ci_lower"] = float(ci_lo)
            entry[f"delta_{m}_ci_upper"] = float(ci_hi)
            entry[f"delta_{m}_sign_negative"] = int(np.sum(arr < 0))

        plant_summary.append(entry)

    # Print summary table
    print(f"\n  {'Plant':<25s} {'Δ CDH':>10s} {'95% CI':>22s} {'Δ Energy':>12s} {'RAMC wins':>10s}")
    print("  " + "-" * 81)

    ramc_advantage_count = 0
    total_plants = 0
    for s in plant_summary:
        ci_str = f"[{s['delta_deg_hours_cold_occ_ci_lower']:+.2f}, {s['delta_deg_hours_cold_occ_ci_upper']:+.2f}]"
        wins = f"{s['delta_deg_hours_cold_occ_sign_negative']}/{s['n_seeds']}"
        print(f"  {s['plant_label']:<25s} "
              f"{s['delta_deg_hours_cold_occ_mean']:>+10.2f} "
              f"{ci_str:>22s} "
              f"{s['delta_total_energy_kWh_mean']:>+12.0f} "
              f"{wins:>10s}")
        total_plants += 1
        if s["delta_deg_hours_cold_occ_mean"] < 0:
            ramc_advantage_count += 1

    print(f"\n  RAMC advantage (negative Δ CDH): {ramc_advantage_count}/{total_plants} plants")

    # Save
    with open(out_dir / "delta_analysis.json", "w") as f:
        json.dump(plant_summary, f, indent=2)

    # Also save as CSV
    import csv
    csv_path = out_dir / "delta_summary.csv"
    if plant_summary:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=plant_summary[0].keys())
            writer.writeheader()
            writer.writerows(plant_summary)

    print(f"  Saved: delta_analysis.json, delta_summary.csv")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_a1_experiments()
