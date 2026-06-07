#!/usr/bin/env python3
"""
run_solver_budget_sensitivity.py
================================

NMPC solver-budget sensitivity (revision plan P0; addresses Reviewer #2's
concern that RAMC's closed-loop advantage may be an artifact of the fixed
25-iteration Adam budget).

Question
--------
Does RAMC's ranking over the fidelity baseline persist as the NMPC solver is
given more optimization budget? If the advantage were an optimization artifact,
giving the solver more iterations would erase it. If it is a genuine
model-quality effect, the advantage should persist (and may grow).

Design
------
Iteration sweep at the default horizon/block/lr, holding everything else fixed:
  - N_Adam in {25, 50, 100}
  - 2 models: Fidelity vs RAMC lambda=1.5e-3 (the headline model)
  - 2 scenarios: nominal, forecast_error  (aligns with the run_a4 horizon sweep)
  - 5 paired seeds  (matches the ablation / margin-grid seed set)
  => 3 budgets x 2 models x 2 scenarios x 5 seeds = 60 experiments

This is the consistent-provenance version: it OVERRIDES the checkpoints to the
NEW training dir (RAMC_FULL_cvar_20260518_182822) so the table matches the
ablation, margin-grid, and Figure-8 numbers rather than the original-submission
run_a4 (old checkpoints, 3 seeds, {10,25,50}).

Adding cold_snap: set INCLUDE_COLD_SNAP = True below (adds 30 experiments).

Output
------
results_solver_budget_{timestamp}/
    raw/result_*.json, all_results.json, config.json
    sensitivity_summary.csv   (per budget x scenario: paired RAMC-Fidelity dCDH + CI)

Place this file in the same assignments folder as run_a4.py so the shared.*
imports resolve, then run:
    python run_solver_budget_sensitivity.py
    python run_solver_budget_sensitivity.py --quick   # iters {25,100}, 1 seed
"""

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

THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    SCENARIO_DIR, get_results_dir,
    NMPC_DEFAULTS,
    W_COLD, W_HOT, W_ENERGY, W_TERMINAL,
    STEPS_PER_EPISODE,
    setup_imports,
)
from shared.stats_utils import bootstrap_ci

setup_imports()

from closed_loop_nmpc import ClosedLoopNMPCSimulator
from shared_constants import ENERGY_COST_RATE


# =============================================================================
# Configuration
# =============================================================================

DT_SECONDS = 600
SIMULATION_STEPS = STEPS_PER_EPISODE  # 1008 (7 days)

# --- NEW checkpoints (consistency with ablation + margin grid). Edit if your
#     tree differs. ---
NEW_RESULTS_DIR = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
    r"\results\RAMC_FULL_cvar_20260518_182822"
)
CHECKPOINT_FIDELITY = NEW_RESULTS_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth"
CHECKPOINT_RAMC_15E4 = NEW_RESULTS_DIR / "RAMC_lambda_0.0015_op_cvar_rollout_a1.0_best.pth"

MODELS = {
    "Fidelity":    str(CHECKPOINT_FIDELITY),
    "RAMC_1.5e-3": str(CHECKPOINT_RAMC_15E4),
}

# Solver-budget grid (the only thing that varies). Horizon/block/lr fixed at default.
ADAM_ITERS = [25, 50, 100]

# 5 paired seeds, matching run_nmpc_matrix / ablation / margin grid.
SEEDS = [42, 123, 456, 789, 1000]

# Scenarios: nominal + forecast_error (aligns with the run_a4 horizon sweep).
INCLUDE_COLD_SNAP = False
SCENARIOS = [
    {"name": "nominal",
     "truth": str(SCENARIO_DIR / "nominal_truth.csv"),
     "forecast": str(SCENARIO_DIR / "nominal_forecast.csv")},
    {"name": "forecast_error",
     "truth": str(SCENARIO_DIR / "forecast_error_truth.csv"),
     "forecast": str(SCENARIO_DIR / "forecast_error_forecast.csv")},
]
if INCLUDE_COLD_SNAP:
    SCENARIOS.append(
        {"name": "cold_snap",
         "truth": str(SCENARIO_DIR / "cold_snap_truth.csv"),
         "forecast": str(SCENARIO_DIR / "cold_snap_forecast.csv")})

# Fixed NMPC settings (default horizon/block/lr; only adam_iters varies).
H_DEFAULT = NMPC_DEFAULTS["horizon"]        # 24
BS_DEFAULT = NMPC_DEFAULTS["block_size"]    # 4
LR_DEFAULT = NMPC_DEFAULTS["learning_rate"] # 0.05

METRICS = ["deg_hours_cold_occ", "cvar90_cold_occ_C",
           "peak_cold_violation_occ_C", "total_energy_kWh"]


def create_initial_state(seed: int) -> np.ndarray:
    """Seed-dependent initial state (MUST match run_nmpc_matrix.py)."""
    rng = np.random.RandomState(seed)
    base = np.array([19.5, 17.0, 18.0, 35.0, 32.0, 30.0])
    perturb = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0), rng.uniform(-5.0, 5.0),
                        rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)])
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


# =============================================================================
# Runner
# =============================================================================

def run(output_dir, quick=False):
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    iters_list = [25, 100] if quick else ADAM_ITERS
    seeds = [SEEDS[0]] if quick else SEEDS

    experiments = []
    exp_id = 1
    for n_iter in iters_list:
        for model_name, model_path in MODELS.items():
            for scenario in SCENARIOS:
                for seed in seeds:
                    experiments.append({
                        "id": exp_id, "model_name": model_name,
                        "model_path": model_path, "n_iter": n_iter,
                        "scenario": scenario["name"],
                        "scenario_truth": scenario["truth"],
                        "scenario_forecast": scenario["forecast"],
                        "seed": seed,
                    })
                    exp_id += 1
    total = len(experiments)

    print("#" * 70)
    print("NMPC SOLVER-BUDGET SENSITIVITY (P0)")
    print("#" * 70)
    print(f"Output: {output_dir}")
    print(f"Checkpoints: {NEW_RESULTS_DIR}")
    print(f"Budgets: {iters_list} | Models: {list(MODELS)} | "
          f"Scenarios: {[s['name'] for s in SCENARIOS]} | Seeds: {seeds}")
    print(f"Fixed: H={H_DEFAULT}, block={BS_DEFAULT}, lr={LR_DEFAULT}")
    print(f"Total: {total} experiments\n")

    config = {
        "experiment_type": "solver_budget_sensitivity",
        "version": "P0_SOLVER_BUDGET_NEWCKPT",
        "adam_iters": iters_list,
        "fixed": {"horizon": H_DEFAULT, "block_size": BS_DEFAULT, "lr": LR_DEFAULT},
        "models": {k: ("0.0" if "Fidelity" in k else "1.5e-3") for k in MODELS},
        "scenarios": [s["name"] for s in SCENARIOS],
        "seeds": seeds,
        "checkpoint_dir": str(NEW_RESULTS_DIR),
        "run_timestamp": datetime.now().isoformat(),
        "note": ("Iteration sweep at default horizon/block/lr to test whether "
                 "RAMC's ranking over Fidelity persists as the Adam solver budget "
                 "increases. New checkpoints, 5 seeds, for consistency with the "
                 "ablation and margin-grid runs."),
    }
    with open(raw_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    sim_cache = {}
    results_all = []
    start_total = time.time()

    for exp in experiments:
        rf = raw_dir / f"result_{exp['id']:03d}.json"
        if rf.exists():
            with open(rf) as f:
                cached = json.load(f)
            if cached.get("success"):
                results_all.append(cached)
                print(f"  EXP {exp['id']:3d}/{total} SKIP "
                      f"{exp['model_name']} | iter{exp['n_iter']} | "
                      f"{exp['scenario']} | s{exp['seed']}")
                continue

        print(f"\n  EXP {exp['id']}/{total}: {exp['model_name']} | "
              f"iter{exp['n_iter']} | {exp['scenario']} | seed={exp['seed']}")
        start = time.time()
        try:
            cache_key = (exp["model_name"], exp["n_iter"], exp["scenario"])
            if cache_key not in sim_cache:
                sim = ClosedLoopNMPCSimulator(
                    nn_model_path=exp["model_path"],
                    weather_truth_path=exp["scenario_truth"],
                    weather_forecast_path=exp["scenario_forecast"],
                    nmpc_horizon=H_DEFAULT,
                    nmpc_block_size=BS_DEFAULT,
                    nmpc_n_iter=exp["n_iter"],
                    nmpc_lr=LR_DEFAULT,
                    nmpc_grad_clip=NMPC_DEFAULTS["grad_clip"],
                    w_energy=W_ENERGY, w_cold=W_COLD, w_hot=W_HOT,
                    w_du=1e-3, w_terminal=W_TERMINAL, w_trust=0.0,
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
                initial_state=init_state, simulation_steps=SIMULATION_STEPS,
                verbose=False, log_interval=250,
            )
            elapsed = time.time() - start
            metrics = extract_metrics(sim_results)
            result = {
                "experiment_id": exp["id"], "model_name": exp["model_name"],
                "n_iter": exp["n_iter"], "scenario": exp["scenario"],
                "seed": exp["seed"], "success": True, "elapsed_s": elapsed,
                "metrics": metrics,
            }
            print(f"    CDH={metrics['deg_hours_cold_occ']:.2f} | "
                  f"Peak={metrics['peak_cold_violation_occ_C']:.2f} | "
                  f"Solve={metrics['median_solve_time_ms']:.0f}ms | {elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            result = {
                "experiment_id": exp["id"], "model_name": exp["model_name"],
                "n_iter": exp["n_iter"], "scenario": exp["scenario"],
                "seed": exp["seed"], "success": False, "error": str(e),
            }

        results_all.append(result)
        with open(rf, "w") as f:
            json.dump(result, f, indent=2)

        done = len([r for r in results_all if "success" in r])
        et = time.time() - start_total
        eta = (et / done) * (total - done) if done else 0
        print(f"    Progress: {done}/{total} | ETA: {eta/60:.1f}min")

    with open(raw_dir / "all_results.json", "w") as f:
        json.dump({"config": config, "results": results_all}, f, indent=2)

    analyze(results_all, output_dir)
    print(f"\n  Saved to: {output_dir}  ({(time.time()-start_total)/60:.1f} min)")
    return results_all


# =============================================================================
# Analysis: paired RAMC - Fidelity dCDH per (budget, scenario), with CI
# =============================================================================

def analyze(results, output_dir):
    ok = [r for r in results if r.get("success")]
    if not ok:
        print("  No successful results.")
        return

    # index[(model, n_iter, scenario, seed)] = metrics
    idx = {}
    for r in ok:
        idx[(r["model_name"], r["n_iter"], r["scenario"], r["seed"])] = r["metrics"]

    iters_present = sorted(set(r["n_iter"] for r in ok))
    scen_present = []
    for s in [sc["name"] for sc in SCENARIOS]:
        if any(r["scenario"] == s for r in ok):
            scen_present.append(s)
    seeds_present = sorted(set(r["seed"] for r in ok))

    rows = []
    for n_iter in iters_present:
        for scen in scen_present:
            # paired by seed
            dcdh, fid_cdh, ramc_cdh = [], [], []
            solve_fid, solve_ramc = [], []
            for seed in seeds_present:
                kf = ("Fidelity", n_iter, scen, seed)
                kr = ("RAMC_1.5e-3", n_iter, scen, seed)
                if kf in idx and kr in idx:
                    fc = idx[kf]["deg_hours_cold_occ"]
                    rc = idx[kr]["deg_hours_cold_occ"]
                    fid_cdh.append(fc); ramc_cdh.append(rc)
                    dcdh.append(rc - fc)
                    solve_fid.append(idx[kf]["median_solve_time_ms"])
                    solve_ramc.append(idx[kr]["median_solve_time_ms"])
            if not dcdh:
                continue
            arr = np.array(dcdh)
            mean_d, lo, hi = bootstrap_ci(arr, n_resamples=10_000)
            rows.append({
                "n_iter": n_iter, "scenario": scen, "n_seeds": len(dcdh),
                "fidelity_CDH_mean": float(np.mean(fid_cdh)),
                "ramc_CDH_mean": float(np.mean(ramc_cdh)),
                "delta_CDH_mean": float(mean_d),
                "delta_CDH_ci_lo": float(lo),
                "delta_CDH_ci_hi": float(hi),
                "ramc_advantage": bool(mean_d < 0),
                "ci_excludes_zero": bool(hi < 0 or lo > 0),
                "fidelity_solve_ms_median": float(np.median(solve_fid)),
                "ramc_solve_ms_median": float(np.median(solve_ramc)),
            })

    with open(output_dir / "sensitivity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 72)
    print("SOLVER-BUDGET SENSITIVITY  (paired RAMC - Fidelity, dCDH)")
    print("Negative dCDH favors RAMC. CI from 10k bootstrap over seeds.")
    print("=" * 72)
    for scen in scen_present:
        print(f"\n  {scen}:")
        print(f"    {'N_Adam':>7s} {'Fid CDH':>9s} {'RAMC CDH':>9s} "
              f"{'dCDH':>8s} {'95% CI':>20s}  verdict")
        for row in rows:
            if row["scenario"] != scen:
                continue
            ci = f"[{row['delta_CDH_ci_lo']:+.2f}, {row['delta_CDH_ci_hi']:+.2f}]"
            sig = "*" if row["ci_excludes_zero"] else " "
            verdict = "RAMC better" if row["ramc_advantage"] else "Fidelity better"
            print(f"    {row['n_iter']:>7d} {row['fidelity_CDH_mean']:>9.2f} "
                  f"{row['ramc_CDH_mean']:>9.2f} {row['delta_CDH_mean']:>+8.2f} "
                  f"{ci:>20s}{sig} {verdict}")

    # Trend headline
    print("\n  " + "-" * 68)
    for scen in scen_present:
        srows = sorted([r for r in rows if r["scenario"] == scen],
                       key=lambda r: r["n_iter"])
        if len(srows) >= 2:
            first, last = srows[0], srows[-1]
            trend = ("widens" if last["delta_CDH_mean"] < first["delta_CDH_mean"]
                     else "narrows")
            print(f"  {scen}: RAMC dCDH {first['delta_CDH_mean']:+.2f} "
                  f"(N={first['n_iter']}) -> {last['delta_CDH_mean']:+.2f} "
                  f"(N={last['n_iter']}); advantage {trend} with budget")
    print("  " + "-" * 68)
    print("\n  Interpretation: if the advantage persists or widens as N_Adam grows,")
    print("  the closed-loop benefit is not an artifact of the fixed solver budget.")
    print(f"\n  Saved: {output_dir / 'sensitivity_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMPC solver-budget sensitivity")
    parser.add_argument("--quick", action="store_true",
                        help="iters {25,100}, 1 seed, all models/scenarios")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output) if args.output else (THIS_DIR / f"results_solver_budget_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    run(out, quick=args.quick)
