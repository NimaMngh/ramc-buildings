#!/usr/bin/env python3
"""
run_margin_grid.py
==================

Static-margin baseline at a grid of margins, for the fair Pareto comparison
the Revision Plan requires (R1.4).

For each margin m, the SAME Fidelity_Baseline_rollout checkpoint is used,
but the NMPC planner's occupied lower comfort bound is raised:
    T_min^MPC = T_min + m
This uses the comfort_margin_C parameter of ClosedLoopNMPCSimulator
(additive, default 0.0, verified to be a no-op on the ablation paths).
Post-simulation metrics are computed against the TRUE 20 degC occupied
bound, so cold-violation metrics stay comparable across margins.

Grid:      m in {0.1, 0.2, ..., 1.0} degC (m=0.0 == A0 Fidelity, already
           available from the ablation run, so it is skipped here).
Scenarios: forecast_error, cold_snap (same two as the ablation).
Seeds:     same 5 paired seeds as the main paper.
Total:     10 margins x 2 scenarios x 5 seeds = 100 experiments.

Outputs (same layout as run_nmpc_matrix.py):
    results_NMPC_margin_grid_{timestamp}/matrix/
        aggregate_metrics.csv
        all_results.json
        config.json
        result_exp*.json
        traj_exp*.json

Usage (run from the M6_NMPC_Direct_Shooting directory):
    python run_margin_grid.py            # full grid (100 experiments)
    python run_margin_grid.py --quick    # 3 margins, 1 seed, 2 days (smoke)
    python run_margin_grid.py --resume N # resume from experiment N
"""

import matplotlib
matplotlib.use("Agg")

import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Reuse all shared config + helpers from the matrix runner so that metric
# extraction is byte-identical between the ablation and the margin grid.
from run_nmpc_matrix import (
    _extract_metrics,
    _get_nmpc_kwargs,
    create_initial_state,
    RAMC_MODELS_DIR,
    WEATHER_DIR,
    SEEDS,
    RC_PARAMS,
    DT_SECONDS,
    SIMULATION_STEPS,
    STEPS_PER_DAY,
)
from closed_loop_nmpc import ClosedLoopNMPCSimulator
from rc_ground_truth import RCGroundTruthModel

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Single model: the fidelity baseline; the margin is applied at runtime ──
FIDELITY_MODEL = {
    "name": "Fidelity_Baseline_rollout",
    "path": str(RAMC_MODELS_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth"),
}

# ── Margin grid (degC). m=0.0 equals A0, already in the ablation results. ──
MARGINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ── Scenarios: same two as the ablation ──
SCENARIOS = [
    {"name": "forecast_error",
     "truth_file": str(WEATHER_DIR / "forecast_error_truth.csv"),
     "forecast_file": str(WEATHER_DIR / "forecast_error_forecast.csv")},
    {"name": "cold_snap",
     "truth_file": str(WEATHER_DIR / "cold_snap_truth.csv"),
     "forecast_file": str(WEATHER_DIR / "cold_snap_forecast.csv")},
]


def margin_label(m: float) -> str:
    """Label matching the existing plotting convention (Fidelity_margin_0.3)."""
    return f"Fidelity_margin_{m:.1f}"


def run_margin_grid(output_dir, resume_from=None,
                    margins=None, seeds=None, sim_steps=None):
    margins = margins if margins is not None else MARGINS
    seeds = seeds if seeds is not None else SEEDS
    sim_steps = sim_steps if sim_steps is not None else SIMULATION_STEPS

    matrix_dir = output_dir / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    exp_id = 1
    for m in margins:
        for scenario in SCENARIOS:
            for seed in seeds:
                experiments.append({
                    "id": exp_id, "margin": m,
                    "scenario": scenario, "seed": seed,
                })
                exp_id += 1
    total = len(experiments)

    print(f"Margins: {len(margins)}, Scenarios: {len(SCENARIOS)}, Seeds: {len(seeds)}")
    print(f"Total: {total} experiments")
    print(f"Margin values: {margins}")

    config = {
        "experiment_type": "nmpc_margin_grid",
        "experiment_version": "R1_4_PARETO_MARGIN",
        "model": FIDELITY_MODEL["name"],
        "margins_C": list(margins),
        "scenarios": [s["name"] for s in SCENARIOS],
        "seeds": list(seeds),
        "simulation_steps": sim_steps,
        "rc_params": RC_PARAMS,
        "run_timestamp": datetime.now().isoformat(),
        "note": (
            "Static-margin baseline for the fair Pareto comparison (R1.4). "
            "Same Fidelity_Baseline_rollout checkpoint at every margin; the "
            "NMPC occupied lower comfort bound is raised by m via "
            "comfort_margin_C. Metrics are evaluated against the true 20 degC "
            "occupied bound so cold-violation metrics stay comparable."
        ),
    }
    with open(matrix_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    nmpc_kwargs = _get_nmpc_kwargs()

    # Simulator cache keyed by (margin, scenario). Each entry sets its own
    # comfort_margin_C at construction.
    sim_cache = {}

    def get_simulator(m, scenario_cfg):
        key = (m, scenario_cfg["name"])
        if key not in sim_cache:
            print(f"  Creating simulator: margin=+{m:.1f}C, {scenario_cfg['name']}")
            sim = ClosedLoopNMPCSimulator(
                nn_model_path=FIDELITY_MODEL["path"],
                weather_truth_path=scenario_cfg["truth_file"],
                weather_forecast_path=scenario_cfg["forecast_file"],
                comfort_margin_C=m,                       # <- the margin
                verbose_init=(len(sim_cache) == 0),
                **nmpc_kwargs,
            )
            sim.ground_truth = RCGroundTruthModel(
                params=RC_PARAMS, dt_seconds=DT_SECONDS
            )
            sim_cache[key] = sim
        return sim_cache[key]

    results = []
    start_total = time.time()

    for exp in experiments:
        if resume_from and exp["id"] < resume_from:
            rf = matrix_dir / f"result_exp{exp['id']}.json"
            if rf.exists():
                with open(rf) as f:
                    results.append(json.load(f))
            continue

        print(f"\n{'='*60}")
        print(f"EXP {exp['id']}/{total}: margin=+{exp['margin']:.1f}C | "
              f"{exp['scenario']['name']} | seed={exp['seed']}")
        start = time.time()
        try:
            sim = get_simulator(exp["margin"], exp["scenario"])
            sim.reset_for_new_seed()
            torch.manual_seed(exp["seed"])
            init_state = create_initial_state(exp["seed"])

            sim_results = sim.simulate_episode(
                initial_state=init_state,
                simulation_steps=sim_steps,
                verbose=True,
                log_interval=100,
            )
            elapsed = time.time() - start
            metrics = _extract_metrics(sim_results)
            metrics["initial_state"] = init_state.tolist()

            result = {
                "experiment_id": exp["id"],
                "model": margin_label(exp["margin"]),
                "margin_C": exp["margin"],
                "scenario": exp["scenario"]["name"],
                "seed": exp["seed"],
                "success": True,
                "elapsed_s": elapsed,
                "metrics": metrics,
            }

            traj = {
                "T_air": sim_results["states"][:, 0].tolist(),
                "T_ret": sim_results["states"][:, 5].tolist(),
                "T_supply": sim_results["controls"][:, 0].tolist(),
                "mdot": sim_results["controls"][:, 1].tolist(),
                "Tmin": sim_results["Tmin_series"].tolist(),
                "Tmax": sim_results["Tmax_series"].tolist(),
                "occupancy": sim_results["occupancy_series"].tolist(),
                "energy_kWh": sim_results["energy_kWh_step"],
            }
            with open(matrix_dir / f"traj_exp{exp['id']}.json", "w") as f:
                json.dump(traj, f)

            print(f"  OK  E={metrics['total_energy_kWh']:.0f}kWh | "
                  f"CDH={metrics['deg_hours_cold_occ']:.3f} | "
                  f"Peak={metrics['peak_cold_violation_occ_C']:.2f} | "
                  f"CVaR90={metrics['cvar90_cold_occ_C']:.3f} | {elapsed:.0f}s")

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            result = {
                "experiment_id": exp["id"],
                "model": margin_label(exp["margin"]),
                "margin_C": exp["margin"],
                "scenario": exp["scenario"]["name"],
                "seed": exp["seed"],
                "success": False,
                "error": str(e),
            }

        results.append(result)
        with open(matrix_dir / f"result_exp{exp['id']}.json", "w") as f:
            json.dump(result, f, indent=2)

        done = len([r for r in results if "success" in r])
        elapsed_total = time.time() - start_total
        eta = (elapsed_total / done) * (total - done) if done else 0
        print(f"  Progress: {done}/{total} | ETA: {eta/60:.1f}min")

    with open(matrix_dir / "all_results.json", "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)

    _aggregate(results, matrix_dir)

    elapsed_total = time.time() - start_total
    n_ok = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"COMPLETE: {n_ok}/{total} in {elapsed_total/60:.1f}min")
    print(f"Saved to: {matrix_dir}")
    return results


def _aggregate(results, output_dir):
    successful = [r for r in results if r.get("success")]
    if not successful:
        return
    margins = sorted(set(r["margin_C"] for r in successful))
    scenarios = sorted(set(r["scenario"] for r in successful))
    key_metrics = [
        "total_energy_kWh", "deg_hours_cold_occ",
        "cvar90_cold_occ_C", "cvar95_cold_occ_C",
        "peak_cold_violation_occ_C", "hours_cold_outside_occ",
    ]
    rows = []
    for m in margins:
        for sc in scenarios:
            exps = [r for r in successful
                    if r["margin_C"] == m and r["scenario"] == sc]
            if not exps:
                continue
            row = {"model": f"Fidelity_margin_{m:.1f}", "margin_C": m,
                   "scenario": sc, "n_seeds": len(exps)}
            for metric in key_metrics:
                vals = [r["metrics"][metric] for r in exps]
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    print(f"  Saved: {output_dir / 'aggregate_metrics.csv'}")

    for sc in scenarios:
        print(f"\n  Scenario: {sc}")
        print(f"  {'Margin':>8s} {'Energy':>10s} {'CDH':>8s} "
              f"{'CVaR90':>8s} {'Peak':>8s}")
        print(f"  {'-'*46}")
        for m in margins:
            sub = [r for r in successful
                   if r["margin_C"] == m and r["scenario"] == sc]
            if not sub:
                continue
            e = np.mean([r["metrics"]["total_energy_kWh"] for r in sub])
            dh = np.mean([r["metrics"]["deg_hours_cold_occ"] for r in sub])
            cv = np.mean([r["metrics"]["cvar90_cold_occ_C"] for r in sub])
            pk = np.mean([r["metrics"]["peak_cold_violation_occ_C"] for r in sub])
            print(f"  {'+'+format(m,'.1f'):>8s} {e:>10.1f} {dh:>8.3f} "
                  f"{cv:>8.4f} {pk:>8.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pareto margin grid (R1.4)")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--resume", "-r", type=int, default=None)
    parser.add_argument("--quick", "-q", action="store_true")
    args = parser.parse_args()

    margins = MARGINS
    seeds = SEEDS
    sim_steps = SIMULATION_STEPS
    if args.quick:
        margins = [0.3, 0.5, 1.0]
        seeds = [42]
        sim_steps = 2 * STEPS_PER_DAY
        print(f"QUICK MODE: {len(margins)} margins, 1 seed, 2 days "
              f"= {len(margins) * len(SCENARIOS)} experiments")

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SCRIPT_DIR / f"results_NMPC_margin_grid_{ts}"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print("PARETO MARGIN GRID (R1.4)")
    print(f"{'#'*70}")
    print(f"Output: {output_dir}")
    print(f"Model: {FIDELITY_MODEL['name']} (margin applied at NMPC level)")

    run_margin_grid(output_dir, resume_from=args.resume,
                    margins=margins, seeds=seeds, sim_steps=sim_steps)
