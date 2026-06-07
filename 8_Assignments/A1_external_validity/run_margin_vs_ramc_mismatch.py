#!/usr/bin/env python3
"""
run_margin_vs_ramc_mismatch.py
===============================

Plant-mismatch stress test extending the P0.5 / R1.4 analysis.

QUESTION
--------
On the MATCHED plant, a static occupied-setpoint margin Pareto-dominates RAMC.
But a margin is a control-level constant bolted onto an average-accurate model,
whereas RAMC reshapes the model's dynamics under perturbation. The premise of
the paper is that real buildings deviate from the identified RC model. So the
decisive, not-yet-tested question is:

    When the evaluation plant deviates from the model used to generate the
    training data, does a margin tuned on the matched plant stay competitive,
    or does RAMC's model-level robustness produce an advantage the margin
    cannot replicate?

DESIGN
------
This reuses the Figure 9 perturbed-plant machinery (shared.perturbed_rc_plant)
and adds the margin baseline as an explicit competitor via the patched
comfort_margin_C parameter (default 0.0 -> no-op, verified bitwise-identical
on the ablation paths).

Models (all planned with the SAME NMPC; the only differences are the planning
model and the planning-time comfort margin):
  - Fidelity        : MSE model, margin 0.0      (Figure 9 reference, = margin 0)
  - RAMC_1.5e-3     : CVaR-trained model, margin 0.0   (model-level intervention)
  - Margin_0.3      : MSE model, comfort_margin_C = 0.3 (control-level)
  - Margin_0.5      : MSE model, comfort_margin_C = 0.5 (control-level)
  - Margin_0.8      : MSE model, comfort_margin_C = 0.8 (equal-energy match to
                      RAMC on the matched plant, forecast_error scenario)

Plants: the 14 variants from shared.perturbed_rc_plant
        (1 matched + 4 one-at-a-time + 8 random + 1 combined-severe).
Scenario: forecast_error (same as Figure 9 / run_a1.py).
Seeds: REDUCED_SEEDS (3), same as run_a1.py.

Total: 5 models x 14 plants x 3 seeds = 210 experiments.

Metrics are evaluated against the TRUE 20 degC occupied bound (the margin
shifts only the PLANNING bound), so CDH / peak / CVaR are directly comparable
across models.

CONSISTENCY NOTE
----------------
paths.py points CHECKPOINT_* at the OLD training dir (RAMC_FULL_cvar_20260307).
This script OVERRIDES to the NEW dir (RAMC_FULL_cvar_20260518_182822) so the
matched-plant point reproduces the ablation / margin-grid numbers exactly.

OUTPUT
------
results_margin_mismatch_{timestamp}/
    raw/result_*.json, all_results.json
    aggregate_by_category.csv      (model x plant-category means)
    ramc_vs_margin.csv             (paired RAMC - margin deltas per category, w/ CI)
    decision_summary.txt

Usage (place in the same assignments folder as run_a1.py so the shared.*
imports resolve, then run):
    python run_margin_vs_ramc_mismatch.py
    python run_margin_vs_ramc_mismatch.py --quick   # 1 plant subset, 1 seed
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    SCENARIO_DIR,
    REDUCED_SEEDS,
    NMPC_DEFAULTS, COMFORT_BAND,
    W_COLD, W_HOT, W_ENERGY, W_TERMINAL,
    STEPS_PER_EPISODE,
    setup_imports,
)
from shared.perturbed_rc_plant import (
    generate_all_plant_variants,
    validate_plant_variant,
    save_plant_variants,
)
from shared.stats_utils import bootstrap_ci

setup_imports()

from closed_loop_nmpc import ClosedLoopNMPCSimulator
from rc_ground_truth import RCGroundTruthModel
from shared_constants import ENERGY_COST_RATE


# =============================================================================
# Configuration
# =============================================================================

DT_SECONDS = 600
SIMULATION_STEPS = STEPS_PER_EPISODE  # 1008 (7 days)

# --- Override checkpoints to the NEW training dir (consistency with the
#     ablation + margin-grid runs). Adjust the absolute path if your tree
#     differs. ---
NEW_RESULTS_DIR = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
    r"\results\RAMC_FULL_cvar_20260518_182822"
)
CHECKPOINT_FIDELITY = NEW_RESULTS_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth"
CHECKPOINT_RAMC_15E4 = NEW_RESULTS_DIR / "RAMC_lambda_0.0015_op_cvar_rollout_a1.0_best.pth"

# Models. "margin" is the planning-time occupied lower-bound offset (degC).
# Margin models reuse the Fidelity (MSE) checkpoint.
MODELS = {
    "Fidelity":     {"path": str(CHECKPOINT_FIDELITY),  "lambda": 0.0,    "margin": 0.0},
    "RAMC_1.5e-3":  {"path": str(CHECKPOINT_RAMC_15E4), "lambda": 0.0015, "margin": 0.0},
    "Margin_0.3":   {"path": str(CHECKPOINT_FIDELITY),  "lambda": None,   "margin": 0.3},
    "Margin_0.5":   {"path": str(CHECKPOINT_FIDELITY),  "lambda": None,   "margin": 0.5},
    "Margin_0.8":   {"path": str(CHECKPOINT_FIDELITY),  "lambda": None,   "margin": 0.8},
}
MARGIN_MODELS = ["Margin_0.3", "Margin_0.5", "Margin_0.8"]

SCENARIO = {
    "name": "forecast_error",
    "truth": str(SCENARIO_DIR / "forecast_error_truth.csv"),
    "forecast": str(SCENARIO_DIR / "forecast_error_forecast.csv"),
}

SEEDS = REDUCED_SEEDS  # [42, 123, 456]

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


def extract_key_metrics(results: dict) -> dict:
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
# Runner
# =============================================================================

def run_experiments(output_dir, quick=False):
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Generating plant variants...")
    variants = generate_all_plant_variants(n_random=8, random_seed=42)
    valid = [v for v in variants if validate_plant_variant(v["params"], v["label"])]
    save_plant_variants(valid, output_dir / "plant_variants.json")
    print(f"  {len(valid)} valid plant variants")

    seeds = SEEDS
    if quick:
        valid = [valid[0], valid[-1]]  # matched + combined_severe
        seeds = [SEEDS[0]]
        print(f"  QUICK MODE: {len(valid)} plants x {len(seeds)} seed "
              f"x {len(MODELS)} models")

    experiments = []
    exp_id = 1
    for model_name, cfg in MODELS.items():
        for v in valid:
            for seed in seeds:
                experiments.append({
                    "id": exp_id, "model_name": model_name,
                    "model_path": cfg["path"], "model_lambda": cfg["lambda"],
                    "model_margin": cfg["margin"],
                    "plant_label": v["label"], "plant_category": v["category"],
                    "plant_params": v["params"], "seed": seed,
                })
                exp_id += 1
    total = len(experiments)
    print(f"\n[Step 2] Matrix: {len(MODELS)} models x {len(valid)} plants "
          f"x {len(seeds)} seeds = {total}")

    config = {
        "experiment_type": "margin_vs_ramc_plant_mismatch",
        "version": "P0.5_EXTENSION_MISMATCH",
        "models": {k: {"lambda": v["lambda"], "margin": v["margin"]}
                   for k, v in MODELS.items()},
        "scenario": SCENARIO["name"],
        "seeds": seeds,
        "n_plant_variants": len(valid),
        "checkpoint_dir": str(NEW_RESULTS_DIR),
        "run_timestamp": datetime.now().isoformat(),
        "note": ("Tests whether a matched-plant-tuned static margin stays "
                 "competitive with RAMC under RC plant mismatch. Margin applied "
                 "via comfort_margin_C (planning bound only); metrics evaluated "
                 "against the true 20 degC bound."),
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
                print(f"  EXP {exp['id']:3d}/{total} SKIP (cached) "
                      f"{exp['model_name']} | {exp['plant_label']} | s{exp['seed']}")
                continue

        print(f"\n{'='*64}\n  EXP {exp['id']}/{total}: {exp['model_name']} | "
              f"{exp['plant_label']} ({exp['plant_category']}) | seed={exp['seed']}")
        start = time.time()
        try:
            cache_key = (exp["model_name"], exp["plant_label"])
            if cache_key not in sim_cache:
                print("    Creating simulator...")
                sim = ClosedLoopNMPCSimulator(
                    nn_model_path=exp["model_path"],
                    weather_truth_path=SCENARIO["truth"],
                    weather_forecast_path=SCENARIO["forecast"],
                    comfort_margin_C=exp["model_margin"],   # NEW: planning margin
                    verbose_init=(len(sim_cache) == 0),
                    **NMPC_KWARGS,
                )
                # CRITICAL: replace plant with the perturbed variant
                sim.ground_truth = RCGroundTruthModel(
                    params=exp["plant_params"], dt_seconds=DT_SECONDS,
                )
                sim_cache[cache_key] = sim
            else:
                sim = sim_cache[cache_key]

            sim.reset_for_new_seed()
            torch.manual_seed(exp["seed"])
            init_state = create_initial_state(exp["seed"])

            sim_results = sim.simulate_episode(
                initial_state=init_state, simulation_steps=SIMULATION_STEPS,
                verbose=False, log_interval=200,
            )
            elapsed = time.time() - start
            metrics = extract_key_metrics(sim_results)
            result = {
                "experiment_id": exp["id"], "model_name": exp["model_name"],
                "model_lambda": exp["model_lambda"], "model_margin": exp["model_margin"],
                "plant_label": exp["plant_label"], "plant_category": exp["plant_category"],
                "seed": exp["seed"], "success": True, "elapsed_s": elapsed,
                "metrics": metrics,
            }
            print(f"    CDH={metrics['deg_hours_cold_occ']:.2f} | "
                  f"Peak={metrics['peak_cold_violation_occ_C']:.2f} | "
                  f"CVaR={metrics['cvar90_cold_occ_C']:.3f} | "
                  f"E={metrics['total_energy_kWh']:.0f} | {elapsed:.0f}s")
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            result = {
                "experiment_id": exp["id"], "model_name": exp["model_name"],
                "plant_label": exp["plant_label"], "plant_category": exp["plant_category"],
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
# Analysis: does RAMC beat the margins as mismatch grows?
# =============================================================================

CATEGORY_ORDER = ["matched", "one_at_a_time", "random_ensemble", "combined_severe"]


def analyze(results, output_dir):
    import csv
    ok = [r for r in results if r.get("success")]
    if not ok:
        print("  No successful results to analyze.")
        return

    # group[(model, category)] = list of metric dicts
    group = defaultdict(list)
    for r in ok:
        group[(r["model_name"], r["plant_category"])].append(r["metrics"])

    cats = [c for c in CATEGORY_ORDER
            if any(k[1] == c for k in group.keys())]

    # ---- aggregate_by_category.csv ----
    agg_rows = []
    for model_name in MODELS:
        for c in cats:
            ms = group.get((model_name, c), [])
            if not ms:
                continue
            row = {"model": model_name, "plant_category": c, "n": len(ms)}
            for m in METRICS:
                vals = [x[m] for x in ms]
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
            agg_rows.append(row)
    with open(output_dir / "aggregate_by_category.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)

    # ---- paired RAMC - margin deltas per category ----
    # Pair by (plant_label, seed). Negative dCDH => RAMC better (less discomfort).
    def by_plant_seed(model_name):
        d = {}
        for r in ok:
            if r["model_name"] == model_name:
                d[(r["plant_label"], r["seed"])] = r["metrics"]
        return d

    ramc = by_plant_seed("RAMC_1.5e-3")
    cmp_rows = []
    for margin_model in MARGIN_MODELS:
        mg = by_plant_seed(margin_model)
        # organize keys by category
        cat_of = {(r["plant_label"], r["seed"]): r["plant_category"] for r in ok}
        for c in cats:
            keys = [k for k in ramc if k in mg and cat_of.get(k) == c]
            if not keys:
                continue
            row = {"margin_model": margin_model, "plant_category": c,
                   "n_pairs": len(keys)}
            for m in METRICS:
                deltas = np.array([ramc[k][m] - mg[k][m] for k in keys])
                mean_d, lo, hi = bootstrap_ci(deltas, n_resamples=10_000)
                row[f"d_{m}_mean"] = float(mean_d)
                row[f"d_{m}_ci_lo"] = float(lo)
                row[f"d_{m}_ci_hi"] = float(hi)
                row[f"d_{m}_ramc_better"] = int(np.sum(deltas < 0))
            cmp_rows.append(row)
    with open(output_dir / "ramc_vs_margin.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader(); w.writerows(cmp_rows)

    # ---- decision summary ----
    lines = []
    lines.append("=" * 72)
    lines.append("PLANT-MISMATCH DECISION SUMMARY")
    lines.append("Does RAMC's comfort position vs the margins improve as the")
    lines.append("plant deviates from the identified model?")
    lines.append("=" * 72)

    # CDH means per model per category (the headline comfort metric)
    lines.append("\nMean cold degree-hours (CDH) by model x plant category:")
    header = f"  {'model':<14s}" + "".join(f"{c[:10]:>13s}" for c in cats)
    lines.append(header)
    for model_name in MODELS:
        cells = []
        for c in cats:
            ms = group.get((model_name, c), [])
            cells.append(f"{np.mean([x['deg_hours_cold_occ'] for x in ms]):>13.2f}"
                         if ms else f"{'-':>13s}")
        lines.append(f"  {model_name:<14s}" + "".join(cells))

    lines.append("\nRAMC minus margin, CDH (negative => RAMC better), by category:")
    for margin_model in MARGIN_MODELS:
        lines.append(f"\n  vs {margin_model}:")
        for row in cmp_rows:
            if row["margin_model"] != margin_model:
                continue
            d = row["d_deg_hours_cold_occ_mean"]
            lo = row["d_deg_hours_cold_occ_ci_lo"]
            hi = row["d_deg_hours_cold_occ_ci_hi"]
            verdict = "RAMC better" if d < 0 else "margin better"
            sig = "" if (lo <= 0 <= hi) else "  (95% CI excludes 0)"
            lines.append(f"    {row['plant_category']:<18s} "
                         f"dCDH={d:>+7.2f} [{lo:>+6.2f},{hi:>+6.2f}] {verdict}{sig}")

    # Headline: for each margin, does RAMC go from worse->better as mismatch grows?
    lines.append("\n" + "-" * 72)
    lines.append("HEADLINE")
    for margin_model in MARGIN_MODELS:
        rows = {r["plant_category"]: r for r in cmp_rows
                if r["margin_model"] == margin_model}
        d_matched = rows.get("matched", {}).get("d_deg_hours_cold_occ_mean")
        d_severe = rows.get("combined_severe", {}).get("d_deg_hours_cold_occ_mean")
        if d_matched is None or d_severe is None:
            continue
        trend = "improves" if d_severe < d_matched else "worsens"
        flips = (d_matched > 0 and d_severe < 0)
        msg = (f"  vs {margin_model}: RAMC's CDH gap goes {d_matched:+.2f} (matched) "
               f"-> {d_severe:+.2f} (severe); RAMC position {trend} under mismatch")
        if flips:
            msg += "  *** RAMC OVERTAKES the margin under severe mismatch ***"
        lines.append(msg)
    lines.append("-" * 72)
    lines.append("\nInterpretation:")
    lines.append("  If RAMC overtakes a margin only under severe mismatch, the")
    lines.append("  model-level robustness is a genuine, margin-proof contribution.")
    lines.append("  If the margins stay ahead across ALL categories, concede and")
    lines.append("  reposition RAMC as a model-training method (P0.5 result stands).")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(output_dir / "decision_summary.txt", "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Margin vs RAMC under plant mismatch")
    parser.add_argument("--quick", action="store_true",
                        help="2 plants (matched + severe), 1 seed, all 5 models")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output) if args.output else (THIS_DIR / f"results_margin_mismatch_{ts}")
    out.mkdir(parents=True, exist_ok=True)

    print("#" * 70)
    print("MARGIN vs RAMC UNDER PLANT MISMATCH (P0.5 extension)")
    print("#" * 70)
    print(f"Output: {out}")
    print(f"Checkpoints: {NEW_RESULTS_DIR}")

    run_experiments(out, quick=args.quick)
