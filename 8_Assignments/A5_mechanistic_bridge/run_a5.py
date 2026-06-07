"""
Assignment 5 — Mechanistic Bridge: Gradient Alignment Analysis
================================================================

Tests whether RAMC improves planning-relevant model behaviour — specifically
gradient alignment with the true plant — not just one-step prediction stats.

Methodology:
  1. Reconstruct full 6-state trajectories from stored controls + weather
     by re-simulating with the RC plant
  2. Sample N operating points from the Fidelity model's closed-loop trajectory
     (this represents the states the controller actually visits)
  3. At each point, compute cost gradients through both the NN and RC plant
  4. Measure cosine similarity, stratified by comfort-critical region

Three models compared:
  - Fidelity Baseline (λ=0): the primary comparator
  - RAMC λ=5e-4: the harmful-regime model (best open-loop, worst closed-loop)
  - RAMC λ=1.5e-3: the selected model (best closed-loop)

Expected outcome (from Assignment 5 document):
  - λ=5e-4 should show DEGRADED gradient alignment (explaining why more
    Adam iterations make it worse — the optimiser follows misleading gradients)
  - λ=1.5e-3 should show EQUAL OR BETTER alignment than Fidelity,
    especially near the comfort boundary

Run:
    %cd <project_root>/8_Assignments
    %run A5_mechanistic_bridge/run_a5.py
"""

import matplotlib
matplotlib.use("Agg")

import sys
import json
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Imports from shared infrastructure ──
THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    CHECKPOINT_FIDELITY, CHECKPOINT_RAMC_5E4, CHECKPOINT_RAMC_15E4,
    SCENARIO_DIR, NMPC_ALL_RESULTS_JSON,
    NMPC_RESULTS_DIR, A5_DIR, get_results_dir,
    NMPC_DEFAULTS, setup_imports,
)
from shared.gradient_alignment import (
    analyse_single_point,
    cosine_similarity,
)

setup_imports()

from rc_ground_truth import RCGroundTruthModel
from load_ramc_model import load_ramc_model
from shared_constants import (
    get_occupancy_status, get_comfort_bounds,
    ENERGY_COST_RATE,
    Q_INTERNAL_OCCUPIED_W, Q_INTERNAL_UNOCCUPIED_W,
)

# =============================================================================
# Configuration
# =============================================================================

MODELS_TO_ANALYSE = {
    "Fidelity": CHECKPOINT_FIDELITY,
    "RAMC_5e-4": CHECKPOINT_RAMC_5E4,
    "RAMC_1.5e-3": CHECKPOINT_RAMC_15E4,
}

# Operating points are sampled from Fidelity's forecast_error trajectory (seed 42)
# This is the trajectory the controller actually visits
REFERENCE_TRAJECTORY_EXP = "exp1"  # Fidelity, nominal, seed 42

# Number of operating points to sample
N_POINTS = 40  # Manageable: ~40 × 3 models × ~1s each = ~2 min total

# NMPC settings for gradient computation
BLOCK_SIZE = NMPC_DEFAULTS["block_size"]   # 4
HORIZON = NMPC_DEFAULTS["horizon"]         # 24
N_BLOCKS = HORIZON // BLOCK_SIZE           # 6


# =============================================================================
# Step 1: Reconstruct full state trajectory from stored data
# =============================================================================

def reconstruct_trajectory(
    traj_data: dict,
    weather_df: pd.DataFrame,
    rc_params: dict = None,
) -> np.ndarray:
    """
    Re-simulate the RC plant using stored controls to get full 6-state vectors.

    The trajectory files only store T_air and T_ret. We need all 6 states
    for gradient computation through both NN and RC models.

    Returns: (T+1, 6) state trajectory
    """
    plant = RCGroundTruthModel(params=rc_params, dt_seconds=600)

    T_supply = np.array(traj_data["T_supply"])
    mdot = np.array(traj_data["mdot"])
    T = len(T_supply)

    # Initial state: we know T_air[0] and T_ret[0], estimate others
    T_air_0 = traj_data["T_air"][0]
    T_ret_0 = traj_data["T_ret"][0]
    # Reasonable estimates for unmeasured states
    x0 = np.array([
        T_air_0,
        T_air_0 - 2.0,       # T_env slightly below T_air
        T_air_0 - 1.0,       # T_int close to T_air
        (T_supply[0] + T_ret_0) / 2,  # T_rad1
        (T_supply[0] + T_ret_0 * 2) / 3,  # T_rad2
        T_ret_0,              # T_ret
    ])

    states = np.zeros((T + 1, 6))
    states[0] = x0

    for k in range(T):
        u = np.array([T_supply[k], mdot[k]])

        # Build disturbance from weather
        if k < len(weather_df):
            row = weather_df.iloc[k]
            d = np.array([
                float(row["T_out_C"]),
                float(row["Q_solar_W"]),
                float(row["Q_internal_W"]),
            ])
        else:
            d = np.array([-10.0, 0.0, 30000.0])

        states[k + 1] = plant.step(states[k], u, d)

    return states


# =============================================================================
# Step 2: Sample operating points
# =============================================================================

def sample_operating_points(
    states: np.ndarray,
    controls: np.ndarray,
    weather_df: pd.DataFrame,
    traj_data: dict,
    n_points: int = 40,
    seed: int = 42,
) -> list:
    """
    Sample operating points from the reconstructed trajectory.

    Stratified sampling: ~50% from comfort-critical region (T_air < Tmin + 1°C),
    ~50% from non-critical region.

    Each operating point includes:
      - x0: full 6-state vector
      - U_blocked: (n_blocks, 2) control sequence (current + horizon, blocked)
      - D_horizon: (H, 3) disturbance forecast
      - Tmin_seq, Tmax_seq: comfort bounds for horizon
    """
    T = len(controls)
    rng = np.random.RandomState(seed)

    Tmin_arr = np.array(traj_data["Tmin"])
    Tmax_arr = np.array(traj_data["Tmax"])

    # Classify steps
    T_air = states[:-1, 0]  # exclude final state (no control)
    comfort_margin = T_air - Tmin_arr
    critical_mask = comfort_margin < 1.0
    noncritical_mask = ~critical_mask

    # Ensure we don't sample from the last H steps (need full horizon)
    valid_mask = np.zeros(T, dtype=bool)
    valid_mask[:T - HORIZON - 1] = True

    critical_indices = np.where(valid_mask & critical_mask)[0]
    noncritical_indices = np.where(valid_mask & noncritical_mask)[0]

    # Stratified sample
    n_critical = min(n_points // 2, len(critical_indices))
    n_noncritical = min(n_points - n_critical, len(noncritical_indices))

    if n_critical > 0:
        sampled_critical = rng.choice(critical_indices, n_critical, replace=False)
    else:
        sampled_critical = np.array([], dtype=int)

    if n_noncritical > 0:
        sampled_noncritical = rng.choice(noncritical_indices, n_noncritical, replace=False)
    else:
        sampled_noncritical = np.array([], dtype=int)

    sampled_indices = np.sort(np.concatenate([sampled_critical, sampled_noncritical]))

    print(f"  Sampled {len(sampled_indices)} operating points: "
          f"{n_critical} comfort-critical, {n_noncritical} non-critical")

    # Build operating point data
    points = []
    for k in sampled_indices:
        k = int(k)
        x0 = states[k].copy()

        # Build blocked control sequence from current trajectory
        U_blocked = np.zeros((N_BLOCKS, 2))
        for b in range(N_BLOCKS):
            start_h = k + b * BLOCK_SIZE
            end_h = min(start_h + BLOCK_SIZE, T)
            if start_h < T:
                U_blocked[b, 0] = np.mean(controls[start_h:end_h, 0])
                U_blocked[b, 1] = np.mean(controls[start_h:end_h, 1])
            else:
                U_blocked[b] = U_blocked[b - 1] if b > 0 else [45.0, 0.5]

        # Build disturbance horizon
        D_horizon = np.zeros((HORIZON, 3))
        for h in range(HORIZON):
            idx = k + h
            if idx < len(weather_df):
                row = weather_df.iloc[idx]
                D_horizon[h] = [
                    float(row["T_out_C"]),
                    float(row["Q_solar_W"]),
                    float(row["Q_internal_W"]),
                ]
            else:
                D_horizon[h] = D_horizon[h - 1] if h > 0 else [-10.0, 0.0, 30000.0]

        # Comfort bounds for horizon
        Tmin_seq = np.zeros(HORIZON)
        Tmax_seq = np.zeros(HORIZON)
        for h in range(HORIZON):
            idx = k + h
            if idx < len(Tmin_arr):
                Tmin_seq[h] = Tmin_arr[idx]
                Tmax_seq[h] = Tmax_arr[idx]
            else:
                Tmin_seq[h] = 20.0
                Tmax_seq[h] = 22.0

        points.append({
            "step_index": k,
            "x0": x0,
            "U_blocked": U_blocked,
            "D_horizon": D_horizon,
            "Tmin_seq": Tmin_seq,
            "Tmax_seq": Tmax_seq,
            "comfort_critical": bool(comfort_margin[k] < 1.0),
            "comfort_margin": float(comfort_margin[k]),
        })

    return points


# =============================================================================
# Step 3: Run gradient alignment analysis
# =============================================================================

def run_gradient_analysis(
    models: dict,
    operating_points: list,
    rc_plant,
) -> list:
    """
    Compute gradient alignment at each operating point for each model.
    """
    results = []

    for model_name, nn_model in models.items():
        print(f"\n  Analysing: {model_name}")
        nn_model.eval()

        for i, pt in enumerate(operating_points):
            if i % 10 == 0:
                print(f"    Point {i+1}/{len(operating_points)}...")

            try:
                analysis = analyse_single_point(
                    nn_model=nn_model,
                    rc_plant=rc_plant,
                    x0=pt["x0"],
                    U_blocked=pt["U_blocked"],
                    D_horizon=pt["D_horizon"],
                    Tmin_seq=pt["Tmin_seq"],
                    Tmax_seq=pt["Tmax_seq"],
                    block_size=BLOCK_SIZE,
                    w_cold=63.0,
                    w_energy=0.9,
                    w_terminal=20.0,
                )

                results.append({
                    "model": model_name,
                    "step_index": pt["step_index"],
                    "comfort_critical": pt["comfort_critical"],
                    "comfort_margin": pt["comfort_margin"],
                    **{k: v for k, v in analysis.items()
                       if k not in ("grad_nn", "grad_rc")},
                    # Store full gradients only for debugging if needed
                })

            except Exception as e:
                print(f"    WARNING: Point {i} failed for {model_name}: {e}")
                results.append({
                    "model": model_name,
                    "step_index": pt["step_index"],
                    "comfort_critical": pt["comfort_critical"],
                    "comfort_margin": pt["comfort_margin"],
                    "cos_similarity": float("nan"),
                    "error": str(e),
                })

    return results


# =============================================================================
# Step 4: Summarise and report
# =============================================================================

def summarise_results(results: list):
    """Print stratified summary table and return summary dict."""
    print(f"\n{'='*75}")
    print("  Gradient Alignment Summary (cosine similarity with RC plant)")
    print(f"{'='*75}")

    models = sorted(set(r["model"] for r in results))
    summary = []

    # Overall
    print(f"\n  {'Model':<20s} {'Region':<18s} {'n':>4s} {'cos(θ) mean':>12s} "
          f"{'cos(θ) med':>11s} {'cos>0':>7s}")
    print("  " + "-" * 74)

    for model in models:
        for region, label in [("all", "All"), (True, "Comfort-critical"),
                              (False, "Non-critical")]:
            if region == "all":
                subset = [r for r in results
                          if r["model"] == model
                          and np.isfinite(r.get("cos_similarity", float("nan")))]
            else:
                subset = [r for r in results
                          if r["model"] == model
                          and r.get("comfort_critical") == region
                          and np.isfinite(r.get("cos_similarity", float("nan")))]

            if not subset:
                continue

            cos_vals = np.array([r["cos_similarity"] for r in subset])
            n = len(cos_vals)
            mean_cos = float(np.mean(cos_vals))
            med_cos = float(np.median(cos_vals))
            pct_positive = float(np.mean(cos_vals > 0) * 100)

            print(f"  {model:<20s} {label:<18s} {n:>4d} "
                  f"{mean_cos:>+12.4f} {med_cos:>+11.4f} {pct_positive:>6.1f}%")

            summary.append({
                "model": model,
                "region": label,
                "n": n,
                "cos_mean": mean_cos,
                "cos_median": med_cos,
                "cos_std": float(np.std(cos_vals)),
                "cos_p25": float(np.percentile(cos_vals, 25)),
                "cos_p75": float(np.percentile(cos_vals, 75)),
                "pct_positive": pct_positive,
            })

        print()  # blank line between models

    return summary


def generate_figures(results: list, summary: list, out_dir: Path):
    """Generate publication-quality gradient alignment figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    models = sorted(set(r["model"] for r in results))
    colors = {"Fidelity": "#3498db", "RAMC_5e-4": "#9b59b6", "RAMC_1.5e-3": "#27ae60"}

    # ── Figure 1: Cosine similarity vs comfort margin, all 3 models ──
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    for model in models:
        subset = [r for r in results if r["model"] == model
                  and np.isfinite(r.get("cos_similarity", float("nan")))]
        if not subset:
            continue
        margins = [r["comfort_margin"] for r in subset]
        cos_vals = [r["cos_similarity"] for r in subset]
        ax.scatter(margins, cos_vals, alpha=0.6, s=40,
                   color=colors.get(model, "gray"), label=model, edgecolors="k",
                   linewidths=0.3)

    ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1,
               label="Comfort-critical threshold")
    ax.set_xlabel("Comfort margin: T_air − T_min (°C)", fontsize=11)
    ax.set_ylabel("Cosine similarity (NN vs RC gradient)", fontsize=11)
    ax.set_title("Gradient Alignment: NN Planning Models vs RC Ground Truth", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"gradient_alignment_scatter.{ext}", dpi=200)
    plt.close(fig)

    # ── Figure 2: Box plot by model and region ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (critical, title) in zip(axes, [(True, "Comfort-Critical (margin < 1°C)"),
                                             (False, "Non-Critical (margin ≥ 1°C)")]):
        data = []
        labels = []
        box_colors = []
        for model in models:
            subset = [r["cos_similarity"] for r in results
                      if r["model"] == model
                      and r.get("comfort_critical") == critical
                      and np.isfinite(r.get("cos_similarity", float("nan")))]
            data.append(subset if subset else [0])
            labels.append(model)
            box_colors.append(colors.get(model, "gray"))

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)

        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Cosine similarity" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Gradient Alignment by Comfort Region", fontsize=13, fontweight="bold")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"gradient_alignment_boxplot.{ext}", dpi=200)
    plt.close(fig)

    print(f"  Saved figures to {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  Assignment 5 — Mechanistic Bridge: Gradient Alignment Analysis")
    print("#" * 70)

    out_dir = get_results_dir(A5_DIR)
    fig_dir = get_results_dir(A5_DIR, "figures")

    # ── Load trajectory data (Fidelity, nominal, seed 42) ──
    print("\n[Step 1] Loading reference trajectory...")
    traj_dir = NMPC_RESULTS_DIR / "matrix"

    # Find the Fidelity forecast_error seed=42 trajectory
    with open(NMPC_ALL_RESULTS_JSON) as f:
        all_data = json.load(f)

    # Find experiment ID for Fidelity + forecast_error + seed 42
    target_exp = None
    for exp in all_data["results"]:
        if (exp["model"] == "Fidelity_Baseline_rollout"
                and exp["scenario"] == "forecast_error"
                and exp["seed"] == 42
                and exp["success"]):
            target_exp = exp
            break

    if target_exp is None:
        # Fallback: use first successful Fidelity experiment
        for exp in all_data["results"]:
            if exp["model"] == "Fidelity_Baseline_rollout" and exp["success"]:
                target_exp = exp
                break

    exp_id = target_exp["experiment_id"]
    print(f"  Using experiment {exp_id}: {target_exp['model']} | "
          f"{target_exp['scenario']} | seed={target_exp['seed']}")

    traj_path = traj_dir / f"traj_exp{exp_id}.json"
    with open(traj_path) as f:
        traj_data = json.load(f)

    # Load weather data for this scenario
    scenario_name = target_exp["scenario"]
    weather_truth = pd.read_csv(SCENARIO_DIR / f"{scenario_name}_truth.csv")
    print(f"  Weather: {scenario_name}, {len(weather_truth)} steps")

    # ── Reconstruct full state trajectory ──
    print("\n[Step 2] Reconstructing full 6-state trajectory via RC plant...")
    states = reconstruct_trajectory(traj_data, weather_truth)
    controls = np.column_stack([traj_data["T_supply"], traj_data["mdot"]])
    print(f"  States shape: {states.shape}, Controls shape: {controls.shape}")

    # ── Sample operating points ──
    print("\n[Step 3] Sampling operating points (stratified)...")
    points = sample_operating_points(
        states, controls, weather_truth, traj_data,
        n_points=N_POINTS, seed=42,
    )

    # ── Load NN models ──
    print("\n[Step 4] Loading NN models...")
    nn_models = {}
    for name, path in MODELS_TO_ANALYSE.items():
        print(f"  Loading {name}...")
        model = load_ramc_model(path, device="cpu", dtype=torch.float64, verbose=False)
        model.eval()
        nn_models[name] = model

    # ── Create RC plant ──
    rc_plant = RCGroundTruthModel(dt_seconds=600)

    # ── Run gradient analysis ──
    print("\n[Step 5] Computing gradient alignment...")
    start_time = time.time()
    results = run_gradient_analysis(nn_models, points, rc_plant)
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    # ── Summarise ──
    summary = summarise_results(results)

    # ── Save ──
    # Remove non-serializable items
    results_save = [{k: v for k, v in r.items()} for r in results]
    with open(out_dir / "gradient_alignment_results.json", "w") as f:
        json.dump(results_save, f, indent=2, default=str)

    with open(out_dir / "gradient_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: gradient_alignment_results.json, gradient_alignment_summary.json")

    # ── Figures ──
    print("\n[Step 6] Generating figures...")
    generate_figures(results, summary, fig_dir)

    # ── Interpretation ──
    print(f"\n{'='*70}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("  If λ=1.5e-3 has HIGHER cos(θ) than Fidelity in comfort-critical region:")
    print("    -> RAMC improved gradient quality where it matters most")
    print("  If λ=5e-4 has LOWER cos(θ) than Fidelity:")
    print("    -> Explains why it's the worst closed-loop performer")
    print("    -> The optimiser follows misleading gradients")
    print("  If the difference is concentrated in the comfort-critical region:")
    print("    -> CVaR training achieved exactly its design goal")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
