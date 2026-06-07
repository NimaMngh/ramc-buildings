#!/usr/bin/env python3
"""
NMPC Experimental Matrix Runner — Causal Ablation Edition
==========================================================

This is the closed-loop ablation runner for the Revision Plan R2.2/R2.3
(causal attribution of RAMC's closed-loop benefit). The original main-paper
matrix has been replaced with the four ablation variants at a matched
magnitude.

Models:
  A0  Fidelity baseline (λ=0, MSE on un-perturbed inputs)
  A1  Perturbation-only training, RC-plant grounded labels (γ=0.0015)
  A2  Mean operational-cost regularization (λ_μ=0.0015)
  A3  CVaR tail-cost regularization (RAMC, λ=0.0015)

Scenarios (per Revision Plan, the two most informative):
  forecast_error
  cold_snap

Seeds: same 5 paired initial-condition seeds as the main paper.

Total experiments: 4 models × 2 scenarios × 5 seeds = 40.

Reports ΔCDH, ΔCVaR₀.₉, Δpeak violation, ΔE relative to A0 with paired
95% confidence intervals.

Workflow:
  Phase 0: Validation checklist (A-D) on each model
  Phase 1: Pairwise comparison (A0 vs A3 — headline ablation contrast)
  Phase 2: Full matrix (4 models × 2 scenarios × 5 seeds = 40 experiments)

Usage:
  python run_nmpc_matrix.py                          # Full ablation matrix
  python run_nmpc_matrix.py --quick                  # 1 seed, 2 days (smoke)
  python run_nmpc_matrix.py --validate-only          # Just run checklist
  python run_nmpc_matrix.py --pairwise               # Just A0 vs A3
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys
import matplotlib.pyplot as plt

# ─── Path setup (same as run_experimental_matrix.py) ─────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # 7_Close Loop Performance

sys.path.insert(0, str(SCRIPT_DIR))                             # M6
sys.path.insert(0, str(PROJECT_ROOT / "M4_Closed_Loop_Simulator"))

NN_ARCH_PATH = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
)
sys.path.insert(0, str(NN_ARCH_PATH))

# Import NMPC modules
from closed_loop_nmpc import (
    ClosedLoopNMPCSimulator,
    run_pairwise_comparison,
    run_validation_checklist,
)
from rc_ground_truth import RCGroundTruthModel
from shared_constants import (
    OCC_TARGET_C, DEADBAND_C, UNOCC_TARGET_C,
    T_SUPPLY_MIN, T_SUPPLY_MAX, MDOT_MIN, MDOT_MAX,
    ENERGY_COST_RATE,
)


# =============================================================================
# Configuration — MUST MATCH run_experimental_matrix.py
# =============================================================================

DT_SECONDS = 600
SIMULATION_DAYS = 7
STEPS_PER_DAY = 144
SIMULATION_STEPS = SIMULATION_DAYS * STEPS_PER_DAY

# NMPC-specific settings (replacing QP settings)
NMPC_HORIZON = 24         # 4 hours (shorter than QP's 144 = 24h)
NMPC_BLOCK_SIZE = 4       # 40-min blocks -> 6 blocks × 2 inputs = 12 vars
NMPC_N_ITER = 25          # Adam iterations per MPC step
NMPC_LR = 0.05            # Adam learning rate
NMPC_GRAD_CLIP = 5.0

# Objective weights — SAME for all models (non-negotiable for fair comparison)
W_ENERGY = 0.9            # Matched to Phase 2 energy_cost_rate
W_COLD = 63.0             # Matched to Phase 2 LOSS_CONFIG w_comfort
W_HOT = 30.0              # Softer on hot side (heating context)
W_DU = 1e-3               # Slew penalty
W_TERMINAL = 20.0         # Terminal cold penalty
W_TRUST = 0.0             # Trust region (0 = off by default)

RC_PARAMS = {
    "C_air": 84426246.51832934,
    "C_env": 661376048.5634619,
    "C_int": 6002555765.428176,
    "C_rad": 597376.9622532368,
    "R_ex": 0.0003079967462046204,
    "R_ae": 0.00016180515492072352,
    "R_ai": 5.818704417041902e-05,
    "K_rad": 283.97226311004107,
    "a_rad": 0.2795410071110868,
    "A_sol": 0.5598500539656959,
}

# ─── Model paths (UPDATE THESE to match your local setup) ────────────────────
# Updated to point at the post-revision training run that produced the four
# ablation variants (A0, A1, A2, A3).
RAMC_MODELS_DIR = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
    r"\results\RAMC_FULL_cvar_20260518_182822"
)

# Matched magnitude for the ablation (Revision Plan R2.2/R2.3). The same
# value is used for A1's γ, A2's λ_μ, and A3's λ so the contrast across the
# three variants reflects the choice of reduction operator only.
ABLATION_LAMBDA = 0.0015
LAMBDA_MAX_RISK = ABLATION_LAMBDA   # kept for backward compatibility below

MODELS = [
    # ── A0: Fidelity baseline ───────────────────────────────────────────
    {
        "name": "Fidelity_Baseline_rollout",
        "path": str(RAMC_MODELS_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth"),
        "description": "A0: Fidelity (MSE on un-perturbed inputs, no risk term)",
        "lambda": 0.0,
    },
    # ── A1: Perturbation-only training with RC-plant labels ─────────────
    {
        "name": f"PertOnly_gamma_{ABLATION_LAMBDA}_rollout",
        "path": str(RAMC_MODELS_DIR /
                    f"PertOnly_gamma_{ABLATION_LAMBDA}_rollout_a1.0_best.pth"),
        "description": "A1: Perturbation-only (RC-grounded labels, no operational cost)",
        "lambda": None,
    },
    # ── A2: Mean operational-cost regularization ─────────────────────────
    {
        "name": f"MeanCost_lambda_{ABLATION_LAMBDA}_rollout",
        "path": str(RAMC_MODELS_DIR /
                    f"MeanCost_lambda_{ABLATION_LAMBDA}_rollout_a1.0_best.pth"),
        "description": "A2: Mean operational-cost regularization (no tail focus)",
        "lambda": ABLATION_LAMBDA,
    },
    # ── A3: CVaR tail-cost regularization (the proposed RAMC) ────────────
    {
        "name": f"RAMC_lambda_{ABLATION_LAMBDA}_rollout",
        "path": str(RAMC_MODELS_DIR /
                    f"RAMC_lambda_{ABLATION_LAMBDA}_op_cvar_rollout_a1.0_best.pth"),
        "description": "A3: CVaR tail-cost regularization (RAMC, matched λ)",
        "lambda": ABLATION_LAMBDA,
    },
]

# ─── Weather paths (UPDATE THESE) ────────────────────────────────────────────
WEATHER_DIR = PROJECT_ROOT / "M1_Weather_Data" / "data_ramc_epw"

# Per Revision Plan: ablation is evaluated on the two most informative
# scenarios only. Nominal is omitted to keep the experiment count down to
# 4 × 2 × 5 = 40, as the plan recommends for cost control.
SCENARIOS = [
    {"name": "forecast_error",
     "truth_file": str(WEATHER_DIR / "forecast_error_truth.csv"),
     "forecast_file": str(WEATHER_DIR / "forecast_error_forecast.csv"),
     "description": "Biased forecast (warm bias + AR(1) noise on T_out)"},
    {"name": "cold_snap",
     "truth_file": str(WEATHER_DIR / "cold_snap_truth.csv"),
     "forecast_file": str(WEATHER_DIR / "cold_snap_forecast.csv"),
     "description": "48h cold dip"},
]

SEEDS = [42, 123, 456, 789, 1000]


# =============================================================================
# Initial state (matches run_experimental_matrix.py exactly)
# =============================================================================

def create_initial_state(seed: int) -> np.ndarray:
    """Create seed-dependent initial state."""
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


# =============================================================================
# Metric extraction (compatible with existing analysis code)
# =============================================================================

def _extract_metrics(results: Dict) -> Dict[str, float]:
    """Extract metrics from simulation results."""
    return {
        # Energy
        'total_energy_kWh': float(results['total_energy_kWh']),
        'total_energy_cost': float(results['total_energy_cost']),

        # PRIMARY: Cold violations
        'peak_cold_violation_occ_C': float(results['peak_cold_violation_occ_C']),
        'deg_hours_cold_occ': float(results['deg_hours_cold_occ']),
        'hours_cold_outside_occ': float(results['hours_cold_outside_occ']),
        'hours_cold_outside_occ_025': float(results['hours_cold_outside_occ_025']),
        'n_cold_violations_occ': int(results['n_cold_violations_occ']),
        'var90_cold_occ_C': float(results['var90_cold_occ_C']),
        'cvar90_cold_occ_C': float(results['cvar90_cold_occ_C']),
        'var95_cold_occ_C': float(results['var95_cold_occ_C']),
        'cvar95_cold_occ_C': float(results['cvar95_cold_occ_C']),

        # SECONDARY: Warm violations
        'peak_warm_violation_occ_C': float(results['peak_warm_violation_occ_C']),
        'deg_hours_warm_occ': float(results['deg_hours_warm_occ']),
        'hours_warm_outside_occ': float(results['hours_warm_outside_occ']),
        'n_warm_violations_occ': int(results['n_warm_violations_occ']),
        'var90_warm_occ_C': float(results['var90_warm_occ_C']),
        'cvar90_warm_occ_C': float(results['cvar90_warm_occ_C']),
        'var95_warm_occ_C': float(results['var95_warm_occ_C']),
        'cvar95_warm_occ_C': float(results['cvar95_warm_occ_C']),

        # Band (combined)
        'peak_band_violation_occ_C': float(results['peak_band_violation_occ_C']),
        'deg_hours_band_occ': float(results['deg_hours_band_occ']),
        'hours_band_outside_occ': float(results['hours_band_outside_occ']),
        'n_band_violations_occ': int(results['n_band_violations_occ']),
        'var90_band_occ_C': float(results['var90_band_occ_C']),
        'cvar90_band_occ_C': float(results['cvar90_band_occ_C']),
        'var95_band_occ_C': float(results['var95_band_occ_C']),
        'cvar95_band_occ_C': float(results['cvar95_band_occ_C']),

        # Legacy
        'deg_hours_occ': float(results['deg_hours_occ']),
        'deg_hours_below_occ': float(results['deg_hours_below_occ']),
        'deg_hours_above_occ': float(results['deg_hours_above_occ']),
        'hours_outside_occ': float(results['hours_outside_occ']),
        'peak_violation_occ_C': float(results['peak_violation_occ_C']),
        'n_violations_occ': int(results['n_violations_occ']),
        'var90_occ_C': float(results['var90_occ_C']),
        'cvar90_occ_C': float(results['cvar90_occ_C']),
        'var95_occ_C': float(results['var95_occ_C']),
        'cvar95_occ_C': float(results['cvar95_occ_C']),

        # T_air
        'T_air_occ_mean_C': float(results['T_air_occ_mean_C']),
        'T_air_occ_min_C': float(results['T_air_occ_min_C']),
        'T_air_occ_max_C': float(results['T_air_occ_max_C']),
        'T_air_mean_C': float(results['T_air_mean_C']),
        'T_air_min_C': float(results['T_air_min_C']),
        'T_air_max_C': float(results['T_air_max_C']),

        # Full simulation
        'deg_hours_full': float(results['deg_hours_full']),

        # Solver
        'fallback_rate': float(np.mean(results['fallback_used'])),
        'user_limit_rate': float(np.mean(results.get('user_limit_used', [False]))),
        'median_solve_time_ms': float(np.median(results['solver_time_ms'])),
        'p95_solve_time_ms': float(np.percentile(results['solver_time_ms'], 95)),
        'max_solve_time_ms': float(np.max(results['solver_time_ms'])),
        'occupied_hours': float(results['occupied_hours']),
        'peak_Q_heat_kW': float(np.max(results['Q_heat_W_step']) / 1000.0),

        # OSQP compat (all -1 for NMPC)
        'median_osqp_iters_last': -1.0,
        'p95_osqp_iters_last': -1.0,
        'max_osqp_iters_last': -1.0,
        'median_osqp_iters_total': -1.0,
        'p95_osqp_iters_total': -1.0,
        'max_osqp_iters_total': -1.0,

        # RAMC-style cost
        'total_stage_cost_ramc': float(results['total_stage_cost_ramc']),
        'total_stage_cost_ramc_occ': float(results['total_stage_cost_ramc_occ']),
        'total_comfort_cost_ramc': float(results['total_comfort_cost_ramc']),
        'total_energy_cost_ramc': float(results['total_energy_cost_ramc']),
        'cvar90_stage_cost_occ': float(results['cvar90_stage_cost_occ']),
        'cvar95_stage_cost_occ': float(results['cvar95_stage_cost_occ']),
        'cvar90_stage_cost_all': float(results['cvar90_stage_cost_all']),
        'cvar95_stage_cost_all': float(results['cvar95_stage_cost_all']),
        'var90_stage_cost_occ': float(results['var90_stage_cost_occ']),
        'var95_stage_cost_occ': float(results['var95_stage_cost_occ']),

        # Stabilization compat (N/A for NMPC)
        'rho_A_before_median': -1.0,
        'rho_A_after_max': -1.0,
        'pct_steps_stabilized': 0.0,

        # First-step slack compat (N/A for NMPC)
        'median_cold_slack_0': -1.0,
        'p95_cold_slack_0': -1.0,
        'max_cold_slack_0': -1.0,
        'median_warm_slack_0': -1.0,

        # NMPC-specific
        'nmpc_median_loss_reduction': float(np.nanmedian(
            results.get('nmpc_loss_reduction_step', [0.0]))),
        'nmpc_mean_loss_reduction': float(np.nanmean(
            results.get('nmpc_loss_reduction_step', [0.0]))),
    }


# =============================================================================
# NMPC-specific NMPC settings as a dict for the simulator
# =============================================================================

def _get_nmpc_kwargs() -> dict:
    """Return NMPC kwargs that stay constant across all experiments."""
    return {
        'nmpc_horizon': NMPC_HORIZON,
        'nmpc_block_size': NMPC_BLOCK_SIZE,
        'nmpc_n_iter': NMPC_N_ITER,
        'nmpc_lr': NMPC_LR,
        'nmpc_grad_clip': NMPC_GRAD_CLIP,
        'w_energy': W_ENERGY,
        'w_cold': W_COLD,
        'w_hot': W_HOT,
        'w_du': W_DU,
        'w_terminal': W_TERMINAL,
        'w_trust': W_TRUST,
        'du_max': np.array([2.0, 0.3]),
        'energy_cost_rate': ENERGY_COST_RATE,
        'dtype': torch.float64,
    }


# =============================================================================
# Phase 0: Validation
# =============================================================================

def run_phase0_validation(output_dir: Path) -> bool:
    """
    Run validation checklist on each model.

    Returns True if all models pass.
    """
    print("\n" + "#" * 70)
    print("PHASE 0: VALIDATION CHECKLIST")
    print("#" * 70)

    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    all_pass = True

    for model_cfg in MODELS:
        print(f"\n{'─'*50}")
        print(f"Validating: {model_cfg['name']}")
        print(f"{'─'*50}")

        try:
            results = run_validation_checklist(
                model_path=model_cfg['path'],
                weather_truth_path=SCENARIOS[0]['truth_file'],
                dtype=torch.float64,
                verbose=True,
            )

            # Save validation results
            save_results = {k: v for k, v in results.items()
                          if k != 'all_pass'}
            # Convert non-serializable items
            for check_key in save_results:
                if isinstance(save_results[check_key], dict):
                    for k, v in save_results[check_key].items():
                        if isinstance(v, np.ndarray):
                            save_results[check_key][k] = v.tolist()

            with open(val_dir / f"validation_{model_cfg['name']}.json", 'w') as f:
                json.dump({
                    'model': model_cfg['name'],
                    'all_pass': results['all_pass'],
                    'checks': {
                        'A_one_step_match': results['A_one_step']['match'],
                        'B_rollout_match': results['B_rollout']['match'],
                        'C_gradient_consistent': results['C_gradient']['directionally_consistent'],
                        'D_loss_decreased': results['D_optimization']['loss_decreased'],
                    }
                }, f, indent=2)

            if not results['all_pass']:
                all_pass = False
                print(f"  {model_cfg['name']} FAILED validation!")

        except Exception as e:
            print(f"  Validation failed for {model_cfg['name']}: {e}")
            all_pass = False

    if all_pass:
        print(f"\nALL MODELS PASSED VALIDATION")
    else:
        print(f"\nSOME MODELS FAILED — review before proceeding")

    return all_pass


# =============================================================================
# Phase 1: Pairwise comparison (A0 vs A3 — headline ablation contrast)
# =============================================================================

def run_phase1_pairwise(output_dir: Path):
    """
    Headline ablation comparison: A0 Fidelity vs A3 RAMC at the matched
    magnitude λ=0.0015, on forecast_error and cold_snap, seed 42.
    """
    print("\n" + "#" * 70)
    print("PHASE 1: PAIRWISE COMPARISON (A0 vs A3 — headline ablation contrast)")
    print("#" * 70)

    pair_dir = output_dir / "pairwise"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Find models
    model_a = next(m for m in MODELS if m['name'] == 'Fidelity_Baseline_rollout')
    model_b = next(m for m in MODELS if m['name'] == f'RAMC_lambda_{ABLATION_LAMBDA}_rollout')

    # Scenarios to test (forecast_error and cold_snap; these are the only
    # two in the SCENARIOS list now, but we still iterate for clarity)
    test_scenarios = list(SCENARIOS)

    nmpc_kwargs = _get_nmpc_kwargs()
    seed = 42

    for scenario in test_scenarios:
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario['name']} (seed={seed})")
        print(f"{'─'*60}")

        init_state = create_initial_state(seed)

        comparison = run_pairwise_comparison(
            model_a_path=model_a['path'],
            model_b_path=model_b['path'],
            weather_truth_path=scenario['truth_file'],
            weather_forecast_path=scenario['forecast_file'],
            initial_state=init_state,
            simulation_steps=SIMULATION_STEPS,
            seed=seed,
            label_a=model_a['name'],
            label_b=model_b['name'],
            verbose=True,
            **nmpc_kwargs,
        )

        # Save comparison
        comp_save = {
            'label_a': comparison['label_a'],
            'label_b': comparison['label_b'],
            'scenario': scenario['name'],
            'seed': seed,
            'comparison_metrics': comparison['comparison_metrics'],
            'metrics_a': _extract_metrics(comparison['results_a']),
            'metrics_b': _extract_metrics(comparison['results_b']),
        }
        with open(pair_dir / f"pairwise_{scenario['name']}_seed{seed}.json", 'w') as f:
            json.dump(comp_save, f, indent=2)

        # Save trajectories
        for label, results in [('a', comparison['results_a']),
                               ('b', comparison['results_b'])]:
            traj = {
                'T_air': results['states'][:, 0].tolist(),
                'T_ret': results['states'][:, 5].tolist(),
                'T_supply': results['controls'][:, 0].tolist(),
                'mdot': results['controls'][:, 1].tolist(),
                'Tmin': results['Tmin_series'].tolist(),
                'Tmax': results['Tmax_series'].tolist(),
                'occupancy': results['occupancy_series'].tolist(),
            }
            with open(pair_dir / f"traj_{label}_{scenario['name']}_seed{seed}.json", 'w') as f:
                json.dump(traj, f)


# =============================================================================
# Phase 2: Full matrix (4 models × 2 scenarios × 5 seeds = 40)
# =============================================================================

def run_phase2_full_matrix(
    output_dir: Path,
    resume_from: Optional[int] = None,
) -> List[Dict]:
    """
    Run full NMPC ablation matrix.

    4 models × 2 scenarios × 5 seeds = 40 experiments.
    """
    print("\n" + "#" * 70)
    print("PHASE 2: FULL NMPC ABLATION MATRIX")
    print("#" * 70)

    matrix_dir = output_dir / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    exp_id = 1
    for model in MODELS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                experiments.append({
                    'id': exp_id,
                    'model': model,
                    'scenario': scenario,
                    'seed': seed,
                })
                exp_id += 1

    total = len(experiments)

    print(f"Models: {len(MODELS)}, Scenarios: {len(SCENARIOS)}, Seeds: {len(SEEDS)}")
    print(f"Total: {total} experiments")
    print(f"\nNMPC settings:")
    print(f"  Horizon: {NMPC_HORIZON} steps ({NMPC_HORIZON * DT_SECONDS / 3600:.1f}h)")
    print(f"  Block size: {NMPC_BLOCK_SIZE} ({NMPC_BLOCK_SIZE * DT_SECONDS / 60:.0f}min)")
    print(f"  Iterations: {NMPC_N_ITER}, LR: {NMPC_LR}")
    print(f"  dtype: float64")
    print(f"\nAblation variants:")
    for i, m in enumerate(MODELS, 1):
        print(f"  {i}. {m['name']}: {m['description']}")

    # Save config
    config = {
        'experiment_type': 'nmpc_direct_shooting',
        'experiment_version': 'ABLATION_R2_2_R2_3',
        'building': 'Radiator-heated commercial building',
        'comfort_occupied_C': [OCC_TARGET_C - DEADBAND_C, OCC_TARGET_C + DEADBAND_C],
        'comfort_target_C': OCC_TARGET_C,
        'comfort_deadband_C': DEADBAND_C,
        'primary_metric': 'cold_only',
        'ablation_lambda': ABLATION_LAMBDA,
        'models': [m['name'] for m in MODELS],
        'model_lambdas': {m['name']: m.get('lambda') for m in MODELS},
        'scenarios': [s['name'] for s in SCENARIOS],
        'seeds': SEEDS,
        'simulation_steps': SIMULATION_STEPS,
        'simulation_days': SIMULATION_DAYS,
        'timestep_seconds': DT_SECONDS,
        'energy_cost_rate': ENERGY_COST_RATE,
        'nmpc_settings': {
            'horizon': NMPC_HORIZON,
            'block_size': NMPC_BLOCK_SIZE,
            'n_iter': NMPC_N_ITER,
            'lr': NMPC_LR,
            'grad_clip': NMPC_GRAD_CLIP,
            'dtype': 'float64',
        },
        'objective_weights': {
            'w_energy': W_ENERGY,
            'w_cold': W_COLD,
            'w_hot': W_HOT,
            'w_du': W_DU,
            'w_terminal': W_TERMINAL,
            'w_trust': W_TRUST,
        },
        'actuator_limits': {
            'T_supply_min_C': T_SUPPLY_MIN,
            'T_supply_max_C': T_SUPPLY_MAX,
            'mdot_min_kg_s': MDOT_MIN,
            'mdot_max_kg_s': MDOT_MAX,
        },
        'du_max': [2.0, 0.3],
        'rc_params': RC_PARAMS,
        'run_timestamp': datetime.now().isoformat(),
        'note': (
            'Closed-loop causal ablation (Revision Plan R2.2/R2.3). Four loss-function '
            f'variants (A0 Fidelity, A1 PertOnly, A2 MeanCost, A3 RAMC) at matched '
            f'magnitude λ=γ={ABLATION_LAMBDA}. Evaluated on the two most informative '
            'scenarios (forecast_error, cold_snap) over five paired seeds. The ONLY '
            'difference between controllers is the planning model f_θ.'
        ),
    }

    with open(matrix_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ── Simulator cache ──
    nmpc_kwargs = _get_nmpc_kwargs()
    simulator_cache: Dict[tuple, ClosedLoopNMPCSimulator] = {}

    def get_simulator(model_cfg, scenario_cfg) -> ClosedLoopNMPCSimulator:
        key = (model_cfg['name'], scenario_cfg['name'])
        if key not in simulator_cache:
            print(f"  Creating NMPC simulator: {key}")
            sim = ClosedLoopNMPCSimulator(
                nn_model_path=model_cfg['path'],
                weather_truth_path=scenario_cfg['truth_file'],
                weather_forecast_path=scenario_cfg['forecast_file'],
                verbose_init=(len(simulator_cache) == 0),
                **nmpc_kwargs,
            )
            # Ensure same RC plant for all
            sim.ground_truth = RCGroundTruthModel(
                params=RC_PARAMS, dt_seconds=DT_SECONDS
            )
            simulator_cache[key] = sim
        return simulator_cache[key]

    # ── Run experiments ──
    results = []
    start_total = time.time()

    for exp in experiments:
        if resume_from and exp['id'] < resume_from:
            rf = matrix_dir / f"result_exp{exp['id']}.json"
            if rf.exists():
                with open(rf) as f:
                    results.append(json.load(f))
            continue

        print(f"\n{'='*60}")
        print(f"EXP {exp['id']}/{total}: {exp['model']['name']} | "
              f"{exp['scenario']['name']} | seed={exp['seed']}")

        start = time.time()

        try:
            sim = get_simulator(exp['model'], exp['scenario'])
            sim.reset_for_new_seed()

            torch.manual_seed(exp['seed'])
            init_state = create_initial_state(exp['seed'])

            sim_results = sim.simulate_episode(
                initial_state=init_state,
                simulation_steps=SIMULATION_STEPS,
                verbose=True,
                log_interval=100,
            )

            elapsed = time.time() - start
            metrics = _extract_metrics(sim_results)
            metrics['initial_state'] = init_state.tolist()

            result = {
                'experiment_id': exp['id'],
                'model': exp['model']['name'],
                'model_lambda': exp['model'].get('lambda'),
                'scenario': exp['scenario']['name'],
                'seed': exp['seed'],
                'success': True,
                'elapsed_s': elapsed,
                'metrics': metrics,
            }

            # Save trajectory
            traj = {
                'T_air': sim_results['states'][:, 0].tolist(),
                'T_ret': sim_results['states'][:, 5].tolist(),
                'T_supply': sim_results['controls'][:, 0].tolist(),
                'mdot': sim_results['controls'][:, 1].tolist(),
                'Tmin': sim_results['Tmin_series'].tolist(),
                'Tmax': sim_results['Tmax_series'].tolist(),
                'occupancy': sim_results['occupancy_series'].tolist(),
                'energy_kWh': sim_results['energy_kWh_step'],
                'stage_cost_ramc': sim_results.get('stage_cost_ramc_step', []),
                'solver_time_ms': sim_results['solver_time_ms'],
                'solver_status': [str(s) for s in sim_results['solver_status_step']],
                'nmpc_loss_initial': sim_results.get('nmpc_loss_initial_step', []),
                'nmpc_loss_final': sim_results.get('nmpc_loss_final_step', []),
                'nmpc_loss_reduction': sim_results.get('nmpc_loss_reduction_step', []),
            }
            with open(matrix_dir / f"traj_exp{exp['id']}.json", 'w') as f:
                json.dump(traj, f)

            print(f"  Energy: {metrics['total_energy_kWh']:.0f} kWh | "
                  f"Cold-DH: {metrics['deg_hours_cold_occ']:.3f} | "
                  f"Peak-Cold: {metrics['peak_cold_violation_occ_C']:.2f}°C | "
                  f"CVaR90-Cold: {metrics['cvar90_cold_occ_C']:.3f}°C | "
                  f"Time: {elapsed:.1f}s")

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            result = {
                'experiment_id': exp['id'],
                'model': exp['model']['name'],
                'scenario': exp['scenario']['name'],
                'seed': exp['seed'],
                'success': False,
                'error': str(e),
            }

        results.append(result)
        with open(matrix_dir / f"result_exp{exp['id']}.json", 'w') as f:
            json.dump(result, f, indent=2)

        done = len([r for r in results if 'success' in r])
        elapsed_total = time.time() - start_total
        eta = (elapsed_total / done) * (total - done) if done > 0 else 0
        print(f"  Progress: {done}/{total} | ETA: {eta / 60:.1f}min")

    # ── Save all results ──
    with open(matrix_dir / "all_results.json", 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)

    # ── Generate analysis ──
    elapsed_total = time.time() - start_total
    successful = sum(1 for r in results if r.get('success'))

    print(f"\n{'='*70}")
    print(f"COMPLETE: {successful}/{total} successful in {elapsed_total / 60:.1f}min")
    print(f"{'='*70}")

    _generate_aggregate_table(results, matrix_dir)
    _generate_paired_differences(results, matrix_dir)
    _generate_figures(results, matrix_dir)

    print(f"\nSaved to: {matrix_dir}")
    return results


# =============================================================================
# Analysis functions
# =============================================================================

def _generate_aggregate_table(results: List[Dict], output_dir: Path):
    """Generate aggregate metrics table (model × scenario)."""
    successful = [r for r in results if r.get('success')]
    if not successful:
        return

    print(f"\n{'='*70}")
    print("AGGREGATE METRICS TABLE")
    print(f"{'='*70}")

    models = sorted(set(r['model'] for r in successful))
    scenarios = sorted(set(r['scenario'] for r in successful))

    key_metrics = [
        'total_energy_kWh', 'deg_hours_cold_occ',
        'cvar90_cold_occ_C', 'cvar95_cold_occ_C',
        'peak_cold_violation_occ_C', 'hours_cold_outside_occ',
    ]

    rows = []
    for model in models:
        for scenario in scenarios:
            exps = [r for r in successful
                    if r['model'] == model and r['scenario'] == scenario]
            if not exps:
                continue
            row = {'model': model, 'scenario': scenario, 'n_seeds': len(exps)}
            for metric in key_metrics:
                vals = [r['metrics'][metric] for r in exps]
                row[f'{metric}_mean'] = float(np.mean(vals))
                row[f'{metric}_std'] = float(np.std(vals))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "aggregate_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print summary table
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}")
        print(f"  {'Model':<40s} {'Energy(kWh)':>12s} {'DH-Cold':>10s} "
              f"{'CVaR90':>10s} {'PeakCold':>10s}")
        print(f"  {'─'*82}")
        for model in models:
            subset = [r for r in successful
                      if r['model'] == model and r['scenario'] == scenario]
            if not subset:
                continue
            e = np.mean([r['metrics']['total_energy_kWh'] for r in subset])
            dh = np.mean([r['metrics']['deg_hours_cold_occ'] for r in subset])
            cv = np.mean([r['metrics']['cvar90_cold_occ_C'] for r in subset])
            pk = np.mean([r['metrics']['peak_cold_violation_occ_C'] for r in subset])
            print(f"  {model:<40s} {e:>12.1f} {dh:>10.3f} {cv:>10.4f} {pk:>10.3f}")


def _generate_paired_differences(results: List[Dict], output_dir: Path):
    """Generate paired differences (each variant vs A0 Fidelity_Baseline)
    with paired 95% confidence intervals."""
    successful = [r for r in results if r.get('success')]
    if not successful:
        return

    baseline = 'Fidelity_Baseline_rollout'
    scenarios = sorted(set(r['scenario'] for r in successful))
    seeds = sorted(set(r['seed'] for r in successful))
    models = sorted(set(r['model'] for r in successful))

    comparison_metrics = [
        'cvar90_cold_occ_C', 'cvar95_cold_occ_C',
        'peak_cold_violation_occ_C', 'deg_hours_cold_occ',
        'hours_cold_outside_occ', 'total_energy_kWh',
    ]

    print(f"\n{'='*70}")
    print("PAIRED DIFFERENCES vs A0 (Fidelity_Baseline), with 95% CI")
    print(f"{'='*70}")

    # Use t-distribution critical value for paired 95% CI on n_pairs samples
    try:
        from scipy import stats as _stats
        def _ci95(vals):
            arr = np.asarray(vals, dtype=float)
            n = len(arr)
            if n < 2:
                return float('nan'), float('nan')
            se = arr.std(ddof=1) / np.sqrt(n)
            tcrit = _stats.t.ppf(0.975, df=n - 1)
            half = tcrit * se
            return float(arr.mean() - half), float(arr.mean() + half)
    except ImportError:
        # Fallback if scipy not available — normal approximation
        def _ci95(vals):
            arr = np.asarray(vals, dtype=float)
            n = len(arr)
            if n < 2:
                return float('nan'), float('nan')
            se = arr.std(ddof=1) / np.sqrt(n)
            half = 1.96 * se
            return float(arr.mean() - half), float(arr.mean() + half)

    rows = []
    for scenario in scenarios:
        for model in models:
            if model == baseline:
                continue
            diffs = {m: [] for m in comparison_metrics}
            for seed in seeds:
                r_base = next(
                    (r for r in successful
                     if r['model'] == baseline and r['scenario'] == scenario
                     and r['seed'] == seed), None)
                r_model = next(
                    (r for r in successful
                     if r['model'] == model and r['scenario'] == scenario
                     and r['seed'] == seed), None)
                if r_base and r_model:
                    for m in comparison_metrics:
                        diffs[m].append(
                            r_model['metrics'][m] - r_base['metrics'][m]
                        )

            if not diffs[comparison_metrics[0]]:
                continue

            row = {'model': model, 'scenario': scenario,
                   'n_pairs': len(diffs[comparison_metrics[0]])}
            for m in comparison_metrics:
                vals = diffs[m]
                row[f'{m}_mean_diff'] = float(np.mean(vals))
                row[f'{m}_std_diff'] = float(np.std(vals))
                lo, hi = _ci95(vals)
                row[f'{m}_ci95_lo'] = lo
                row[f'{m}_ci95_hi'] = hi
            rows.append(row)

            # Print
            print(f"\n  {model} vs {baseline} | {scenario} "
                  f"({len(diffs[comparison_metrics[0]])} pairs)")
            for m in comparison_metrics:
                vals = diffs[m]
                mean_d = np.mean(vals)
                lo, hi = _ci95(vals)
                better = "better" if mean_d < 0 else "worse"
                if 'energy' in m:
                    better = "less" if mean_d < 0 else "more"
                print(f"    Δ {m:<35s}: {mean_d:>+.4f}  "
                      f"95% CI [{lo:>+.4f}, {hi:>+.4f}]  ({better})")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "paired_differences.csv", index=False)


def _generate_figures(results: List[Dict], output_dir: Path):
    """Generate comparison figures for the ablation matrix."""
    successful = [r for r in results if r.get('success')]
    if not successful:
        return

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(set(r['model'] for r in successful))
    scenarios = sorted(set(r['scenario'] for r in successful))

    COLORS_MAP = {
        'Fidelity_Baseline_rollout':                       '#3498db',   # blue
        f'PertOnly_gamma_{ABLATION_LAMBDA}_rollout':       '#7f7f7f',   # gray
        f'MeanCost_lambda_{ABLATION_LAMBDA}_rollout':      '#8c564b',   # brown
        f'RAMC_lambda_{ABLATION_LAMBDA}_rollout':          '#27ae60',   # green
    }

    SHORT_NAMES = {
        'Fidelity_Baseline_rollout':                       'A0 Fidelity',
        f'PertOnly_gamma_{ABLATION_LAMBDA}_rollout':       'A1 Pert-only',
        f'MeanCost_lambda_{ABLATION_LAMBDA}_rollout':      'A2 Mean cost',
        f'RAMC_lambda_{ABLATION_LAMBDA}_rollout':          'A3 RAMC ',
    }

    # ── CVaR90 Cold boxplot ──
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        data = []
        for mn in models:
            vals = [r['metrics']['cvar90_cold_occ_C'] for r in successful
                    if r['scenario'] == scenario and r['model'] == mn]
            data.append(vals if vals else [0])
        short_labels = [SHORT_NAMES.get(m, m) for m in models]
        bp = ax.boxplot(data, tick_labels=short_labels, patch_artist=True)
        for patch, m in zip(bp['boxes'], models):
            patch.set_facecolor(COLORS_MAP.get(m, 'gray'))
            patch.set_alpha(0.5)
        ax.set_ylabel('CVaR90 Cold Violation (°C)')
        ax.set_title(scenario)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Causal ablation — CVaR90 Cold (PRIMARY)', fontsize=12)
    plt.tight_layout()
    plt.savefig(figs_dir / "nmpc_boxplot_cvar90_cold.png", dpi=150)
    plt.close()

    # ── Energy boxplot ──
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        data = []
        for mn in models:
            vals = [r['metrics']['total_energy_kWh'] for r in successful
                    if r['scenario'] == scenario and r['model'] == mn]
            data.append(vals if vals else [0])
        short_labels = [SHORT_NAMES.get(m, m) for m in models]
        bp = ax.boxplot(data, tick_labels=short_labels, patch_artist=True)
        for patch, m in zip(bp['boxes'], models):
            patch.set_facecolor(COLORS_MAP.get(m, 'gray'))
            patch.set_alpha(0.5)
        ax.set_ylabel('Total Energy (kWh)')
        ax.set_title(scenario)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Causal ablation — Energy Consumption', fontsize=12)
    plt.tight_layout()
    plt.savefig(figs_dir / "nmpc_boxplot_energy.png", dpi=150)
    plt.close()

    # ── Solver timing boxplot ──
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        data = []
        for mn in models:
            vals = [r['metrics']['median_solve_time_ms'] for r in successful
                    if r['scenario'] == scenario and r['model'] == mn]
            data.append(vals if vals else [0])
        short_labels = [SHORT_NAMES.get(m, m) for m in models]
        bp = ax.boxplot(data, tick_labels=short_labels, patch_artist=True)
        for patch, m in zip(bp['boxes'], models):
            patch.set_facecolor(COLORS_MAP.get(m, 'gray'))
            patch.set_alpha(0.5)
        ax.set_ylabel('Median Solve Time (ms)')
        ax.set_title(scenario)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Causal ablation — Solver Timing', fontsize=12)
    plt.tight_layout()
    plt.savefig(figs_dir / "nmpc_boxplot_solve_time.png", dpi=150)
    plt.close()

    print(f"  Saved figures to {figs_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="NMPC Direct Shooting Causal Ablation Matrix (R2.2/R2.3)"
    )
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--resume", "-r", type=int, default=None,
                        help="Resume from experiment ID")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode: 1 seed, 2 days")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation checklist")
    parser.add_argument("--pairwise", action="store_true",
                        help="Only run pairwise comparison (A0 vs A3)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip Phase 0 validation")
    args = parser.parse_args()

    if args.quick:
        SIMULATION_DAYS = 2
        SIMULATION_STEPS = SIMULATION_DAYS * STEPS_PER_DAY
        SEEDS.clear()
        SEEDS.append(42)
        print(f"QUICK MODE: 1 seed, {SIMULATION_DAYS} days, "
              f"{len(SCENARIOS)} scenarios, {len(MODELS)} models "
              f"= {len(MODELS) * len(SCENARIOS) * 1} experiments")

    # Output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SCRIPT_DIR / f"results_NMPC_ablation_{timestamp}"
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print("CAUSAL ABLATION: NMPC DIRECT SHOOTING (R2.2/R2.3)")
    print(f"{'#'*70}")
    print(f"Output: {output_dir}")
    print(f"Controller: Direct-shooting NMPC (NO linearization, NO QP)")
    print(f"Horizon: {NMPC_HORIZON} steps ({NMPC_HORIZON * DT_SECONDS / 3600:.1f}h)")
    print(f"Block size: {NMPC_BLOCK_SIZE} ({NMPC_BLOCK_SIZE * DT_SECONDS / 60:.0f}min)")
    print(f"Simulation: {SIMULATION_DAYS} days, {len(SEEDS)} seeds")
    print(f"Ablation magnitude: λ=γ={ABLATION_LAMBDA}")

    # ── Phase 0: Validation ──
    if args.validate_only:
        run_phase0_validation(output_dir)
        exit(0)

    if not args.skip_validation:
        all_pass = run_phase0_validation(output_dir)
        if not all_pass:
            print("\nValidation failed. Use --skip-validation to proceed anyway.")
            exit(1)

    # ── Phase 1: Pairwise ──
    if args.pairwise:
        run_phase1_pairwise(output_dir)
        exit(0)

    # Always run pairwise first (expert recommendation)
    run_phase1_pairwise(output_dir)

    # ── Phase 2: Full matrix ──
    run_phase2_full_matrix(output_dir, resume_from=args.resume)
