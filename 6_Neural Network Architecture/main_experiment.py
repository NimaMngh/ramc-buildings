# -*- coding: utf-8 -*-
"""
RAMC Main Experiment Script

Runs the full RAMC experiment pipeline: data loading with train-only clamp
bounds, model training, test-set rollout evaluation, bias checking, analysis
(including Table IV per-state RMSE and Table V T_air bias CSV exports and a
methods summary), and report generation.
"""

import os
import json
import time
from datetime import datetime
import copy

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thermal_dynamics_net import (
    create_dataloaders, 
    ThermalDynamicsNet,
    build_clamp_bounds_from_dataset,
)
from trainers import TrainingManager
from openloop_rollout_eval import (
    evaluate_rollouts,
    compare_rollouts_multiple_models,
    plot_rollout_rmse,
)


# =============================================================================
# CHOOSE EXPERIMENT MODE HERE
# =============================================================================
EXPERIMENT_MODE = "FULL"  # "SCOUT" or "FULL"

# =============================================================================
# CHOOSE RISK OPERATOR: "std" or "cvar"
# =============================================================================
RISK_OPERATOR = "cvar"  # Change to "cvar" for CVaR experiments


# =============================================================================
# GLOBAL RESULTS DIRECTORY
# =============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("results", f"RAMC_{EXPERIMENT_MODE}_{RISK_OPERATOR}_{TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_DATA_PATH = "RAMC_training_data_N3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_ABLATIONS = False  # Global toggle for ablation handling


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    "state_dim": 6,
    "control_dim": 2,
    "disturbance_dim": 3,
    "hidden_dims": [256, 256, 128],
    "dropout_rate": 0.0,
}

# =============================================================================
# RAMC Loss Configuration - CVaR-ready
# =============================================================================

# Set num_perturbations based on risk operator
if RISK_OPERATOR.strip().lower() in ("cvar", "cvar_ru", "cvar_quantile"):
    # CVaR requires more samples for reliable tail estimation
    DEFAULT_TRAIN_K = 32  # Minimum 16, preferably 32
    DEFAULT_EVAL_K = 256
else:
    # STD/Variance can work with fewer samples
    DEFAULT_TRAIN_K = 8
    DEFAULT_EVAL_K = 256

LOSS_CONFIG = {
    # Risk configuration - No whitespace in risk_operator
    "risk_operator": RISK_OPERATOR.strip().lower(),
    "cvar_alpha": 0.9,
    "cvar_method": "ru",
    "cvar_n_steps": 10,
    
    "num_perturbations": DEFAULT_TRAIN_K,
    "perturb_forward_chunk_size": 65536,  # was 16384 — T4 can handle 4x more

    
    # P5A: Antithetic sampling for variance reduction
    "use_antithetic": True,

    # Gaussian perturbations
    "sigma_state": 1.0,
    "sigma_rad_scale": 0.5,
    "sigma_T_supply": 0.5,
    "sigma_mdot": 0.01,
    "sigma_T_out": 1.0,
    "sigma_Q_solar": 500.0,
    # Perturbation sigma: ~5-10% of typical occupied gains
    "sigma_Q_internal": 5000.0,    # Was 200 W — now meaningful at building scale
    "clamp_physical": True,
    
    # P2: clamp_bounds will be set from dataset (train only)
    "clamp_bounds": None,
    
    # Occupancy flip perturbations
    "p_occupancy_flip": 0.02,
    # Occupancy flip nominal: match actual occupied baseline
    "q_internal_nominal": 40000.0,  # Was 1000 W — now represents ~midpoint of occupied gains
                                      # When flipped: occupied->0 or unoccupied->40kW

    # Stage cost parameters
    "comfort_bounds": (20.0, 22.0),
    "comfort_hinge": "softplus",
    "comfort_beta": 0.5,
    "dt_minutes": 10.0,
    "energy_cost_rate": 0.9,
    "w_comfort": 63.0,
    "w_energy": 1.0,
    "cost_scale": 1.0,
    "t_ret_index": 5,

    # Option A (comfort-bounds alignment):
    # The training dataset was generated with OCC_TARGET_C=22.0, DEADBAND_C=0.5,
    # so its Tmin/Tmax columns encode [21.5, 22.5] °C for occupied periods.
    # Phase 3 MPC and all closed-loop evaluation use [20, 22] °C.
    # Setting ignore_dataset_bounds=True causes RAMCTrainer to discard the
    # dataset's Tmin/Tmax tensors and fall back to comfort_bounds=(20.0, 22.0)
    # for every CVaR stage-cost call during training and validation.
    # This ensures the risk-shaping incentive during training is aligned with
    # the comfort constraint enforced at deployment.
    "ignore_dataset_bounds": True,

    # Fidelity parameters
    "mse_normalize": True,
    # fidelity_weights[i] corresponds to state dimension i (must match model/dataset ordering).
    # We emphasize state[0] (T_air) and state[t_ret_index=5] (T_ret) equally.
    "fidelity_weights": [2.0, 1.0, 1.0, 0.5, 0.5, 2.0],

    # ── Rollout-aware fidelity (extended fidelity regularisation) ──────────────
    # Adds α · L_fidelity_rollout to the total loss during training.
    # When alpha_rollout=0 the trainer behaves identically to standard RAMC
    # (full backward compatibility preserved).
    "alpha_rollout": 1.0,           # Rollout fidelity weight α (0 = disabled)
    "rollout_horizon": 6,           # H_r: steps per training rollout (1 hour @ 10 min)
    "rollout_batch_size": 64,       # B_r: sequences per rollout mini-batch
    "rollout_step_weights": "linear",  # "linear" (emphasises later steps) or "uniform"

}

# =============================================================================
# Mode-Specific Configurations
# =============================================================================

if EXPERIMENT_MODE == "SCOUT":
    TRAINING_CONFIG = {
        "num_epochs": 50,
        "batch_size": 2048,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 10,
        "train_split": 0.7,
        "val_split": 0.2,
        "test_split": 0.1,
        "warmup_epochs": 3,
        "lambda_ramp_epochs": 10,
    }
    LAMBDA_VALUES = [0.0, 0.002, 0.005, 0.01]
    ROLLOUT_HORIZON = 12

elif EXPERIMENT_MODE == "FULL":
    # Tesla T4 has 15 GB — CVaR with K=32 can handle much larger batches
    batch_size = 8192 if RISK_OPERATOR.strip().lower().startswith("cvar") else 16384
    
    TRAINING_CONFIG = {
        "num_epochs": 200,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 30,
        "train_split": 0.7,
        "val_split": 0.2,
        "test_split": 0.1,
        "warmup_epochs": 5,
        "lambda_ramp_epochs": 15,
    }
    LAMBDA_VALUES = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 5e-3]
    ROLLOUT_HORIZON = 24

else:
    raise ValueError("EXPERIMENT_MODE must be 'SCOUT' or 'FULL'")


# =============================================================================
# TABLE EXPORT FUNCTIONS (for Paper Tables IV and V)
# =============================================================================

def export_per_state_fidelity_csv(results: list, save_path: str) -> pd.DataFrame:
    """
    Export per-state RMSE results to CSV for Paper Table IV.
    
    Args:
        results: List of dicts from analyze_per_state_fidelity()
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with the exported data
    """
    if not results:
        print(f"  WARNING: No results to export for per-state fidelity")
        return pd.DataFrame()
    
    rows = []
    for r in results:
        row = {
            "Model": r["model_name"],
            "T_air_bias_C": r.get("t_air_bias", float('nan')),
        }
        # Add per-state RMSE columns
        for state in ["T_air", "T_env", "T_int", "T_rad1", "T_rad2", "T_ret"]:
            row[f"{state}_RMSE_C"] = r.get(f"{state}_rmse", float('nan'))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"  Table IV (per-state RMSE) saved: {save_path}")
    return df


def export_bias_table_csv(bias_results: list, save_path: str) -> pd.DataFrame:
    """
    Export T_air bias results to CSV for Paper Table V.
    
    Args:
        bias_results: List of dicts from check_temperature_bias()
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with the exported data
    """
    if not bias_results:
        print(f"  WARNING: No results to export for bias table")
        return pd.DataFrame()
    
    rows = []
    for r in bias_results:
        t_air_bias = r.get("t_air_bias", float('nan'))
        status = "OK" if abs(t_air_bias) < 0.3 else "WARNING" if abs(t_air_bias) < 0.5 else "CRITICAL"
        rows.append({
            "Model": r["model"],
            "T_air_Bias_C": t_air_bias,
            "T_air_RMSE_C": r.get("t_air_rmse", float('nan')),
            "Status": status,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"  Table V (T_air bias) saved: {save_path}")
    return df


def generate_methods_summary(
    loss_config: dict, 
    training_config: dict, 
    lambda_values: list,
    clamp_bounds: dict,
    save_path: str
) -> str:
    """
    Generate a text summary of key parameters for the Paper Methods section.
    
    This creates a reference document to ensure paper text matches code configuration.
    
    Args:
        loss_config: LOSS_CONFIG dictionary
        training_config: TRAINING_CONFIG dictionary  
        lambda_values: List of lambda values used
        clamp_bounds: Clamp bounds dictionary (after correction)
        save_path: Path to save the summary
        
    Returns:
        Summary text string
    """
    # Format lambda values in scientific notation
    lambda_str = ", ".join([f"{lam:.0e}" for lam in lambda_values])
    
    summary = f"""
================================================================================
RAMC TRAINING CONFIGURATION SUMMARY (for Paper Methods Section)
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This file documents the exact configuration used in training.
Use this to ensure paper text matches code implementation.

--------------------------------------------------------------------------------
RISK CONFIGURATION
--------------------------------------------------------------------------------
Risk Operator: {loss_config.get('risk_operator', 'N/A')}
CVaR Alpha (α): {loss_config.get('cvar_alpha', 'N/A')}
CVaR Method: {loss_config.get('cvar_method', 'N/A')}
CVaR Optimization Steps: {loss_config.get('cvar_n_steps', 'N/A')}

Lambda (λ) Grid: [{lambda_str}]
  Paper notation: λ ∈ {{{lambda_str}}}

--------------------------------------------------------------------------------
PERTURBATION CONFIGURATION
--------------------------------------------------------------------------------
Training K (num_perturbations): {loss_config.get('num_perturbations', 'N/A')}
Evaluation K: {DEFAULT_EVAL_K}
Antithetic Sampling: {loss_config.get('use_antithetic', 'N/A')}

Perturbation Standard Deviations:
  σ_state: {loss_config.get('sigma_state', 'N/A')} °C (all state variables)
  σ_rad_scale: {loss_config.get('sigma_rad_scale', 'N/A')} (multiplier for radiator states)
  σ_T_supply: {loss_config.get('sigma_T_supply', 'N/A')} °C
  σ_mdot: {loss_config.get('sigma_mdot', 'N/A')} kg/s
  σ_T_out: {loss_config.get('sigma_T_out', 'N/A')} °C
  σ_Q_solar: {loss_config.get('sigma_Q_solar', 'N/A')} W
  σ_Q_internal: {loss_config.get('sigma_Q_internal', 'N/A')} W

Occupancy Flip Probability: {loss_config.get('p_occupancy_flip', 'N/A')}
Nominal Internal Gains: {loss_config.get('q_internal_nominal', 'N/A')} W

--------------------------------------------------------------------------------
DISTURBANCE CLAMP BOUNDS
--------------------------------------------------------------------------------
These bounds are derived from the actual dataset ranges:

  T_out: {clamp_bounds.get('dist', {}).get('T_out', 'N/A')} °C
  Q_solar: {clamp_bounds.get('dist', {}).get('Q_solar', 'N/A')} W
    Note: Previous incorrect value was (0, 1500) W which clipped 99% of data
  Q_internal: {clamp_bounds.get('dist', {}).get('Q_internal', 'N/A')} W
    Note: Consistent with q_internal_nominal = 40000.0 W

State Clamp Bounds (from training data quantiles 0.1%-99.9%):
"""
    
    if clamp_bounds and "state" in clamp_bounds:
        for name, (lo, hi) in clamp_bounds["state"].items():
            summary += f"  {name}: [{lo:.2f}, {hi:.2f}] °C\n"
    
    summary += f"""
Control Clamp Bounds (from training data quantiles):
"""
    
    if clamp_bounds and "control" in clamp_bounds:
        for name, (lo, hi) in clamp_bounds["control"].items():
            unit = "°C" if "T_" in name else "kg/s"
            summary += f"  {name}: [{lo:.2f}, {hi:.2f}] {unit}\n"
    
    summary += f"""
--------------------------------------------------------------------------------
STAGE COST CONFIGURATION
--------------------------------------------------------------------------------
Comfort Bounds: {loss_config.get('comfort_bounds', 'N/A')} °C
Comfort Hinge Function: {loss_config.get('comfort_hinge', 'N/A')}
Comfort Beta (softplus smoothness): {loss_config.get('comfort_beta', 'N/A')}
Timestep: {loss_config.get('dt_minutes', 'N/A')} minutes
District Heating Cost Rate: {loss_config.get('energy_cost_rate', 'N/A')} SEK/kWh
Weight Comfort (w_comfort): {loss_config.get('w_comfort', 'N/A')}
Weight Energy (w_energy): {loss_config.get('w_energy', 'N/A')}
Cost Scale: {loss_config.get('cost_scale', 'N/A')}
Return Temperature Index: {loss_config.get('t_ret_index', 'N/A')}

--------------------------------------------------------------------------------
FIDELITY CONFIGURATION
--------------------------------------------------------------------------------
MSE Normalization: {loss_config.get('mse_normalize', 'N/A')}
Fidelity Weights: {loss_config.get('fidelity_weights', 'N/A')}
  [T_air, T_env, T_int, T_rad1, T_rad2, T_ret]
  Note: T_air weighted 4x (regulated output, comfort constraints);
        T_ret weighted 1x (enters energy proxy directly);
        T_rad1, T_rad2 weighted 0.5x (internal discretization nodes)

--------------------------------------------------------------------------------
TRAINING CONFIGURATION
--------------------------------------------------------------------------------
Epochs: {training_config.get('num_epochs', 'N/A')}
Batch Size: {training_config.get('batch_size', 'N/A')}
Learning Rate: {training_config.get('learning_rate', 'N/A')}
Weight Decay: {training_config.get('weight_decay', 'N/A')}
Gradient Clip Norm: {training_config.get('grad_clip_norm', 'N/A')}
Early Stopping Patience: {training_config.get('early_stopping_patience', 'N/A')}
Warmup Epochs (λ=0): {training_config.get('warmup_epochs', 'N/A')}
Lambda Ramp Epochs: {training_config.get('lambda_ramp_epochs', 'N/A')}

Data Split:
  Train: {training_config.get('train_split', 'N/A') * 100:.0f}%
  Validation: {training_config.get('val_split', 'N/A') * 100:.0f}%
  Test: {training_config.get('test_split', 'N/A') * 100:.0f}%
  Split Mode: time-based (sequential, no shuffle)

--------------------------------------------------------------------------------
SUGGESTED PAPER TEXT (copy-paste ready)
--------------------------------------------------------------------------------
"We train RAMC models with CVaR at α={loss_config.get('cvar_alpha', 0.9)}, 
using K={loss_config.get('num_perturbations', 32)} perturbation samples during 
training and K={DEFAULT_EVAL_K} for evaluation. The risk-aversion parameter 
λ is swept over {{{lambda_str}}}. Perturbations follow Gaussian distributions 
with σ_state={loss_config.get('sigma_state', 1.0)}°C for state variables and 
calibrated standard deviations for controls and disturbances (see Table X). 
Antithetic sampling is used for variance reduction. Training uses Adam optimizer 
with learning rate {training_config.get('learning_rate', 1e-3)}, batch size 
{training_config.get('batch_size', 2048)}, and early stopping with patience 
{training_config.get('early_stopping_patience', 30)} epochs."

================================================================================
"""
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Methods summary saved: {save_path}")
    
    return summary


# =============================================================================
# GPU OPTIMIZATION HELPERS
# =============================================================================

def enable_tf32_if_available():
    if not torch.cuda.is_available():
        return

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print("TF32 enabled")


def set_random_seeds(seed: int = 42):
    import random as _random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_experiment_header():
    print("=" * 90)
    print(f"RAMC TRAINING EXPERIMENT - MODE: {EXPERIMENT_MODE} | RISK: {RISK_OPERATOR}")
    print("=" * 90)
    print(f"Experiment Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results Dir     : {RESULTS_DIR}")
    print(f"Device          : {DEVICE}")
    print(f"Risk Operator   : {RISK_OPERATOR}")
    print(f"Training K      : {DEFAULT_TRAIN_K}")
    
    if DEVICE == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print()

    print("IMPROVEMENTS ENABLED:")
    print("  P1:  Baseline hygiene (Raw MSE vs Fidelity naming)")
    print("  P2:  Configurable clamp bounds from TRAIN data quantiles")
    print("  P3:  Open-loop multi-step rollout evaluation (TEST only)")
    print("  P4:  Energy proxy (Q) error metrics")
    print("  P5A: Antithetic sampling for variance reduction")
    print("  P7:  Episode segmentation for valid rollouts")
    print("  P9:  CVaR RU works under torch.no_grad()")
    print("  P10: Decomposed risk (comfort vs energy)")
    print("  P11: Occupancy-conditional evaluation")
    print("  Skip risk computation when lambda=0")
    print("  Test-only rollout evaluation")
    print("  Data-derived disturbance bounds (Q_solar: 40kW, Q_internal: 1kW)")
    print("  CSV export for Tables IV and V")
    print("  Methods summary generator for paper")
    print("  Rollout-aware fidelity (alpha · L_fidelity_rollout)")
    print("=" * 90)


def load_and_prepare_data():
    """Load data with episode segmentation and build clamp bounds from TRAIN only."""
    print("\nSTEP 1: DATA LOADING AND PREPARATION")
    print("-" * 70)

    if not os.path.exists(CSV_DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {CSV_DATA_PATH}")

    print(f"Loading data from: {CSV_DATA_PATH}")

    # Now returns split_indices
    train_loader, val_loader, test_loader, full_dataset, split_indices = create_dataloaders(
        csv_file_path=CSV_DATA_PATH,
        batch_size=TRAINING_CONFIG["batch_size"],
        train_split=TRAINING_CONFIG["train_split"],
        val_split=TRAINING_CONFIG["val_split"],
        test_split=TRAINING_CONFIG["test_split"],
        device=DEVICE,
        split_mode='time',
        expected_dt_seconds=600,
    )

    from thermal_dynamics_net import PerturbedLabelDataset
    # When the ablation matrix is active, attach pre-generated perturbed
    # labels to the training dataset. 
    PERTURBED_LABELS_PATH = "RAMC_perturbed_labels_K4.npz"
    global USE_ABLATIONS
    USE_ABLATIONS = os.path.exists(PERTURBED_LABELS_PATH)
    if USE_ABLATIONS:
        full_dataset_wrapped = PerturbedLabelDataset(full_dataset, PERTURBED_LABELS_PATH)
        print(f"A1 perturbed-labels file found, ablation matrix will run.")
        
        from torch.utils.data import DataLoader, Subset
        # We MUST re-wrap the train_loader because it was created with the unwrapped dataset!
        train_dataset = Subset(full_dataset_wrapped, split_indices['train'])
        pin_memory = (DEVICE == 'cuda')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=TRAINING_CONFIG["batch_size"], 
            shuffle=True, 
            num_workers=2, 
            pin_memory=pin_memory
        )
        
        # Re-wrap val_loader with the same wrapper so validation also sees
        # the perturbed labels. Model selection for A1 then weights both
        # fidelity and the perturbation MSE, matching how RAMC validation
        # weights both fidelity and CVaR.
        val_dataset = Subset(full_dataset_wrapped, split_indices['val'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory,
        )
        
        full_dataset = full_dataset_wrapped
    else:
        print(f"No perturbed-labels file at {PERTURBED_LABELS_PATH}; "
              f"skipping A1 and A2 (run generate_perturbed_labels.py first).")

    # Build clamp bounds from TRAINING data only (no test leakage)
    print("\nFIX: Building clamp bounds from TRAINING data only...")
    clamp_bounds = build_clamp_bounds_from_dataset(
        full_dataset, 
        qlo=0.001, 
        qhi=0.999,
        indices=split_indices['train'],  # Train only
    )
    
    # Disturbance bounds based on the actual dataset range
    print("\nUsing physical/data-derived bounds for disturbances:")
    print("  Q_solar: (0, 40000) W — covers dataset peak ~25kW with buffer")
    print("  Q_internal: (0, 100000) W — covers building-level gains range ~3.6k-79k W")
    print("")
    
    clamp_bounds["dist"] = {
        "T_out": (-30.0, 40.0),
        "Q_solar": (0.0, 40000.0),
        "Q_internal": (0.0, 100000.0),  # Matches actual building-level gains range
    }
    
    print("  Disturbance bounds:")
    for k, v in clamp_bounds["dist"].items():
        print(f"    {k}: {v}")
    
    # Verify against dataset quantiles
    print("\n  Dataset disturbance quantiles (for verification):")
    dist_cols = ["T_out_k", "Q_solar_trans_k", "Q_internal_k"]
    train_df = full_dataset.df.iloc[split_indices['train']]
    for col, name in zip(dist_cols, ["T_out", "Q_solar", "Q_internal"]):
        q_min = train_df[col].quantile(0.001)
        q_max = train_df[col].quantile(0.999)
        print(f"    {name}: data range [{q_min:.1f}, {q_max:.1f}]")
    
    # Keep train-quantile bounds for states and controls
    print("\nKeeping TRAIN-QUANTILE bounds for states and controls:")
    for category in ["state", "control"]:
        print(f"  {category}:")
        for name, (lo, hi) in clamp_bounds[category].items():
            print(f"    {name}: [{lo:.2f}, {hi:.2f}]")
    
    # Update LOSS_CONFIG with computed bounds
    LOSS_CONFIG["clamp_bounds"] = clamp_bounds
    # Safety check: fidelity_weights must match state dimension and t_ret_index
    fw = LOSS_CONFIG["fidelity_weights"]
    assert isinstance(fw, (list, tuple)), "fidelity_weights must be a list/tuple"
    assert len(fw) == MODEL_CONFIG["state_dim"], (
        f"Expected {MODEL_CONFIG['state_dim']} fidelity weights, got {len(fw)}"
    )
    assert 0 <= LOSS_CONFIG["t_ret_index"] < len(fw), (
        f"t_ret_index={LOSS_CONFIG['t_ret_index']} out of range for fidelity_weights (len={len(fw)})"
    )
    # P7: Report episode information
    ep_lengths = full_dataset.get_episode_lengths()
    print(f"\nEpisode information:")
    print(f"  Number of episodes: {len(ep_lengths)}")
    print(f"  Episode length range: [{min(ep_lengths.values())}, {max(ep_lengths.values())}] samples")
    
    # P3: Report valid rollout starts for test horizon
    valid_starts = full_dataset.valid_rollout_starts(ROLLOUT_HORIZON)
    print(f"  Valid rollout starts (H={ROLLOUT_HORIZON}): {len(valid_starts)}")
    
    # Report test-only valid starts
    test_min = int(split_indices['test'].min())
    test_max = int(split_indices['test'].max())
    test_valid = valid_starts[(valid_starts >= test_min) & (valid_starts <= test_max - ROLLOUT_HORIZON + 1)]
    print(f"  Test-only valid rollout starts: {len(test_valid)}")

    print(f"\nData loading completed")
    print(f"  Training batches  : {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches      : {len(test_loader)}")

    return train_loader, val_loader, test_loader, full_dataset, split_indices, clamp_bounds


def run_training_experiment(train_loader, val_loader, loss_config, full_dataset=None, split_indices=None):
    """Run the main RAMC training experiment."""
    print("\nSTEP 2: MODEL TRAINING (RAMC)")
    print("-" * 70)
    print(f"Risk operator         : {loss_config.get('risk_operator', 'std')}")
    print(f"Training K            : {loss_config.get('num_perturbations', 4)}")
    print(f"Antithetic sampling   : {loss_config.get('use_antithetic', True)}")
    print(f"Rollout fidelity α    : {loss_config.get('alpha_rollout', 0.0)}")
    print(f"Rollout horizon H_r   : {loss_config.get('rollout_horizon', 6)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    training_manager = TrainingManager(
        device=DEVICE,
        save_dir=RESULTS_DIR,
        model_config=MODEL_CONFIG,
    )

    print("Training models:")
    print("  - Raw MSE baseline")
    print("  - Fidelity baseline")
    print(f"  - RAMC variants for lambda_risk in: {LAMBDA_VALUES}")

    # Pass train_indices so RAMCTrainer can build the RolloutSequenceSampler
    train_indices = split_indices["train"] if split_indices is not None else None

    start_time = time.time()

    results = training_manager.train_all_baselines(
        train_loader=train_loader,
        val_loader=val_loader,
        lambda_values=LAMBDA_VALUES,
        num_epochs=TRAINING_CONFIG["num_epochs"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        grad_clip_norm=TRAINING_CONFIG["grad_clip_norm"],
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"],
        warmup_epochs=TRAINING_CONFIG.get("warmup_epochs", 5),
        lambda_ramp_epochs=TRAINING_CONFIG.get("lambda_ramp_epochs", 15),
        loss_params=loss_config,
        include_mse_baseline=True,
        include_fidelity_only=True,
        include_ablations=USE_ABLATIONS,
        ablation_lambda=1.5e-3,
        verbose=True,
        dataset=full_dataset,
        train_indices=train_indices,
    )

    dt = time.time() - start_time
    print(f"\nRAMC training completed in {dt/60:.1f} minutes")

    return training_manager, results


def run_rollout_evaluation(training_manager, full_dataset, split_indices, loss_config):
    """Open-loop multi-step rollout evaluation on TEST SET ONLY."""
    print("\nSTEP 3: OPEN-LOOP ROLLOUT EVALUATION (TEST SET ONLY)")
    print("-" * 70)
    print(f"Rollout horizon: {ROLLOUT_HORIZON} steps ({ROLLOUT_HORIZON * 10} minutes)")
    
    test_min = int(split_indices['test'].min())
    test_max = int(split_indices['test'].max())
    print(f"Test set index range: [{test_min}, {test_max}]")
    
    models = {}
    
    # BaseTrainer.load_model now automatically calls enable_normalization_if_stats_present()
    if training_manager.mse_trainer is not None:
        ckpt = os.path.join(RESULTS_DIR, f"{training_manager.mse_trainer.trainer_name}_best.pth")
        if os.path.exists(ckpt):
            training_manager.mse_trainer.load_model(ckpt)
            models["Raw MSE Baseline"] = training_manager.mse_trainer.model
    
    if training_manager.fidelity_trainer is not None:
        ckpt = os.path.join(RESULTS_DIR, f"{training_manager.fidelity_trainer.trainer_name}_best.pth")
        if os.path.exists(ckpt):
            training_manager.fidelity_trainer.load_model(ckpt)
            models["Fidelity Baseline"] = training_manager.fidelity_trainer.model
    
    for lam in sorted(training_manager.ramc_trainers.keys()):
        trainer = training_manager.ramc_trainers[lam]
        ckpt = os.path.join(RESULTS_DIR, f"{trainer.trainer_name}_best.pth")
        if os.path.exists(ckpt):
            trainer.load_model(ckpt)
            models[f"RAMC λ={lam}"] = trainer.model
    
    if len(models) == 0:
        print("No models found for rollout evaluation.")
        return {}
    
    print(f"Evaluating {len(models)} models: {list(models.keys())}")
    
    # Run comparison on TEST SET ONLY
    rollout_results = compare_rollouts_multiple_models(
        models=models,
        dataset=full_dataset,
        horizon=ROLLOUT_HORIZON,
        num_rollouts=100,
        device=DEVICE,
        save_dir=RESULTS_DIR,
        seed=42,
        min_start_idx=test_min,
        max_start_idx=test_max - ROLLOUT_HORIZON + 1,
    )
    
    # Save rollout metrics
    rollout_summary = {}
    for name, res in rollout_results.items():
        rollout_summary[name] = {
            "T_air_rmse_final": res["T_air_rmse_final"],
            "T_ret_rmse_final": res["T_ret_rmse_final"],
            "T_air_growth": res["T_air_growth"],
            "T_ret_growth": res["T_ret_growth"],
            "min_start_idx": res["min_start_idx"],
            "max_start_idx": res["max_start_idx"],
        }
    
    with open(os.path.join(RESULTS_DIR, "rollout_metrics.json"), "w") as f:
        json.dump(rollout_summary, f, indent=2)
    
    print("\nRollout evaluation completed (test set only).")
    return rollout_results


def check_temperature_bias(training_manager, loader, loss_config):
    """Sanity check: verify no large systematic T_air bias."""
    print("\nSTEP 4: T_AIR BIAS / FIDELITY SANITY CHECK")
    print("-" * 70)

    from ramc_losses import evaluate_on_loader

    models = []
    if training_manager.mse_trainer is not None:
        models.append(("Raw MSE Baseline", training_manager.mse_trainer))
    if training_manager.fidelity_trainer is not None:
        models.append(("Fidelity Baseline", training_manager.fidelity_trainer))
    for lam, trainer in sorted(training_manager.ramc_trainers.items()):
        models.append((f"RAMC λ={lam}", trainer))

    eval_config = dict(loss_config)
    eval_config["num_perturbations"] = DEFAULT_EVAL_K
    eval_config["lambda_risk"] = 1.0
    risk_op = str(eval_config.get("risk_operator", "")).strip().lower()
    if risk_op.startswith("cvar"):
        eval_config["cvar_method"] = "empirical"
        
    results = []

    for name, trainer in models:
        ckpt_path = os.path.join(RESULTS_DIR, f"{trainer.trainer_name}_best.pth")
        if not os.path.exists(ckpt_path):
            continue

        trainer.load_model(ckpt_path)
        trainer.model.eval()

        metrics = evaluate_on_loader(
            trainer.model,
            loader,
            loss_config=eval_config,
            device=DEVICE,
            compute_occupancy_split=True,
        )

        # Compute T_air bias
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                # Need to grab only the first 4 if the dataset is wrapped
                states, controls, disturbances, targets = batch[:4]
                states = states.to(DEVICE)
                controls = controls.to(DEVICE)
                disturbances = disturbances.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = trainer.model(states, controls, disturbances)
                all_preds.append(preds[:, 0].detach().cpu())
                all_targets.append(targets[:, 0].detach().cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        t_air_bias = float((all_preds.mean() - all_targets.mean()).item())

        results.append({
            "model": name,
            "t_air_rmse": float(metrics["t_air_rmse"]),
            "t_air_bias": t_air_bias,
            "mse_raw": float(metrics["mse_raw"]),
            "fidelity_loss": float(metrics["fidelity_loss"]),
            "expected_cost": float(metrics["expected_cost"]),
            "risk_loss": float(metrics["risk_loss"]),
            "risk_comfort": float(metrics["risk_comfort_loss"]),
            "risk_energy": float(metrics["risk_energy_loss"]),
            "Q_rmse": float(metrics["Q_rmse"]),
            "Q_bias": float(metrics["Q_bias"]),
        })

        warn = "WARNING" if abs(t_air_bias) > 0.5 else "OK"
        print(
            f"{name:<22} | T_air RMSE={metrics['t_air_rmse']:.4f} | "
            f"Bias={t_air_bias:+.4f} [{warn}] | "
            f"Q_rmse={metrics['Q_rmse']:.1f} W"
        )

    return results


def analyze_and_save_results(training_manager, test_loader, val_loader, loss_config, clamp_bounds):
    """Analyze results with extended metrics, save plots, and export tables."""
    print("\nSTEP 5: RESULTS ANALYSIS AND REPORTING")
    print("-" * 70)

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    training_manager.save_results_summary(filepath=summary_path)

    # Use larger K for evaluation
    analysis_config = dict(loss_config)
    analysis_config["num_perturbations"] = DEFAULT_EVAL_K
    analysis_config["lambda_risk"] = 1.0
    risk_op = str(analysis_config.get("risk_operator", "")).strip().lower()
    if risk_op.startswith("cvar"):
        analysis_config["cvar_method"] = "empirical"
    # =========================================================================
    # VALIDATION SET ANALYSIS
    # =========================================================================
    print("\nVALIDATION SET ANALYSIS")
    print("-" * 70)

    training_manager.analyze_tradeoff(
        val_loader,
        loss_config=analysis_config,
        save_path=os.path.join(RESULTS_DIR, "ramc_tradeoff_VAL.png"),
        selection_policy="combined",
        seeds=[42, 43, 44, 45, 46],
    )

    # Get fidelity results and export to CSV for Table IV
    fidelity_results_val = training_manager.analyze_per_state_fidelity(
        val_loader,
        save_path=os.path.join(RESULTS_DIR, "ramc_per_state_fidelity_VAL.png"),
    )
    
    # Export Table IV (validation)
    export_per_state_fidelity_csv(
        fidelity_results_val, 
        os.path.join(RESULTS_DIR, "table_IV_per_state_rmse_VAL.csv")
    )

    # =========================================================================
    # TEST SET ANALYSIS
    # =========================================================================
    print("\nTEST SET ANALYSIS")
    print("-" * 70)

    training_manager.analyze_tradeoff(
        test_loader,
        loss_config=analysis_config,
        save_path=os.path.join(RESULTS_DIR, "ramc_tradeoff_TEST.png"),
        selection_policy="combined",
        seeds=[42, 43, 44, 45, 46],
    )

    # Get fidelity results and export to CSV for Table IV (test)
    fidelity_results_test = training_manager.analyze_per_state_fidelity(
        test_loader,
        save_path=os.path.join(RESULTS_DIR, "ramc_per_state_fidelity_TEST.png"),
    )
    
    # Export Table IV (test)
    export_per_state_fidelity_csv(
        fidelity_results_test, 
        os.path.join(RESULTS_DIR, "table_IV_per_state_rmse_TEST.csv")
    )

    training_manager.analyze_pareto_frontier(
        test_loader,
        loss_config=analysis_config,
        save_path=os.path.join(RESULTS_DIR, "ramc_pareto_frontier_TEST.png"),
        selection_policy="combined",
    )

    # =========================================================================
    # Generate Methods Summary for Paper
    # =========================================================================
    print("\nGENERATING METHODS SUMMARY FOR PAPER")
    print("-" * 70)
    
    generate_methods_summary(
        loss_config=loss_config,
        training_config=TRAINING_CONFIG,
        lambda_values=LAMBDA_VALUES,
        clamp_bounds=clamp_bounds,
        save_path=os.path.join(RESULTS_DIR, "paper_methods_summary.txt")
    )

    # =========================================================================
    # Save configuration
    # =========================================================================
    config_path = os.path.join(RESULTS_DIR, "experiment_config.json")
    
    # Make loss_config serializable (convert tuples to lists, etc.)
    loss_config_serializable = {}
    for k, v in loss_config.items():
        if isinstance(v, tuple):
            loss_config_serializable[k] = list(v)
        elif isinstance(v, dict):
            # Handle nested dicts like clamp_bounds
            loss_config_serializable[k] = {
                kk: list(vv) if isinstance(vv, tuple) else vv 
                for kk, vv in v.items()
            } if v else v
        else:
            loss_config_serializable[k] = v
    
    config_to_save = {
        "experiment_mode": EXPERIMENT_MODE,
        "risk_operator": RISK_OPERATOR,
        "training_config": TRAINING_CONFIG,
        "model_config": MODEL_CONFIG,
        "loss_config": loss_config_serializable,
        "lambda_values": LAMBDA_VALUES,
        "rollout_horizon": ROLLOUT_HORIZON,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat(),
        "improvements_enabled": [
            "P1: Baseline hygiene",
            "P2: Configurable clamp bounds (train only)",
            "P3: Open-loop rollouts (test only)",
            "P4: Energy proxy metrics",
            "P5A: Antithetic sampling",
            "P7: Episode segmentation",
            "P9: CVaR no_grad fix",
            "P10: Decomposed risk",
            "P11: Occupancy split",
            "Skip risk when lambda=0",
            "Test-only rollout evaluation",
            "Data-derived disturbance bounds (Q_solar: 40kW, Q_internal: 1kW)",
            "NEW v5: CSV export for Tables IV and V",
            "NEW v5: Methods summary generator",
            "A1/A2: Perturbed Labels wrapping and mean logic"
        ],
    }
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_DIR}")


def create_final_report():
    """Create the experiment report."""
    print("\nSTEP 6: GENERATING FINAL REPORT")
    print("-" * 70)

    report_path = os.path.join(RESULTS_DIR, "RAMC_Experiment_Report.txt")

    with open(report_path, "w") as f:
        f.write("RISK-AWARE MODEL CONTROL (RAMC) TRAINING EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Risk Operator: {RISK_OPERATOR}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device        : {DEVICE}\n")
        if torch.cuda.is_available():
            f.write(f"GPU           : {torch.cuda.get_device_name(0)}\n")
        f.write(f"Results Dir   : {RESULTS_DIR}\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Risk operator: {RISK_OPERATOR}\n")
        f.write(f"Training K: {DEFAULT_TRAIN_K}\n")
        f.write(f"Evaluation K: {DEFAULT_EVAL_K}\n")
        if RISK_OPERATOR.strip().lower().startswith("cvar"):
            f.write(f"CVaR alpha: {LOSS_CONFIG['cvar_alpha']}\n")
        f.write(f"Batch size: {TRAINING_CONFIG['batch_size']}\n\n")

        f.write("FIXES IMPLEMENTED:\n")
        f.write("-" * 40 + "\n")
        f.write("1. risk_operator sanitization (.strip().lower())\n")
        f.write("2. Clamp bounds from TRAINING data only\n")
        f.write("3. Rollout evaluation on TEST set only\n")
        f.write("4. Occupancy split uses full cost config\n")
        f.write("5. Skip risk computation when lambda=0\n")
        f.write("6. Data-derived disturbance bounds:\n")
        f.write("   - Q_solar: (0, 40000) W [was 1500, clipped 99% of data]\n")
        f.write("   - Q_internal: (0, 1000) W [was 2000, inconsistent]\n\n")

        f.write("NEW IN v5:\n")
        f.write("-" * 40 + "\n")
        f.write("7. CSV export for Table IV (per-state RMSE)\n")
        f.write("8. CSV export for Table V (T_air bias)\n")
        f.write("9. Methods summary generator for paper\n\n")

        f.write("FILES GENERATED:\n")
        f.write("-" * 40 + "\n")
        f.write("- experiment_config.json\n")
        f.write("- experiment_summary.json\n")
        f.write("- rollout_metrics.json\n")
        f.write("- ramc_tradeoff_*.png\n")
        f.write("- ramc_per_state_fidelity_*.png\n")
        f.write("- ramc_pareto_frontier_*.png\n")
        f.write("- rollout_rmse_*_comparison.png\n")
        f.write("- table_IV_per_state_rmse_VAL.csv  [NEW]\n")
        f.write("- table_IV_per_state_rmse_TEST.csv [NEW]\n")
        f.write("- table_V_bias.csv                 [NEW]\n")
        f.write("- paper_methods_summary.txt        [NEW]\n")
        f.write("- model checkpoints (*.pth)\n")
        f.write("- this report\n")

    print(f"Final report saved: {report_path}")


def main():
    try:
        enable_tf32_if_available()
        set_random_seeds(42)
        print_experiment_header()

        # Step 1: Load data with train-only clamp bounds
        # Also returns clamp_bounds for methods summary
        train_loader, val_loader, test_loader, full_dataset, split_indices, clamp_bounds = load_and_prepare_data()

        # Step 2: Train models
        training_manager, results = run_training_experiment(
            train_loader, val_loader, loss_config=LOSS_CONFIG,
            full_dataset=full_dataset, split_indices=split_indices,
        )

        # Step 3: Rollout evaluation (TEST ONLY)
        rollout_results = run_rollout_evaluation(
            training_manager, full_dataset, split_indices, LOSS_CONFIG
        )

        # Step 4: Bias check
        # Capture results and export to CSV for Table V
        bias_results = check_temperature_bias(training_manager, val_loader, LOSS_CONFIG)
        
        # Export Table V (T_air bias)
        print("\nExporting Table V (T_air bias)...")
        export_bias_table_csv(
            bias_results, 
            os.path.join(RESULTS_DIR, "table_V_bias.csv")
        )

        # Step 5: Analysis (Table IV export and methods summary)
        # Pass clamp_bounds for methods summary
        analyze_and_save_results(
            training_manager, test_loader, val_loader, LOSS_CONFIG, clamp_bounds
        )

        # Step 6: Report
        if EXPERIMENT_MODE == "FULL":
            create_final_report()
            print("\n" + "=" * 90)
            print("RAMC FULL EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 90)
            print(f"\nAll results available in: {RESULTS_DIR}")
            print("\nKey output files for paper:")
            print("  - ramc_tradeoff_VAL.png / ramc_tradeoff_TEST.png  (Fig. 5)")
            print("  - table_IV_per_state_rmse_VAL.csv                 (Table IV)")
            print("  - table_V_bias.csv                                (Table V)")
            print("  - paper_methods_summary.txt                       (Methods section reference)")
            print("  - ramc_pareto_frontier_TEST.png                   (Pareto analysis)")
        else:
            print("\nRAMC SCOUT RUN FINISHED")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()