# -*- coding: utf-8 -*-
"""
Hyperparameter Sensitivity Study — Standalone Post-Hoc Script
==============================================================
Trains a small grid of models at a fixed representative λ, varying:
  - Learning rate: {5e-4, 1e-3}
  - Training K (perturbation samples): {16, 32}

Uses SCOUT-style reduced training (fewer epochs, early stopping)
to verify that results are not fragile to hyperparameter choices.

Run independently of the main training:
    python hyperparameter_sensitivity.py

Outputs:
    - hyperparameter_sensitivity.csv: results table
    - hyperparameter_sensitivity.json: full results
    - hyperparameter_sensitivity_plot.png: visual summary

Author: Nima Monghasemi
"""

import os
import json
import time
import copy
from datetime import datetime
from itertools import product
from typing import Dict, Any, Optional

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from thermal_dynamics_net import (
    create_dataloaders,
    ThermalDynamicsNet,
    build_clamp_bounds_from_dataset,
)
from trainers import RAMCTrainer
from ramc_losses import evaluate_on_loader


# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_DATA_PATH = "RAMC_training_data_N3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory
OUTPUT_DIR = "results/hyperparameter_sensitivity"

# Fixed lambda for sensitivity study (representative mid-range value)
LAMBDA_FIXED = 5e-4

# Hyperparameter grid to sweep
LEARNING_RATES = [5e-4, 1e-3]
K_TRAIN_VALUES = [16, 32]

# SCOUT-style reduced training settings
SCOUT_EPOCHS = 50
SCOUT_PATIENCE = 10
WARMUP_EPOCHS = 3
LAMBDA_RAMP_EPOCHS = 8

# Model architecture (must match main training)
MODEL_CONFIG = {
    "state_dim": 6,
    "control_dim": 2,
    "disturbance_dim": 3,
    "hidden_dims": [256, 256, 128],
    "dropout_rate": 0.0,
}

# Data split (must match main training)
DATA_CONFIG = {
    "batch_size": 2048,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
}

# Base loss configuration (perturbation params, cost params, etc.)
# K and lr will be overridden per run
BASE_LOSS_CONFIG = {
    "risk_operator": "cvar",
    "cvar_alpha": 0.9,
    "cvar_method": "ru",
    "cvar_n_steps": 10,
    "use_antithetic": True,
    "perturb_forward_chunk_size": 16384,

    "sigma_state": 1.0,
    "sigma_rad_scale": 0.5,
    "sigma_T_supply": 0.5,
    "sigma_mdot": 0.01,
    "sigma_T_out": 1.0,
    "sigma_Q_solar": 500.0,
    "sigma_Q_internal": 5000.0,       # RECALIBRATED
    "clamp_physical": True,
    "p_occupancy_flip": 0.02,
    "q_internal_nominal": 40000.0,    # RECALIBRATED

    "comfort_bounds": (20.0, 22.0),
    "comfort_hinge": "softplus",
    "comfort_beta": 0.5,
    "dt_minutes": 10.0,
    "energy_cost_rate": 0.9,
    "w_comfort": 63.0,
    "w_energy": 1.0,
    "cost_scale": 1.0,
    "t_ret_index": 5,

    "mse_normalize": True,
    # fidelity_weights[i] corresponds to state dimension i (must match model/dataset ordering).
    # We emphasize state[0] (T_air) and state[t_ret_index=5] (T_ret) equally.
    "fidelity_weights": [2.0, 1.0, 1.0, 0.5, 0.5, 2.0],

    # Will be set after data loading
    "clamp_bounds": None,
}

# Evaluation K (larger than training K for stable estimates)
EVAL_K = 256

# Number of evaluation seeds for uncertainty estimation
EVAL_SEEDS = [42, 43, 44]


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int = 42):
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_ci(values):
    """Compute mean and 95% CI."""
    arr = np.array(values, dtype=np.float64)
    m = float(np.mean(arr))
    if len(arr) <= 1:
        return m, 0.0
    ci = float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return m, ci


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("HYPERPARAMETER SENSITIVITY STUDY")
    print("=" * 80)
    print(f"Fixed λ           : {LAMBDA_FIXED}")
    print(f"Learning rates    : {LEARNING_RATES}")
    print(f"Training K values : {K_TRAIN_VALUES}")
    print(f"Scout epochs      : {SCOUT_EPOCHS}")
    print(f"Early stopping    : patience={SCOUT_PATIENCE}")
    print(f"Grid size         : {len(LEARNING_RATES) * len(K_TRAIN_VALUES)} runs")
    print(f"Device            : {DEVICE}")
    print()

    # --- Load data ---
    print("Loading data...")
    train_loader, val_loader, test_loader, full_dataset, split_indices = create_dataloaders(
        csv_file_path=CSV_DATA_PATH,
        batch_size=DATA_CONFIG["batch_size"],
        train_split=DATA_CONFIG["train_split"],
        val_split=DATA_CONFIG["val_split"],
        test_split=DATA_CONFIG["test_split"],
        device=DEVICE,
        split_mode="time",
    )

    # Build clamp bounds from TRAIN only
    clamp_bounds = build_clamp_bounds_from_dataset(
        full_dataset, qlo=0.001, qhi=0.999, indices=split_indices["train"],
    )
    clamp_bounds["dist"] = {
        "T_out": (-30.0, 40.0),
        "Q_solar": (0.0, 40000.0),
        "Q_internal": (0.0, 100000.0),
    }

    BASE_LOSS_CONFIG["clamp_bounds"] = clamp_bounds

    # --- Run grid ---
    results = []
    total_start = time.time()

    for lr, k_train in product(LEARNING_RATES, K_TRAIN_VALUES):
        run_name = f"lr_{lr:.0e}_K{k_train}"
        run_dir = os.path.join(OUTPUT_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"RUN: lr={lr}, K_train={k_train}")
        print(f"{'='*60}")

        set_seed(42)

        # Create fresh model
        model = ThermalDynamicsNet(**MODEL_CONFIG)
        model.compute_normalization(train_loader)

        # Set up loss config for this run
        loss_config = copy.deepcopy(BASE_LOSS_CONFIG)
        loss_config["num_perturbations"] = k_train

        # Create trainer
        trainer = RAMCTrainer(
            model=model,
            lambda_risk=LAMBDA_FIXED,
            learning_rate=lr,
            device=DEVICE,
            save_dir=run_dir,
            grad_clip_norm=1.0,
            weight_decay=1e-5,
            **loss_config,
        )

        # Train (SCOUT-style)
        t0 = time.time()
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=SCOUT_EPOCHS,
            early_stopping_patience=SCOUT_PATIENCE,
            warmup_epochs=WARMUP_EPOCHS,
            lambda_ramp_epochs=LAMBDA_RAMP_EPOCHS,
            verbose=False,
        )
        train_time = time.time() - t0

        # Load best checkpoint for evaluation
        best_ckpt = os.path.join(run_dir, f"{trainer.trainer_name}_best.pth")
        if os.path.exists(best_ckpt):
            trainer.load_model(best_ckpt)
        trainer.model.eval()

        # Evaluate on VAL set with multiple seeds
        eval_config = copy.deepcopy(loss_config)
        eval_config["num_perturbations"] = EVAL_K
        eval_config["lambda_risk"] = LAMBDA_FIXED
        risk_op = str(eval_config.get("risk_operator", "")).strip().lower()
        if risk_op.startswith("cvar"):
            eval_config["cvar_method"] = "empirical"

        val_fidelity_list = []
        val_risk_list = []
        val_risk_comfort_list = []
        val_risk_energy_list = []
        val_tair_rmse_list = []

        for seed in EVAL_SEEDS:
            set_seed(seed)
            metrics = evaluate_on_loader(
                trainer.model, val_loader,
                loss_config=eval_config,
                device=DEVICE,
                compute_occupancy_split=False,
            )
            val_fidelity_list.append(metrics["fidelity_loss"])
            val_risk_list.append(metrics["risk_loss"])
            val_risk_comfort_list.append(metrics["risk_comfort_loss"])
            val_risk_energy_list.append(metrics["risk_energy_loss"])
            val_tair_rmse_list.append(metrics["t_air_rmse"])

        fid_mean, fid_ci = mean_ci(val_fidelity_list)
        risk_mean, risk_ci = mean_ci(val_risk_list)
        risk_c_mean, risk_c_ci = mean_ci(val_risk_comfort_list)
        risk_e_mean, risk_e_ci = mean_ci(val_risk_energy_list)
        rmse_mean, rmse_ci = mean_ci(val_tair_rmse_list)

        result = {
            "run_name": run_name,
            "learning_rate": lr,
            "K_train": k_train,
            "lambda": LAMBDA_FIXED,
            "best_epoch": trainer.best_epoch,
            "train_time_s": train_time,
            "val_fidelity_mean": fid_mean,
            "val_fidelity_ci": fid_ci,
            "val_risk_mean": risk_mean,
            "val_risk_ci": risk_ci,
            "val_risk_comfort_mean": risk_c_mean,
            "val_risk_comfort_ci": risk_c_ci,
            "val_risk_energy_mean": risk_e_mean,
            "val_risk_energy_ci": risk_e_ci,
            "val_tair_rmse_mean": rmse_mean,
            "val_tair_rmse_ci": rmse_ci,
        }
        results.append(result)

        print(f"  Best epoch: {trainer.best_epoch}")
        print(f"  Train time: {train_time:.0f}s")
        print(f"  Val fidelity: {fid_mean:.5f} ± {fid_ci:.5f}")
        print(f"  Val risk:     {risk_mean:.5f} ± {risk_ci:.5f}")
        print(f"  Val T_air RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}")

    total_time = time.time() - total_start

    # --- Save results ---
    df = pd.DataFrame(results)

    csv_path = os.path.join(OUTPUT_DIR, "hyperparameter_sensitivity.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nResults CSV: {csv_path}")

    json_path = os.path.join(OUTPUT_DIR, "hyperparameter_sensitivity.json")
    output = {
        "generated": datetime.now().isoformat(),
        "total_time_s": total_time,
        "config": {
            "lambda_fixed": LAMBDA_FIXED,
            "learning_rates": LEARNING_RATES,
            "K_train_values": K_TRAIN_VALUES,
            "scout_epochs": SCOUT_EPOCHS,
            "eval_K": EVAL_K,
            "eval_seeds": EVAL_SEEDS,
        },
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results JSON: {json_path}")

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print("SENSITIVITY RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run':<20} {'lr':>8} {'K':>4} {'Fidelity':>14} {'Risk':>14} {'T_air RMSE':>14}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['run_name']:<20} {row['learning_rate']:>8.0e} {row['K_train']:>4d} "
              f"{row['val_fidelity_mean']:>8.5f}±{row['val_fidelity_ci']:.5f} "
              f"{row['val_risk_mean']:>8.5f}±{row['val_risk_ci']:.5f} "
              f"{row['val_tair_rmse_mean']:>8.4f}±{row['val_tair_rmse_ci']:.4f}")
    print("-" * 80)

    # Check stability: is the ranking consistent?
    best_run = df.loc[df["val_risk_mean"].idxmin()]
    worst_run = df.loc[df["val_risk_mean"].idxmax()]
    spread = (worst_run["val_risk_mean"] - best_run["val_risk_mean"]) / best_run["val_risk_mean"] * 100

    print(f"\nRisk spread across grid: {spread:.1f}%")
    if spread < 10:
        print("  -> Results are STABLE across hyperparameter choices (spread < 10%)")
        print("  -> Paper claim: 'We did not aggressively tune; results are stable.'")
    elif spread < 25:
        print("  -> Results show MODERATE sensitivity to hyperparameters")
        print("  -> Paper claim: 'Results are moderately sensitive to lr/K; we selected based on validation.'")
    else:
        print("  -> Results show HIGH sensitivity to hyperparameters")
        print("  -> Consider investigating further or narrowing the grid.")

    # --- Plot ---
    print("\nGenerating sensitivity plot...")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for k_val in K_TRAIN_VALUES:
        mask = df["K_train"] == k_val
        subset = df[mask].sort_values("learning_rate")
        label = f"K={k_val}"

        axes[0].errorbar(subset["learning_rate"], subset["val_fidelity_mean"],
                         yerr=subset["val_fidelity_ci"], fmt="o-", capsize=4, label=label)
        axes[1].errorbar(subset["learning_rate"], subset["val_risk_mean"],
                         yerr=subset["val_risk_ci"], fmt="o-", capsize=4, label=label)
        axes[2].errorbar(subset["learning_rate"], subset["val_tair_rmse_mean"],
                         yerr=subset["val_tair_rmse_ci"], fmt="o-", capsize=4, label=label)

    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("Validation fidelity loss")
    axes[0].set_title("(a) Fidelity sensitivity")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel(f"Validation CVaR₀.₉ risk")
    axes[1].set_title("(b) Risk sensitivity")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Learning rate")
    axes[2].set_ylabel("T_air RMSE (°C)")
    axes[2].set_title("(c) Prediction accuracy sensitivity")
    axes[2].set_xscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Hyperparameter sensitivity (λ={LAMBDA_FIXED})", fontsize=12, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "hyperparameter_sensitivity_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {plot_path}")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
