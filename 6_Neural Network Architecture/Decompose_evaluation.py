# -*- coding: utf-8 -*-
"""
RAMC Decomposed Evaluation Table Generator
==========================================
Generates a table of decomposed risk metrics (comfort vs energy) for all
trained models, with tail diagnostics.

Tail diagnostics compare 3-4 models (Raw MSE -> Fidelity -> RAMC) to show
the progression of the tail behaviour. The ECDF figure is the main output.

Run after training is complete.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import pandas as pd

# Project modules
from thermal_dynamics_net import (
    create_dataloaders,
    ThermalDynamicsNet,
    RCModelDataset,
    build_clamp_bounds_from_dataset,
)
from ramc_losses import (
    calculate_ramc_loss,
    evaluate_on_loader,
    compute_energy_Q_W,
    stage_cost_ramc,
    risk_from_cost_samples,
    sample_gaussian_perturbations,
)
from tail_diagnostics import (
    summarize_tail, 
    plot_cost_tail_comparison, 
    plot_cost_ecdf_comparison,
    collect_cost_samples,  
)



# =============================================================================
# CONFIGURATION - update these paths to match the local setup
# =============================================================================

# Path to the training results directory
RESULTS_DIR = "RAMC_FULL_cvar/RAMC_FULL_cvar_20260211"  

# Path to the data file
CSV_DATA_PATH = "RAMC_training_data_N3.csv"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation configuration - use LARGER K for stable CVaR estimates
EVAL_K = 256  # Larger K gives more stable tail estimates

# Model configuration (must match training)
MODEL_CONFIG = {
    "state_dim": 6,
    "control_dim": 2,
    "disturbance_dim": 3,
    "hidden_dims": [256, 256, 128],
    "dropout_rate": 0.0,
}

# Loss configuration for evaluation
EVAL_LOSS_CONFIG = {
    "risk_operator": "cvar",
    "cvar_alpha": 0.9,
    "cvar_method": "empirical",
    "cvar_n_steps": 10,
    "num_perturbations": EVAL_K,
    "use_antithetic": True,
    
    # Perturbation sigmas
    "sigma_state": 1.0,
    "sigma_rad_scale": 0.5,
    "sigma_T_supply": 0.5,
    "sigma_mdot": 0.01,
    "sigma_T_out": 1.0,
    "sigma_Q_solar": 500.0,
    "sigma_Q_internal": 5000.0,
    "clamp_physical": True,
    
    # Stage cost parameters
    "comfort_bounds": (20.0, 22.0),
    "comfort_hinge": "softplus",
    "comfort_beta": 0.5,
    "dt_minutes": 10.0,
    "energy_cost_rate": 0.9,
    "w_comfort": 63,
    "w_energy": 1.0,
    "cost_scale": 1.0,
    "t_ret_index": 5,
    
    # Fidelity
    "mse_normalize": True,
    # fidelity_weights[i] corresponds to state dimension i (must match model/dataset ordering).
    # We emphasize state[0] (T_air) and state[t_ret_index=5] (T_ret) equally.
    "fidelity_weights": [2.0, 1.0, 1.0, 0.5, 0.5, 2.0],
    
    # clamp_bounds will be set after data loading
    "clamp_bounds": None,
}

# Data split configuration (must match training)
DATA_CONFIG = {
    "batch_size": 2048,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def find_model_checkpoints(results_dir: str) -> Dict[str, str]:
    """
    Find all model checkpoint files in the results directory.
    
    Returns:
        Dict mapping model names to checkpoint paths
    """
    checkpoints = {}
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_best.pth"):
            # Extract model name from filename
            model_name = filename.replace("_best.pth", "")
            checkpoints[model_name] = os.path.join(results_dir, filename)
    
    return checkpoints


def load_model(checkpoint_path: str, model_config: dict, device: str, verbose: bool = True) -> ThermalDynamicsNet:
    """Load a trained model from checkpoint with normalization fix and sanity checks."""
    model = ThermalDynamicsNet(**model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Enable normalization if statistics were loaded from checkpoint
    if hasattr(model, 'enable_normalization_if_stats_present'):
        enabled = model.enable_normalization_if_stats_present()
        
        if verbose:
            basename = os.path.basename(checkpoint_path)
            if enabled:
                print(f"    [OK] Normalization enabled for: {basename}")
                # Sanity check prints (as recommended by expert)
                print(f"         normalization_computed = {model.normalization_computed}")
                print(f"         input_std[:3]  = {model.input_std[:3].tolist()}")
                print(f"         output_std[:3] = {model.output_std[:3].tolist()}")
            else:
                print(f"    [WARN] No normalization stats found in: {basename}")
    else:
        # Fallback: manually check and enable
        input_mean_differs = torch.any(model.input_mean.abs() > 1e-6).item()
        output_mean_differs = torch.any(model.output_mean.abs() > 1e-6).item()
        input_std_differs = torch.any((model.input_std - 1.0).abs() > 1e-6).item()
        output_std_differs = torch.any((model.output_std - 1.0).abs() > 1e-6).item()
        
        if input_mean_differs or output_mean_differs or input_std_differs or output_std_differs:
            model.normalization_computed = True
            if verbose:
                print(f"    [OK] Normalization enabled (fallback) for: {os.path.basename(checkpoint_path)}")
    
    model.eval()
    return model


def compute_detailed_metrics(
    model: ThermalDynamicsNet,
    loader,
    loss_config: dict,
    device: str,
    num_seeds: int = 3,
) -> Dict[str, Any]:
    """
    Compute decomposed metrics with uncertainty estimates.
    
    Args:
        model: Trained model
        loader: Data loader (test set)
        loss_config: Loss configuration
        device: Computation device
        num_seeds: Number of random seeds for uncertainty estimation
        
    Returns:
        Dict with mean and std of all metrics
    """
    model.eval()
    
    # Collect metrics across multiple seeds
    all_metrics = {
        # Fidelity metrics
        "mse_raw": [],
        "fidelity_loss": [],
        "t_air_rmse": [],
        "t_ret_rmse": [],
        
        # Total risk
        "risk_total": [],
        
        # Decomposed risk
        "risk_comfort": [],
        "risk_energy": [],
        
        # Expected costs
        "expected_cost": [],
        "expected_comfort": [],
        "expected_energy": [],
        
        # Energy proxy
        "Q_rmse": [],
        "Q_mae": [],
        "Q_bias": [],
        
        # Tail quantiles (additional insight)
        "comfort_p90": [],
        "comfort_p95": [],
        "comfort_p99": [],
        "energy_p90": [],
        "energy_p95": [],
        "energy_p99": [],
    }
    
    for seed in range(num_seeds):
        set_seed(42 + seed)
        
        # Use evaluate_on_loader for standard metrics
        metrics = evaluate_on_loader(
            model, loader,
            loss_config=loss_config,
            device=device,
            compute_occupancy_split=True,
        )
        
        all_metrics["mse_raw"].append(metrics["mse_raw"])
        all_metrics["fidelity_loss"].append(metrics["fidelity_loss"])
        all_metrics["t_air_rmse"].append(metrics["t_air_rmse"])
        all_metrics["risk_total"].append(metrics["risk_loss"])
        all_metrics["risk_comfort"].append(metrics["risk_comfort_loss"])
        all_metrics["risk_energy"].append(metrics["risk_energy_loss"])
        all_metrics["expected_cost"].append(metrics["expected_cost"])
        all_metrics["expected_comfort"].append(metrics["expected_comfort"])
        all_metrics["expected_energy"].append(metrics["expected_energy_cost"])
        all_metrics["Q_rmse"].append(metrics["Q_rmse"])
        all_metrics["Q_mae"].append(metrics["Q_mae"])
        all_metrics["Q_bias"].append(metrics["Q_bias"])
        
        # Compute additional tail quantiles
        tail_metrics = compute_tail_quantiles(model, loader, loss_config, device)
        all_metrics["t_ret_rmse"].append(tail_metrics["t_ret_rmse"])
        all_metrics["comfort_p90"].append(tail_metrics["comfort_p90"])
        all_metrics["comfort_p95"].append(tail_metrics["comfort_p95"])
        all_metrics["comfort_p99"].append(tail_metrics["comfort_p99"])
        all_metrics["energy_p90"].append(tail_metrics["energy_p90"])
        all_metrics["energy_p95"].append(tail_metrics["energy_p95"])
        all_metrics["energy_p99"].append(tail_metrics["energy_p99"])
    
    # Compute mean and std for each metric
    results = {}
    for key, values in all_metrics.items():
        arr = np.array(values)
        results[f"{key}_mean"] = float(np.mean(arr))
        results[f"{key}_std"] = float(np.std(arr))
    
    return results


def compute_tail_quantiles(
    model: ThermalDynamicsNet,
    loader,
    loss_config: dict,
    device: str,
) -> Dict[str, float]:
    """
    Compute tail quantiles for comfort and energy costs.
    
    This gives additional insight into what the CVaR is capturing.
    """
    model.eval()
    
    all_comfort = []
    all_energy = []
    all_t_ret_errors = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                states, controls, disturbances, targets, Tmin, Tmax = batch
            else:
                states, controls, disturbances, targets = batch[:4]
                Tmin, Tmax = None, None
            
            states = states.to(device)
            controls = controls.to(device)
            disturbances = disturbances.to(device)
            targets = targets.to(device)
            
            # Get predictions
            preds = model(states, controls, disturbances)
            
            # Compute stage costs with components
            _, components = stage_cost_ramc(
                preds, controls,
                Tmin=Tmin.to(device) if Tmin is not None else None,
                Tmax=Tmax.to(device) if Tmax is not None else None,
                comfort_bounds=loss_config.get("comfort_bounds", (20.0, 22.0)),
                comfort_hinge=loss_config.get("comfort_hinge", "softplus"),
                comfort_beta=loss_config.get("comfort_beta", 0.5),
                t_ret_index=loss_config.get("t_ret_index", 5),
                dt_minutes=loss_config.get("dt_minutes", 10.0),
                energy_cost_rate=loss_config.get("energy_cost_rate", 0.9),
                w_comfort=63,  # Unweighted for analysis
                w_energy=1.0,
                cost_scale=1.0,
                return_components=True,
            )
            
            all_comfort.append(components["comfort"].cpu())
            all_energy.append(components["energy_cost"].cpu())
            
            # T_ret errors
            t_ret_idx = loss_config.get("t_ret_index", 5)
            t_ret_errors = (preds[:, t_ret_idx] - targets[:, t_ret_idx]).abs().cpu()
            all_t_ret_errors.append(t_ret_errors)
    
    all_comfort = torch.cat(all_comfort, dim=0).numpy()
    all_energy = torch.cat(all_energy, dim=0).numpy()
    all_t_ret_errors = torch.cat(all_t_ret_errors, dim=0).numpy()
    
    return {
        "t_ret_rmse": float(np.sqrt(np.mean(all_t_ret_errors ** 2))),
        "comfort_p90": float(np.percentile(all_comfort, 90)),
        "comfort_p95": float(np.percentile(all_comfort, 95)),
        "comfort_p99": float(np.percentile(all_comfort, 99)),
        "energy_p90": float(np.percentile(all_energy, 90)),
        "energy_p95": float(np.percentile(all_energy, 95)),
        "energy_p99": float(np.percentile(all_energy, 99)),
    }


def extract_lambda_from_name(model_name: str) -> float:
    """Extract lambda value from model name."""
    if "MSE" in model_name and "RAMC" not in model_name:
        return -1.0  # Special marker for MSE baseline
    elif "Fidelity" in model_name or "lambda_0.0_" in model_name:
        return 0.0
    elif "lambda_" in model_name:
        # Parse lambda value from name like "RAMC_lambda_0.001_op_cvar"
        parts = model_name.split("lambda_")
        if len(parts) > 1:
            value_part = parts[1].split("_")[0]
            try:
                return float(value_part)
            except ValueError:
                pass
    return float("nan")


def format_with_uncertainty(mean: float, std: float, precision: int = 4) -> str:
    """Format a value with its uncertainty."""
    if std < 1e-8:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def generate_decomposed_evaluation_table(
    results_dir: str,
    csv_data_path: str,
    model_config: dict,
    eval_loss_config: dict,
    data_config: dict,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    num_seeds: int = 3,
    enable_tail_diagnostics: bool = False,
    tail_file_tag: str = "DIAGNOSTIC",
) -> pd.DataFrame:
    """
    Generate the decomposed evaluation table with tail diagnostics.
    
    Args:
        results_dir: Directory containing trained model checkpoints
        csv_data_path: Path to data CSV file
        model_config: Model architecture configuration
        eval_loss_config: Evaluation loss configuration
        data_config: Data loading configuration
        device: Computation device
        output_dir: Directory to save outputs (defaults to results_dir)
        num_seeds: Number of seeds for uncertainty estimation
        enable_tail_diagnostics: Whether to generate tail diagnostic plots
        
    Returns:
        DataFrame with all metrics
    """
    output_dir = output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 90)
    print("RAMC DECOMPOSED EVALUATION TABLE GENERATOR")
    print("=" * 90)
    print(f"Results directory: {results_dir}")
    print(f"Data file: {csv_data_path}")
    print(f"Device: {device}")
    print(f"Evaluation K: {eval_loss_config['num_perturbations']}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Tail diagnostics: {'ENABLED' if enable_tail_diagnostics else 'DISABLED'}")
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    train_loader, val_loader, test_loader, full_dataset, split_indices = create_dataloaders(
        csv_file_path=csv_data_path,
        batch_size=data_config["batch_size"],
        train_split=data_config["train_split"],
        val_split=data_config["val_split"],
        test_split=data_config["test_split"],
        device=device,
        split_mode='time',
    )
    print(f"  Test set size: {len(split_indices['test'])} samples")
    
    # Build clamp bounds from TRAIN data only (consistent with training)
    print("\nFIX v2: Building clamp bounds from TRAINING data only...")
    clamp_bounds = build_clamp_bounds_from_dataset(
        full_dataset,
        qlo=0.001,
        qhi=0.999,
        indices=split_indices["train"],
    )
    
    # Override disturbances with the physical/data-derived bounds
    print("Using data-derived disturbance bounds (consistent with training):")
    clamp_bounds["dist"] = {
        "T_out": (-30.0, 40.0),
        "Q_solar": (0.0, 40000.0),
        "Q_internal": (0.0, 100000.0),
    }
    for k, v in clamp_bounds["dist"].items():
        print(f"  {k}: {v}")
    
    # Inject into eval loss config
    eval_loss_config = dict(eval_loss_config)
    eval_loss_config["clamp_bounds"] = clamp_bounds
    print("  Clamp bounds injected into eval_loss_config")
    print()
    
    # Step 2: Find model checkpoints
    print("Step 2: Finding model checkpoints...")
    checkpoints = find_model_checkpoints(results_dir)
    print(f"  Found {len(checkpoints)} models:")
    for name in sorted(checkpoints.keys()):
        print(f"    - {name}")
    print()
    
    if len(checkpoints) == 0:
        raise ValueError(f"No model checkpoints found in {results_dir}")
    
    # Step 3: Evaluate each model
    print("Step 3: Evaluating models on TEST set...")
    print("-" * 90)
    
    all_results = []
    
    for model_name in sorted(checkpoints.keys(), key=lambda x: extract_lambda_from_name(x)):
        checkpoint_path = checkpoints[model_name]
        lambda_val = extract_lambda_from_name(model_name)
        
        print(f"\nEvaluating: {model_name} (λ={lambda_val})")
        
        # Load model
        model = load_model(checkpoint_path, model_config, device)
        
        # Compute detailed metrics
        metrics = compute_detailed_metrics(
            model, test_loader, eval_loss_config, device, num_seeds=num_seeds
        )
        
        # Add model info
        metrics["model_name"] = model_name
        metrics["lambda"] = lambda_val
        
        all_results.append(metrics)
        
        # Print key decomposed metrics
        print(f"  Risk (total):   {metrics['risk_total_mean']:.5f} ± {metrics['risk_total_std']:.5f}")
        print(f"  Risk (comfort): {metrics['risk_comfort_mean']:.5f} ± {metrics['risk_comfort_std']:.5f}")
        print(f"  Risk (energy):  {metrics['risk_energy_mean']:.5f} ± {metrics['risk_energy_std']:.5f}")
        print(f"  T_air RMSE:     {metrics['t_air_rmse_mean']:.4f} ± {metrics['t_air_rmse_std']:.4f}")
    
    # Step 4: Create DataFrame
    print("\n" + "=" * 90)
    print("Step 4: Creating results table...")
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    column_order = [
        "model_name", "lambda",
        # Fidelity
        "mse_raw_mean", "mse_raw_std",
        "fidelity_loss_mean", "fidelity_loss_std",
        "t_air_rmse_mean", "t_air_rmse_std",
        "t_ret_rmse_mean", "t_ret_rmse_std",
        # Total risk
        "risk_total_mean", "risk_total_std",
        # Decomposed risk (KEY)
        "risk_comfort_mean", "risk_comfort_std",
        "risk_energy_mean", "risk_energy_std",
        # Expected costs
        "expected_cost_mean", "expected_cost_std",
        "expected_comfort_mean", "expected_comfort_std",
        "expected_energy_mean", "expected_energy_std",
        # Energy proxy
        "Q_rmse_mean", "Q_rmse_std",
        "Q_mae_mean", "Q_mae_std",
        "Q_bias_mean", "Q_bias_std",
        # Tail quantiles
        "comfort_p90_mean", "comfort_p95_mean", "comfort_p99_mean",
        "energy_p90_mean", "energy_p95_mean", "energy_p99_mean",
    ]
    
    # Only include columns that exist
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]
    
    # Step 5: Save outputs
    print("\nStep 5: Saving outputs...")
    
    # Save full CSV
    csv_path = os.path.join(output_dir, "decomposed_evaluation_full.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Full CSV: {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, "decomposed_evaluation_full.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"  JSON: {json_path}")
    
    # Step 6: Generate tail diagnostics (3-4 model comparison)
    if enable_tail_diagnostics:
        print("\n" + "=" * 90)
        print("Step 6: TAIL DIAGNOSTICS (3-4 Model Comparison)")
        print("=" * 90)
        
        # Define models to compare (in order of expected tail performance)
        # NEW (adjust to match whatever λ values you end up training):
        # - λ=3e-4: "no-regret" model (lowest fidelity loss + best T_ret rollout)  
        # - λ=5e-3: strongest one-step risk reduction
        models_to_compare = [
            ("Raw_MSE_Baseline", "Raw MSE"),
            ("Fidelity_Baseline", "Fidelity (λ=0)"),
            ("RAMC_lambda_0.0003_op_cvar", "RAMC (λ=3×10⁻⁴)"),
            ("RAMC_lambda_0.005_op_cvar",  "RAMC (λ=5×10⁻³)"),
        ]
        
        # Filter to only models that exist in checkpoints
        available_models = []
        for ckpt_name, display_name in models_to_compare:
            if ckpt_name in checkpoints:
                available_models.append((ckpt_name, display_name))
            else:
                print(f"  [SKIP] Checkpoint not found: {ckpt_name}")
        
        if len(available_models) < 2:
            print(f"\n  Cannot run tail diagnostics: need at least 2 models, found {len(available_models)}")
        else:
            print(f"\nComparing {len(available_models)} models:")
            for ckpt_name, display_name in available_models:
                print(f"  - {display_name} ({ckpt_name})")
            
            # Build tail config
            tail_cfg = dict(eval_loss_config)
            tail_cfg["num_perturbations"] = 256  # stable tail estimate
            alpha = float(tail_cfg["cvar_alpha"])
            
            print(f"\nCollecting cost samples (K={tail_cfg['num_perturbations']}, 10 batches)...")
            
            # Collect cost samples for all models
            costs_dict = {}
            tail_stats_all = {}
            
            for ckpt_name, display_name in available_models:
                print(f"  Processing: {display_name}...")
                model = load_model(checkpoints[ckpt_name], model_config, device, verbose=False)
                
                costs = collect_cost_samples(
                    model, test_loader, tail_cfg, device, max_batches=10, seed=123
                )
                costs_dict[display_name] = costs
                
                # Compute tail statistics
                mean_c, var_c, cvar_c = summarize_tail(costs, alpha=alpha)
                tail_stats_all[display_name] = {
                    "checkpoint": ckpt_name,
                    "lambda": extract_lambda_from_name(ckpt_name),
                    "mean": mean_c,
                    "var": var_c,
                    "cvar": cvar_c,
                }
                print(f"    Mean={mean_c:.4f}  VaR₀.₉={var_c:.4f}  CVaR₀.₉={cvar_c:.4f}")
            
            # Print summary table
            print(f"\n" + "-" * 70)
            print(f"TAIL SUMMARY (α={alpha}, K={tail_cfg['num_perturbations']})")
            print("-" * 70)
            print(f"{'Model':<25} {'Mean':>10} {'VaR₀.₉':>10} {'CVaR₀.₉':>10}")
            print("-" * 70)
            for name, stats in tail_stats_all.items():
                print(f"{name:<25} {stats['mean']:>10.4f} {stats['var']:>10.4f} {stats['cvar']:>10.4f}")
            print("-" * 70)
            
            # Compute improvements vs Fidelity baseline
            if "Fidelity (λ=0)" in tail_stats_all:
                base_cvar = tail_stats_all["Fidelity (λ=0)"]["cvar"]
                base_var = tail_stats_all["Fidelity (λ=0)"]["var"]
                base_mean = tail_stats_all["Fidelity (λ=0)"]["mean"]
                
                print(f"\nIMPROVEMENTS vs Fidelity (λ=0) Baseline:")
                print("-" * 70)
                print(f"{'Model':<25} {'ΔMean':>12} {'ΔVaR₀.₉':>12} {'ΔCVaR₀.₉':>12}")
                print("-" * 70)
                
                for name, stats in tail_stats_all.items():
                    if name != "Fidelity (λ=0)":
                        mean_reduction = 100 * (base_mean - stats["mean"]) / base_mean
                        var_reduction = 100 * (base_var - stats["var"]) / base_var
                        cvar_reduction = 100 * (base_cvar - stats["cvar"]) / base_cvar
                        
                        print(f"{name:<25} {mean_reduction:>+11.1f}% {var_reduction:>+11.1f}% {cvar_reduction:>+11.1f}%")
                print("-" * 70)
            
            # Generate ECDF plot (main figure for paper)
            ecdf_path = os.path.join(output_dir, f"tail_ecdf_comparison_TEST_{tail_file_tag}.png")
            plot_cost_ecdf_comparison(
                costs_dict,
                alpha=alpha,
                save_path=ecdf_path,
                show=False,
                figsize=(3.5, 2.4),  # IEEE single-column
            )
            print(f"\n  ECDF plot saved: {ecdf_path}")
            print(f"    -> This is a DIAGNOSTIC tail figure (not the canonical paper ECDF)")
            
            # Generate histogram + ECDF combo (supplementary)
            combo_path = os.path.join(output_dir, f"tail_distribution_comparison_TEST_{tail_file_tag}.png")
            plot_cost_tail_comparison(
                costs_dict,
                alpha=alpha,
                save_path=combo_path,
            )
            print(f"  Combo plot saved: {combo_path}")
            print(f"    -> DIAGNOSTIC combo plot (supplementary if needed)")
            
            # Save tail statistics to JSON
            tail_json_path = os.path.join(output_dir, f"tail_statistics_comparison_{tail_file_tag}.json")
            tail_output = {
                "alpha": alpha,
                "num_perturbations": tail_cfg["num_perturbations"],
                "models": tail_stats_all,
            }
            
            # Add improvements if baseline exists
            if "Fidelity (λ=0)" in tail_stats_all:
                tail_output["improvements_vs_baseline"] = {}
                for name, stats in tail_stats_all.items():
                    if name != "Fidelity (λ=0)":
                        tail_output["improvements_vs_baseline"][name] = {
                            "mean_reduction_pct": 100 * (base_mean - stats["mean"]) / base_mean,
                            "var_reduction_pct": 100 * (base_var - stats["var"]) / base_var,
                            "cvar_reduction_pct": 100 * (base_cvar - stats["cvar"]) / base_cvar,
                        }
            
            with open(tail_json_path, "w") as f:
                json.dump(tail_output, f, indent=2)
            print(f"  Tail stats saved: {tail_json_path}")
    
    # Step 7: Print summary tables
    print("\n" + "=" * 90)
    print("DECOMPOSED EVALUATION RESULTS - TEST SET")
    print("=" * 90)
    
    print_summary_table(df)
    print_decision_guidance(df)
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, "decomposed_evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(generate_text_summary(df, eval_loss_config))
    print(f"\n  Summary saved to: {summary_path}")
    
    return df


def print_summary_table(df: pd.DataFrame):
    """Print a nicely formatted summary table."""
    
    print("\n" + "-" * 110)
    print("KEY METRICS COMPARISON")
    print("-" * 110)
    
    # Header
    print(f"{'Model':<28} {'λ':>8} {'Fidelity':>12} {'Risk(tot)':>12} "
          f"{'Risk(comf)':>12} {'Risk(ener)':>12} {'T_air RMSE':>12}")
    print("-" * 110)
    
    for _, row in df.iterrows():
        model_name = row["model_name"]
        if len(model_name) > 26:
            model_name = model_name[:24] + ".."
        
        lambda_str = f"{row['lambda']:.4f}" if row['lambda'] >= 0 else "N/A"
        
        print(f"{model_name:<28} {lambda_str:>8} "
              f"{row['fidelity_loss_mean']:>12.5f} "
              f"{row['risk_total_mean']:>12.5f} "
              f"{row['risk_comfort_mean']:>12.5f} "
              f"{row['risk_energy_mean']:>12.5f} "
              f"{row['t_air_rmse_mean']:>12.4f}")
    
    print("-" * 110)
    
    # Compute relative changes vs baseline
    print("\n" + "-" * 110)
    print("RELATIVE CHANGES VS FIDELITY BASELINE")
    print("-" * 110)
    
    # Find baseline
    baseline_row = df[df["lambda"] == 0.0]
    if len(baseline_row) == 0:
        baseline_row = df[df["model_name"].str.contains("Fidelity", case=False)]
    
    if len(baseline_row) > 0:
        baseline = baseline_row.iloc[0]
        
        print(f"{'Model':<28} {'λ':>8} {'ΔFidelity':>12} {'ΔRisk(tot)':>12} "
              f"{'ΔRisk(comf)':>12} {'ΔRisk(ener)':>12} {'ΔT_air RMSE':>12}")
        print("-" * 110)
        
        for _, row in df.iterrows():
            if row["lambda"] == 0.0 or row["lambda"] < 0:
                continue
            
            model_name = row["model_name"]
            if len(model_name) > 26:
                model_name = model_name[:24] + ".."
            
            # Compute relative changes (%)
            def rel_change(val, base):
                if abs(base) < 1e-10:
                    return float('nan')
                return 100 * (val - base) / abs(base)
            
            d_fid = rel_change(row['fidelity_loss_mean'], baseline['fidelity_loss_mean'])
            d_risk = rel_change(row['risk_total_mean'], baseline['risk_total_mean'])
            d_risk_c = rel_change(row['risk_comfort_mean'], baseline['risk_comfort_mean'])
            d_risk_e = rel_change(row['risk_energy_mean'], baseline['risk_energy_mean'])
            d_rmse = rel_change(row['t_air_rmse_mean'], baseline['t_air_rmse_mean'])
            
            print(f"{model_name:<28} {row['lambda']:>8.4f} "
                  f"{d_fid:>+11.1f}% "
                  f"{d_risk:>+11.1f}% "
                  f"{d_risk_c:>+11.1f}% "
                  f"{d_risk_e:>+11.1f}% "
                  f"{d_rmse:>+11.1f}%")
        
        print("-" * 110)


def print_decision_guidance(df: pd.DataFrame):
    """Print guidance on which story the results support."""
    
    print("\n" + "=" * 90)
    print("DECISION GUIDANCE: WHICH STORY DO YOUR RESULTS SUPPORT?")
    print("=" * 90)
    
    # Find baseline and best RAMC model
    baseline_row = df[df["lambda"] == 0.0]
    if len(baseline_row) == 0:
        baseline_row = df[df["model_name"].str.contains("Fidelity", case=False)]
    
    ramc_rows = df[(df["lambda"] > 0) & (df["lambda"] < 1)]
    
    if len(baseline_row) == 0 or len(ramc_rows) == 0:
        print("  Cannot compute guidance: missing baseline or RAMC models")
        return
    
    baseline = baseline_row.iloc[0]
    
    # Find best RAMC by risk reduction while keeping fidelity reasonable
    best_ramc = None
    best_score = float('-inf')
    
    for _, row in ramc_rows.iterrows():
        # Score: risk reduction minus fidelity increase (both in %)
        risk_reduction = (baseline['risk_total_mean'] - row['risk_total_mean']) / baseline['risk_total_mean']
        fid_increase = (row['fidelity_loss_mean'] - baseline['fidelity_loss_mean']) / baseline['fidelity_loss_mean']
        
        score = risk_reduction - 0.5 * fid_increase  # Penalize fidelity degradation
        
        if score > best_score:
            best_score = score
            best_ramc = row
    
    if best_ramc is None:
        print("  No suitable RAMC model found")
        return
    
    # Compute percentage changes correctly
    comfort_change = (best_ramc['risk_comfort_mean'] - baseline['risk_comfort_mean']) / baseline['risk_comfort_mean'] * 100
    energy_change = (best_ramc['risk_energy_mean'] - baseline['risk_energy_mean']) / baseline['risk_energy_mean'] * 100
    total_change = (best_ramc['risk_total_mean'] - baseline['risk_total_mean']) / baseline['risk_total_mean'] * 100
    fidelity_change = (best_ramc['fidelity_loss_mean'] - baseline['fidelity_loss_mean']) / baseline['fidelity_loss_mean'] * 100
    
    print(f"\nBest RAMC model: {best_ramc['model_name']} (λ={best_ramc['lambda']})")
    print(f"\nChanges vs Fidelity Baseline:")
    print(f"  Total risk:   {total_change:+.1f}%  (negative = improvement)")
    print(f"  Comfort risk: {comfort_change:+.1f}%  (negative = improvement)")
    print(f"  Energy risk:  {energy_change:+.1f}%  (negative = improvement)")
    print(f"  Fidelity cost: {fidelity_change:+.1f}%  (positive = worse)")
    
    # For decision logic, use absolute values to compare magnitudes
    comfort_improvement_magnitude = abs(comfort_change) if comfort_change < 0 else 0
    energy_improvement_magnitude = abs(energy_change) if energy_change < 0 else 0
    
    # Decision logic
    print("\n" + "-" * 70)
    print("RECOMMENDATION:")
    print("-" * 70)
    
    if comfort_improvement_magnitude > energy_improvement_magnitude * 1.5:
        print("""
  -> Your results show STRONGER COMFORT RISK REDUCTION than energy risk.
  
  RECOMMENDED STORY: "Comfort Safety" (Option 2)
  
  Key claim: "RAMC reduces tail discomfort risk by {:.1f}% with
             {:.1f}% increase in prediction error."
  
  Note: Your mixed comfort+energy CVaR is defensible because decomposition
        shows the improvement is comfort-dominated.
""".format(comfort_improvement_magnitude, fidelity_change))
    
    elif energy_improvement_magnitude > comfort_improvement_magnitude * 1.5:
        print("""
  -> Your results show STRONGER ENERGY RISK REDUCTION than comfort risk.
  
  RECOMMENDED STORY: "Economic Risk Reduction" (Option 1)
  
  Key claim: "RAMC reduces tail energy cost variability by {:.1f}% with
             {:.1f}% increase in prediction error."
  
  Consider: Monetizing comfort ($/°C·h) for consistent units and stronger
            "operational cost" framing.
""".format(energy_improvement_magnitude, fidelity_change))
    
    else:
        total_improvement_magnitude = abs(total_change) if total_change < 0 else 0
        print("""
  -> Your results show BALANCED improvement in both comfort and energy risk.
  
  RECOMMENDED STORY: "Composite Operational Risk" (Option 1 or 3)
  
  Key claim: "RAMC reduces tail operational risk by {:.1f}% across both
             comfort and energy dimensions with {:.1f}% fidelity cost."
  
  For Option 1: Monetize comfort for consistent units.
  For Option 3: Report decomposed metrics (as you're doing now).
""".format(total_improvement_magnitude, fidelity_change))
    
    # Check for "no-regret" regime
    no_regret_models = ramc_rows[
        (ramc_rows['risk_total_mean'] < baseline['risk_total_mean']) &
        (ramc_rows['fidelity_loss_mean'] < baseline['fidelity_loss_mean'] * 1.02)
    ]
    
    if len(no_regret_models) > 0:
        print(f"""
  GOOD NEWS: You have {len(no_regret_models)} model(s) in the "NO-REGRET" regime
    where risk decreases with negligible (<2%) fidelity cost.
    
    This is a strong selling point for RAMC!
""")
    
    # Check for stability issues at high lambda
    high_lambda = ramc_rows[ramc_rows['lambda'] > ramc_rows['lambda'].median() * 2]
    if len(high_lambda) > 0:
        worst = high_lambda.loc[high_lambda['fidelity_loss_mean'].idxmax()]
        if worst['fidelity_loss_mean'] > baseline['fidelity_loss_mean'] * 1.5:
            print(f"""
  NOTE: High-λ models (e.g., λ={worst['lambda']}) show significant fidelity
    degradation ({100*(worst['fidelity_loss_mean']/baseline['fidelity_loss_mean']-1):.0f}%), 
    confirming the need for λ tuning.
""")


def generate_text_summary(df: pd.DataFrame, eval_config: dict) -> str:
    """Generate a complete text summary for saving."""
    
    lines = []
    lines.append("=" * 90)
    lines.append("RAMC DECOMPOSED EVALUATION SUMMARY")
    lines.append("=" * 90)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Risk operator: {eval_config.get('risk_operator', 'N/A')}")
    lines.append(f"CVaR alpha: {eval_config.get('cvar_alpha', 'N/A')}")
    lines.append(f"Evaluation K: {eval_config.get('num_perturbations', 'N/A')}")
    lines.append("")
    
    # Full table
    lines.append("-" * 90)
    lines.append("FULL METRICS TABLE")
    lines.append("-" * 90)
    lines.append(df.to_string(index=False))
    lines.append("")
    
    # Key findings
    lines.append("-" * 90)
    lines.append("KEY FINDINGS")
    lines.append("-" * 90)
    
    baseline = df[df["lambda"] == 0.0]
    if len(baseline) == 0:
        baseline = df[df["model_name"].str.contains("Fidelity", case=False)]
    
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        
        ramc = df[(df["lambda"] > 0) & (df["lambda"] < 1)]
        if len(ramc) > 0:
            best_risk = ramc.loc[ramc['risk_total_mean'].idxmin()]
            
            lines.append(f"Baseline (λ=0):")
            lines.append(f"  Risk (total): {baseline['risk_total_mean']:.5f}")
            lines.append(f"  Risk (comfort): {baseline['risk_comfort_mean']:.5f}")
            lines.append(f"  Risk (energy): {baseline['risk_energy_mean']:.5f}")
            lines.append(f"  Fidelity: {baseline['fidelity_loss_mean']:.5f}")
            lines.append("")
            lines.append(f"Best risk model (λ={best_risk['lambda']}):")
            lines.append(f"  Risk (total): {best_risk['risk_total_mean']:.5f} "
                        f"({100*(best_risk['risk_total_mean']/baseline['risk_total_mean']-1):+.1f}%)")
            lines.append(f"  Risk (comfort): {best_risk['risk_comfort_mean']:.5f} "
                        f"({100*(best_risk['risk_comfort_mean']/baseline['risk_comfort_mean']-1):+.1f}%)")
            lines.append(f"  Risk (energy): {best_risk['risk_energy_mean']:.5f} "
                        f"({100*(best_risk['risk_energy_mean']/baseline['risk_energy_mean']-1):+.1f}%)")
            lines.append(f"  Fidelity: {best_risk['fidelity_loss_mean']:.5f} "
                        f"({100*(best_risk['fidelity_loss_mean']/baseline['fidelity_loss_mean']-1):+.1f}%)")
    
    return "\n".join(lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        print("\nPlease update RESULTS_DIR at the top of this script to point to")
        print("the trained model checkpoints directory.")
        print("\nExample: RESULTS_DIR = 'results/RAMC_FULL_cvar_20241231_123456'")
        exit(1)
    
    # Check if data file exists
    if not os.path.exists(CSV_DATA_PATH):
        print(f"ERROR: Data file not found: {CSV_DATA_PATH}")
        print("\nPlease update CSV_DATA_PATH at the top of this script.")
        exit(1)
    
    # Run evaluation
    df = generate_decomposed_evaluation_table(
        results_dir=RESULTS_DIR,
        csv_data_path=CSV_DATA_PATH,
        model_config=MODEL_CONFIG,
        eval_loss_config=EVAL_LOSS_CONFIG,
        data_config=DATA_CONFIG,
        device=DEVICE,
        num_seeds=3,  # Use 3 seeds for faster evaluation, increase to 5 for final results
        enable_tail_diagnostics=False,   # Disabled: canonical ECDF comes from pooled_ecdf_analysis.py
        tail_file_tag="DIAGNOSTIC",
    )
    
    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)
    print(f"\nOutput files saved to: {RESULTS_DIR}")
    print("  - decomposed_evaluation_full.csv")
    print("  - decomposed_evaluation_full.json")
    print("  - decomposed_evaluation_summary.txt")
    print("  - tail_ecdf_comparison_TEST_DIAGNOSTIC.png  [DIAGNOSTIC only, if enabled]")
    print("  - tail_distribution_comparison_TEST_DIAGNOSTIC.png  [DIAGNOSTIC only, if enabled]")
    print("  - tail_statistics_comparison_DIAGNOSTIC.json  [DIAGNOSTIC only, if enabled]")
    print("  NOTE: Canonical ECDF figure comes from pooled_ecdf_analysis.py")
    print("\nNext steps:")
    print("  1. Review the DECISION GUIDANCE above")
    print("  2. Run pooled_ecdf_analysis.py for the canonical ECDF figure")
    print("  3. Use the pooled ECDF (shows Raw MSE -> Fidelity -> RAMC progression)")
    print("  4. Remove Table 8 from the paper as recommended")