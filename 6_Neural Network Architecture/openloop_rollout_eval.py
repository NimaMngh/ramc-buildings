# -*- coding: utf-8 -*-
"""
openloop_rollout_eval.py - Free-run rollout evaluation for RAMC
================================================================
Evaluates model prediction error growth over H-step free-run rollouts.

This is different from openloop_rollout_risk_eval.py which evaluates
CVaR of cumulative costs under forecast uncertainty.

Key features:
- H-step free-run rollouts (model predictions fed back as inputs)
- RMSE tracking per step (error growth analysis)
- Comparison across multiple models
- Test-set only evaluation with episode-aware start selection

Author: RAMC Framework
Date: 2026-01-05
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from thermal_dynamics_net import ThermalDynamicsNet, RCModelDataset


# =============================================================================
# SINGLE MODEL ROLLOUT EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_rollouts(
    model: ThermalDynamicsNet,
    dataset: RCModelDataset,
    horizon: int = 24,
    num_rollouts: int = 100,
    device: str = "cpu",
    seed: int = 42,
    min_start_idx: int = 0,
    max_start_idx: Optional[int] = None,
    state_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate free-run rollout performance for a single model.
    
    Free-run rollout: Start from ground-truth state, then feed model predictions
    back as inputs for subsequent steps. Compare against ground-truth trajectory.
    
    Args:
        model: Trained thermal dynamics model
        dataset: Dataset with states, controls, disturbances, targets
        horizon: Number of steps to roll out (H)
        num_rollouts: Number of rollout trajectories to evaluate
        device: Computation device
        seed: Random seed for reproducible start selection
        min_start_idx: Minimum valid start index (e.g., test set start)
        max_start_idx: Maximum valid start index (e.g., test set end - horizon)
        state_names: Names for each state dimension (for reporting)
        
    Returns:
        Dict with:
        - rmse_per_step: [H] RMSE at each step (averaged over rollouts)
        - rmse_per_state_per_step: [n_states, H] per-state RMSE
        - T_air_rmse_final: Final step T_air RMSE
        - T_ret_rmse_final: Final step T_ret RMSE  
        - T_air_growth: Ratio of final to first step RMSE (error growth)
        - selected_starts: List of start indices used
    """
    model.eval()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    H = int(horizon)
    n_states = dataset.states.shape[1]
    
    if state_names is None:
        state_names = ["T_air", "T_env", "T_int", "T_rad1", "T_rad2", "T_ret"]
    
    # Get valid rollout starts (respecting episode boundaries)
    valid_starts = dataset.valid_rollout_starts(H)
    
    if max_start_idx is None:
        max_start_idx = len(dataset) - H
    
    # Filter to requested range
    valid_starts = valid_starts[
        (valid_starts >= min_start_idx) & (valid_starts <= max_start_idx)
    ]
    
    if len(valid_starts) == 0:
        raise ValueError(f"No valid rollout starts in range [{min_start_idx}, {max_start_idx}]")
    
    num_rollouts = min(num_rollouts, len(valid_starts))
    selected_starts = np.random.choice(valid_starts, size=num_rollouts, replace=False)
    
    print(f"  Running {num_rollouts} rollouts of horizon {H} steps...")
    
    # Accumulators: [num_rollouts, H, n_states]
    all_errors_sq = np.zeros((num_rollouts, H, n_states))
    
    for r_idx, start_idx in enumerate(selected_starts):
        # Initialize with ground-truth state
        x = dataset.states[start_idx].unsqueeze(0).to(device)  # [1, n_states]
        
        for h in range(H):
            idx = start_idx + h
            
            # Get control and disturbance for this step
            u = dataset.controls[idx].unsqueeze(0).to(device)  # [1, n_controls]
            d = dataset.disturbances[idx].unsqueeze(0).to(device)  # [1, n_dist]
            
            # Predict next state
            x_next_pred = model(x, u, d)  # [1, n_states]
            
            # Ground truth next state
            x_next_true = dataset.targets[idx].unsqueeze(0).to(device)  # [1, n_states]
            
            # Compute squared error
            error_sq = (x_next_pred - x_next_true).pow(2).squeeze(0).cpu().numpy()
            all_errors_sq[r_idx, h, :] = error_sq
            
            # Free-run: use prediction as next state
            x = x_next_pred
    
    # Compute RMSE statistics
    # Mean over rollouts, then sqrt
    mse_per_step = np.mean(all_errors_sq, axis=0)  # [H, n_states]
    rmse_per_state_per_step = np.sqrt(mse_per_step).T  # [n_states, H]
    
    # Total RMSE per step (mean over states)
    rmse_per_step = np.sqrt(np.mean(mse_per_step, axis=1))  # [H]
    
    # Extract key metrics
    T_air_rmse = rmse_per_state_per_step[0, :]  # T_air is index 0
    T_ret_rmse = rmse_per_state_per_step[5, :]  # T_ret is index 5
    
    T_air_rmse_final = float(T_air_rmse[-1])
    T_ret_rmse_final = float(T_ret_rmse[-1])
    
    # Error growth ratio (final / first, avoid division by zero)
    T_air_growth = float(T_air_rmse[-1] / (T_air_rmse[0] + 1e-8))
    T_ret_growth = float(T_ret_rmse[-1] / (T_ret_rmse[0] + 1e-8))
    
    result = {
        "horizon": H,
        "num_rollouts": num_rollouts,
        "rmse_per_step": rmse_per_step.tolist(),
        "rmse_per_state_per_step": {
            name: rmse_per_state_per_step[i, :].tolist() 
            for i, name in enumerate(state_names)
        },
        "T_air_rmse_per_step": T_air_rmse.tolist(),
        "T_ret_rmse_per_step": T_ret_rmse.tolist(),
        "T_air_rmse_final": T_air_rmse_final,
        "T_ret_rmse_final": T_ret_rmse_final,
        "T_air_growth": T_air_growth,
        "T_ret_growth": T_ret_growth,
        "selected_starts": selected_starts.tolist(),
        "min_start_idx": int(min_start_idx),
        "max_start_idx": int(max_start_idx),
    }
    
    return result


# =============================================================================
# MULTI-MODEL COMPARISON
# =============================================================================

def compare_rollouts_multiple_models(
    models: Dict[str, ThermalDynamicsNet],
    dataset: RCModelDataset,
    horizon: int = 24,
    num_rollouts: int = 100,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    seed: int = 42,
    min_start_idx: int = 0,
    max_start_idx: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare free-run rollout performance across multiple models.
    
    Args:
        models: Dict mapping model names to ThermalDynamicsNet instances
        dataset: Dataset for evaluation
        horizon: Rollout horizon H
        num_rollouts: Number of rollouts per model
        device: Computation device
        save_dir: Directory to save results and plots
        seed: Random seed (same starts used for all models)
        min_start_idx, max_start_idx: Index bounds (e.g., test set only)
        
    Returns:
        Dict mapping model names to their rollout results
    """
    all_results = {}
    
    print(f"\nComparing rollout performance across {len(models)} models")
    print(f"  Horizon: {horizon} steps ({horizon * 10} minutes)")
    print(f"  Index range: [{min_start_idx}, {max_start_idx}]")
    print("-" * 60)
    
    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")
        
        result = evaluate_rollouts(
            model=model,
            dataset=dataset,
            horizon=horizon,
            num_rollouts=num_rollouts,
            device=device,
            seed=seed,  # Same seed = same start indices
            min_start_idx=min_start_idx,
            max_start_idx=max_start_idx,
        )
        
        all_results[model_name] = result
        
        print(f"    T_air RMSE (final): {result['T_air_rmse_final']:.4f} °C")
        print(f"    T_ret RMSE (final): {result['T_ret_rmse_final']:.4f} °C")
        print(f"    T_air growth ratio: {result['T_air_growth']:.2f}x")
    
    # Save and plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save JSON summary
        json_path = os.path.join(save_dir, "rollout_comparison.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Generate comparison plots
        plot_rollout_rmse(
            all_results,
            save_path=os.path.join(save_dir, f"rollout_rmse_H{horizon}_comparison.png"),
            title=f"Free-Run Rollout RMSE (H={horizon})"
        )
    
    return all_results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_rollout_rmse(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Free-Run Rollout RMSE",
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot rollout RMSE comparison across models.
    
    Creates a 2x2 figure:
    - (a) T_air RMSE over horizon
    - (b) T_ret RMSE over horizon
    - (c) Final step RMSE bar chart
    - (d) Error growth ratio bar chart
    
    Args:
        results: Dict from compare_rollouts_multiple_models()
        save_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    model_names = list(results.keys())
    if len(model_names) == 0:
        print("No results to plot.")
        return
    
    horizon = results[model_names[0]]["horizon"]
    steps = np.arange(1, horizon + 1)
    hours = steps * (10.0 / 60.0)  # Convert 10-minute timesteps to hours
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # -------------------------------------------------------------------------
    # Plot (a): T_air RMSE over horizon
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    for i, (name, res) in enumerate(results.items()):
        rmse = res["T_air_rmse_per_step"]
        label = _shorten_name(name)
        ax.plot(hours, rmse, 'o-', color=colors[i], label=label, markersize=4, alpha=0.8)
    
    ax.set_xlabel("Rollout time (hours)")
    ax.set_ylabel("T_air RMSE (°C)")
    ax.set_title("(a) Air temperature error growth")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([hours[0], hours[-1]])
    
    # -------------------------------------------------------------------------
    # Plot (b): T_ret RMSE over horizon
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    for i, (name, res) in enumerate(results.items()):
        rmse = res["T_ret_rmse_per_step"]
        label = _shorten_name(name)
        ax.plot(hours, rmse, 's-', color=colors[i], label=label, markersize=4, alpha=0.8)
    
    ax.set_xlabel("Rollout time (hours)")
    ax.set_ylabel("T_ret RMSE (°C)")
    ax.set_title("(b) Return temperature error growth")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([hours[0], hours[-1]])
    
    # -------------------------------------------------------------------------
    # Plot (c): Final step RMSE bar chart
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    T_air_final = [results[name]["T_air_rmse_final"] for name in model_names]
    T_ret_final = [results[name]["T_ret_rmse_final"] for name in model_names]
    
    bars1 = ax.bar(x_pos - width/2, T_air_final, width, label="T_air", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, T_ret_final, width, label="T_ret", color="darkorange", alpha=0.8)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Final step RMSE (°C)")
    ax.set_title(f"(c) RMSE at step {horizon} ({horizon * 10 / 60:.1f} h)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_shorten_name(n) for n in model_names], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    # -------------------------------------------------------------------------
    # Plot (d): Error growth ratio bar chart
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    T_air_growth = [results[name]["T_air_growth"] for name in model_names]
    T_ret_growth = [results[name]["T_ret_growth"] for name in model_names]
    
    bars1 = ax.bar(x_pos - width/2, T_air_growth, width, label="T_air", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, T_ret_growth, width, label="T_ret", color="darkorange", alpha=0.8)
    
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label="No growth")
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Error growth ratio (final/first)")
    ax.set_title("(d) Error growth ratio (final / first step)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    
    plt.show()


def _shorten_name(name: str, max_len: int = 20) -> str:
    """Shorten model name for plotting."""
    if len(name) <= max_len:
        return name
    
    # Try to extract key info
    if "RAMC" in name and "λ=" in name:
        # Extract lambda value
        try:
            lam = name.split("λ=")[1].split()[0]
            return f"RAMC λ={lam}"
        except:
            pass
    elif "RAMC" in name and "lambda=" in name:
        try:
            lam = name.split("lambda=")[1].split()[0]
            return f"RAMC λ={lam}"
        except:
            pass
    
    return name[:max_len-2] + ".."


# =============================================================================
# CONVENIENCE FUNCTION FOR SINGLE PLOT
# =============================================================================

def plot_single_rollout_trajectory(
    model: ThermalDynamicsNet,
    dataset: RCModelDataset,
    start_idx: int,
    horizon: int,
    device: str = "cpu",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot a single rollout trajectory comparing prediction vs ground truth.
    
    Useful for visual debugging and presentations.
    """
    model.eval()
    
    H = int(horizon)
    state_names = ["T_air", "T_env", "T_int", "T_rad1", "T_rad2", "T_ret"]
    
    # Run rollout
    x = dataset.states[start_idx].unsqueeze(0).to(device)
    
    pred_trajectory = [x.squeeze(0).cpu().numpy()]
    true_trajectory = [dataset.states[start_idx].numpy()]
    
    with torch.no_grad():
        for h in range(H):
            idx = start_idx + h
            u = dataset.controls[idx].unsqueeze(0).to(device)
            d = dataset.disturbances[idx].unsqueeze(0).to(device)
            
            x_next_pred = model(x, u, d)
            x_next_true = dataset.targets[idx]
            
            pred_trajectory.append(x_next_pred.squeeze(0).cpu().numpy())
            true_trajectory.append(x_next_true.numpy())
            
            x = x_next_pred
    
    pred_trajectory = np.array(pred_trajectory)
    true_trajectory = np.array(true_trajectory)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = np.arange(H + 1)
    
    for i, (ax, name) in enumerate(zip(axes.flatten(), state_names)):
        ax.plot(steps, true_trajectory[:, i], 'b-', label="Ground Truth", linewidth=2)
        ax.plot(steps, pred_trajectory[:, i], 'r--', label="Prediction", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{name} (°C)")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if title is None:
        title = f"Single Rollout Trajectory (start={start_idx}, H={H})"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Open-loop rollout evaluation module for RAMC")
    print("=" * 60)
    print("\nKey functions:")
    print("  - evaluate_rollouts(): Evaluate single model")
    print("  - compare_rollouts_multiple_models(): Compare multiple models")
    print("  - plot_rollout_rmse(): Plot comparison results")
    print("  - plot_single_rollout_trajectory(): Debug single rollout")
    print("\nThis evaluates FREE-RUN rollout error growth (model stability).")
    print("For CVaR under forecast uncertainty, use openloop_rollout_risk_eval.py")