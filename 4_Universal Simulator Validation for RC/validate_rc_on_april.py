# -*- coding: utf-8 -*-
"""
RC Model Validation Script for v5.1 DE-optimized models
Validates the identified RC model on unseen data (e.g., April week)

Author: Nima Monghasemi
Date: January 2026

This script validates the identified N=3 RC model (ground-truth plant)
on unseen data from the first week of April to assess generalization
from winter training data to shoulder season conditions.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
#                    SELF-CONTAINED RC MODEL CLASS
# =============================================================================

class RCModelEquationsWithSolar:
    """
    RC model with explicit solar aperture parameter.
    Self-contained version for validation without external dependencies.
    
    Model Structure:
    - Building: 3 states (T_air, T_env, T_int)
    - Radiator: N states (T_rad_1, ..., T_rad_N)
    - Parameters: 10 (C_air, C_env, C_int, C_rad, R_ex, R_ae, R_ai, K_rad, a_rad, A_sol)
    - Inputs: 5 (T_outdoor, Q_internal, I_solar, T_supply, mdot)
    """
    
    def __init__(self, N_rad_sections=3):
        self.N = N_rad_sections
        self.dt = 600  # 10-minute timestep [s]
        self.Cp_water = 4186.0  # Water specific heat [J/kg·K]
        
        self.param_names = [
            'C_air', 'C_env', 'C_int', 'C_rad', 
            'R_ex', 'R_ae', 'R_ai', 
            'K_rad', 'a_rad', 'A_sol'
        ]
        
        self.state_names = ['T_air', 'T_env', 'T_int'] + [f'T_rad_{i+1}' for i in range(self.N)]
        self.input_names = ['T_outdoor', 'Q_internal', 'I_solar', 'T_supply', 'mdot']
        
    def get_num_states(self):
        """Return total number of states."""
        return 3 + self.N

    def get_derivatives(self, t, x, u, params):
        """
        Calculate dx/dt for ODE solver.
        
        Parameters:
        -----------
        t : float
            Current time [s]
        x : array (3 + N,)
            State vector [T_air, T_env, T_int, T_rad_1, ..., T_rad_N] [°C]
        u : array (5,)
            Input vector [T_outdoor, Q_internal, I_solar, T_supply, mdot]
        params : dict
            Parameter dictionary
            
        Returns:
        --------
        dxdt : array (3 + N,)
            Derivative of state vector [°C/s]
        """
        # Unpack parameters
        C_air = params['C_air']
        C_env = params['C_env']
        C_int = params['C_int']
        C_rad = params['C_rad']
        R_ex = params['R_ex']
        R_ae = params['R_ae']
        R_ai = params['R_ai']
        K_rad = params['K_rad']
        a_rad = params['a_rad']
        A_sol = params['A_sol']
        
        # Unpack states
        T_air = x[0]
        T_env = x[1]
        T_int = x[2]
        T_rad_sections = x[3:]
        
        # Unpack inputs
        T_outdoor = u[0]
        Q_internal = u[1]
        I_solar = u[2]
        T_supply = u[3]
        mdot = u[4]
        
        # === RADIATOR SUB-MODEL ===
        C_rad_section = C_rad / self.N
        temp_diff = np.abs(T_rad_sections - T_air) + 1e-9
        Q_rad_sections = K_rad * np.power(temp_diff, a_rad + 1)
        Q_rad_total = np.sum(Q_rad_sections)
        
        T_rad_in = np.concatenate(([T_supply], T_rad_sections[:-1]))
        dT_rad_dt = (mdot * self.Cp_water * (T_rad_in - T_rad_sections) - Q_rad_sections) / (C_rad_section + 1e-9)
        
        # === BUILDING THERMAL NETWORK ===
        Q_env_air = (T_env - T_air) / R_ae
        Q_int_air = (T_int - T_air) / R_ai
        Q_out_env = (T_outdoor - T_env) / R_ex
        Q_solar = A_sol * I_solar
        
        dT_air_dt = (Q_env_air + Q_int_air + Q_rad_total + Q_internal + Q_solar) / C_air
        dT_env_dt = (Q_out_env - Q_env_air) / C_env
        dT_int_dt = (-Q_int_air) / C_int
        
        return np.concatenate(([dT_air_dt], [dT_env_dt], [dT_int_dt], dT_rad_dt))


# =============================================================================
#                    SIMULATION FUNCTION
# =============================================================================

def simulate_rc_model(params_dict, df, N_sections):
    """
    Run RC model simulation with given parameters.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of identified parameters
    df : pd.DataFrame
        Data with required columns
    N_sections : int
        Number of radiator sections
        
    Returns:
    --------
    T_air_predicted : np.array
        Predicted indoor air temperature
    all_states : np.array or None
        All state trajectories if successful, None otherwise
    """
    model = RCModelEquationsWithSolar(N_rad_sections=N_sections)
    
    t_data = np.arange(len(df)) * model.dt
    t_span = [t_data[0], t_data[-1]]
    
    # Define column mapping with multiple possible names
    column_mappings = {
        'T_outdoor': ['T_outdoor', 'Tout', 'T_out', 'OutdoorTemp'],
        'Q_internal': ['Q_internal_no_solar', 'Q_internal', 'Qint', 'InternalGains'],
        'I_solar': ['I_solar_total', 'Q_solar_total', 'Qsolar', 'I_solar', 'SolarGain'],
        'T_supply': ['T_supply_avg', 'T_supply', 'Tsupply', 'SupplyTemp'],
        'mdot': ['mdot_water_total', 'mdot', 'mdot_total', 'MassFlow']
    }
    
    # Find matching columns
    input_cols = []
    for key, options in column_mappings.items():
        found = False
        for col in options:
            if col in df.columns:
                input_cols.append(col)
                found = True
                break
        if not found:
            raise ValueError(f"Missing column for '{key}'. Tried: {options}. Available: {list(df.columns)}")
    
    print(f"  Using columns: {input_cols}")
    
    # Extract input arrays
    input_arrays = [df[col].values for col in input_cols]
    
    # Create interpolation functions
    input_funcs = [
        interp1d(t_data, arr, kind='linear', fill_value="extrapolate", assume_sorted=True) 
        for arr in input_arrays
    ]
    
    # Initial conditions
    if 'T_air_avg' in df.columns:
        x0_air = df['T_air_avg'].iloc[0]
    elif 'T_air' in df.columns:
        x0_air = df['T_air'].iloc[0]
    else:
        raise ValueError("Cannot find T_air column for initial condition")
    
    x0_supply = df[input_cols[3]].iloc[0]  # T_supply column
    x0 = np.array([x0_air, x0_air, x0_air] + [x0_supply] * N_sections)
    
    print(f"  Initial conditions: T_air={x0_air:.2f}°C, T_supply={x0_supply:.2f}°C")
    
    # ODE wrapper
    def model_wrapper(t, x):
        u_t = [f(t) for f in input_funcs]
        return model.get_derivatives(t, x, u_t, params_dict)
    
    # Solve ODE
    try:
        sol = solve_ivp(
            model_wrapper, 
            t_span, 
            x0, 
            method='RK45',
            t_eval=t_data, 
            atol=1e-3,
            rtol=1e-3,
            max_step=1200  # 20 min max step
        )
        
        if sol.status == 0:
            return sol.y[0], sol.y  # T_air predictions and all states
        else:
            print(f"  ODE solver warning: {sol.message}")
            return np.full(len(df), np.nan), None
            
    except Exception as e:
        print(f"  Simulation error: {e}")
        return np.full(len(df), np.nan), None


# =============================================================================
#                    VALIDATION FUNCTION
# =============================================================================

def validate_model_on_new_data(params_file, data_file, correct_start_date='2024-04-01 00:20:00', 
                                time_freq='10min', date_filter=None):
    """
    Validate identified RC model on new/unseen data.
    
    Parameters:
    -----------
    params_file : str
        Path to JSON file with identified parameters
    data_file : str
        Path to CSV file with validation data
    correct_start_date : str
        The correct starting datetime for the data (overrides CSV DateTime)
    time_freq : str
        Time frequency between samples (default '10min')
    date_filter : dict, optional
        Filter for date selection, e.g., {'month': 4, 'first_n_days': 7}
        
    Returns:
    --------
    results : dict or None
        Dictionary with validation metrics, or None if failed
    """
    print("=" * 70)
    print("RC MODEL VALIDATION ON UNSEEN DATA")
    print("=" * 70)

    # =========================================================================
    # 1. LOAD PARAMETERS
    # =========================================================================
    print("\nLoading identified parameters...")
    try:
        with open(params_file, 'r') as f:
            results_data = json.load(f)
        
        # Handle different JSON structures (backwards compatibility)
        if 'N_sections' in results_data:
            N_sections = results_data['N_sections']
        elif 'N' in results_data:
            N_sections = results_data['N']
        else:
            raise KeyError("Could not find 'N_sections' or 'N' in JSON file")
        
        identified_params = results_data['identified_parameters']
        
        # Get original metrics if available
        orig_rmse = results_data.get('validation_rmse_C', 
                                     results_data.get('validation_rmse', 'N/A'))
        orig_cv = results_data.get('cv_rmse_percent', 'N/A')
        model_version = results_data.get('model_version', 'unknown')
        
        print(f"  Loaded model: N={N_sections} sections, version={model_version}")
        print(f"  Original validation RMSE: {orig_rmse}")
        print(f"  Original CV(RMSE): {orig_cv}")
        print(f"\n  Identified Parameters:")
        for name, value in identified_params.items():
            print(f"    {name:<10s} = {value:.6e}")
            
    except FileNotFoundError:
        print(f"  ERROR: Parameter file not found: {params_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"  ERROR loading parameters: {e}")
        return None

    # =========================================================================
    # 2. LOAD DATA AND CORRECT DATETIME
    # =========================================================================
    print(f"\nLoading data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
        
        print(f"  Loaded {len(df)} timesteps")
        
        # Check if DateTime column exists in CSV (we'll ignore it)
        if 'DateTime' in df.columns:
            csv_start = pd.to_datetime(df['DateTime'].iloc[0])
            csv_end = pd.to_datetime(df['DateTime'].iloc[-1])
            print(f"  CSV shows dates: {csv_start} to {csv_end}")
            print(f"  NOTE: These dates are INCORRECT - they will be overridden")
        
        # Override DateTime with correct April dates
        correct_start = pd.to_datetime(correct_start_date)
        df['DateTime'] = pd.date_range(start=correct_start, periods=len(df), freq=time_freq)
        
        print(f"  DateTime corrected to April dates")
        print(f"  Correct date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        # Apply date filter if specified
        if date_filter:
            original_len = len(df)
            
            if 'month' in date_filter:
                df = df[df['DateTime'].dt.month == date_filter['month']]
                print(f"  Filtered to month {date_filter['month']}: {len(df)} timesteps")
            
            if 'start_date' in date_filter:
                start = pd.to_datetime(date_filter['start_date'])
                df = df[df['DateTime'] >= start]
            
            if 'end_date' in date_filter:
                end = pd.to_datetime(date_filter['end_date'])
                df = df[df['DateTime'] <= end]
            
            if 'first_n_days' in date_filter and len(df) > 0:
                start = df['DateTime'].min()
                end = start + pd.Timedelta(days=date_filter['first_n_days'])
                df = df[df['DateTime'] < end]
                print(f"  Limited to first {date_filter['first_n_days']} days: {len(df)} timesteps")
            
            df = df.reset_index(drop=True)
            print(f"  Filter result: {original_len} -> {len(df)} timesteps")
            
            if len(df) > 0:
                print(f"  Filtered date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        if len(df) == 0:
            print("  ERROR: No data remaining after filtering!")
            return None
            
    except FileNotFoundError:
        print(f"  ERROR: Data file not found: {data_file}")
        return None
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

    # =========================================================================
    # 3. PREPARE DATA COLUMNS
    # =========================================================================
    print("\nPreparing data columns...")
    
    # Create separated solar columns if needed
    if 'Q_internal_no_solar' not in df.columns:
        if 'Q_solar_total' in df.columns and 'Q_internal_total' in df.columns:
            df['Q_internal_no_solar'] = df['Q_internal_total'] - df['Q_solar_total']
            df['I_solar_total'] = df['Q_solar_total']
            print("  Created Q_internal_no_solar and I_solar_total from existing columns")
        elif 'Q_internal_total' in df.columns:
            df['Q_internal_no_solar'] = df['Q_internal_total']
            df['I_solar_total'] = 0
            print("  No solar data found - setting I_solar_total = 0")
        else:
            print("  ERROR: Cannot find required internal gain columns")
            print(f"     Available columns: {list(df.columns)}")
            return None
    
    # Ensure I_solar_total exists
    if 'I_solar_total' not in df.columns:
        if 'Q_solar_total' in df.columns:
            df['I_solar_total'] = df['Q_solar_total']
            print("  Mapped Q_solar_total -> I_solar_total")
        else:
            df['I_solar_total'] = 0
            print("  No solar column found - setting I_solar_total = 0")
    
    # Check for T_air column
    if 'T_air_avg' not in df.columns:
        if 'T_air' in df.columns:
            df['T_air_avg'] = df['T_air']
            print("  Mapped T_air -> T_air_avg")
        else:
            print("  ERROR: Cannot find T_air or T_air_avg column")
            return None
    
    # Print data summary
    print(f"\n  Data Summary:")
    print(f"    T_air:     {df['T_air_avg'].min():.1f} to {df['T_air_avg'].max():.1f}°C")
    print(f"    T_outdoor: {df['T_outdoor'].min():.1f} to {df['T_outdoor'].max():.1f}°C")
    print(f"    T_supply:  {df['T_supply_avg'].min():.1f} to {df['T_supply_avg'].max():.1f}°C")
    print(f"    I_solar:   {df['I_solar_total'].min():.0f} to {df['I_solar_total'].max():.0f} W")

    # =========================================================================
    # 4. RUN SIMULATION
    # =========================================================================
    print("\nRunning simulation...")
    start_time = time.time()
    
    predictions, all_states = simulate_rc_model(identified_params, df, N_sections)
    
    sim_time = time.time() - start_time
    print(f"  Simulation completed in {sim_time:.2f}s")

    # =========================================================================
    # 5. COMPUTE METRICS
    # =========================================================================
    print("\nComputing validation metrics...")
    
    ground_truth = df['T_air_avg'].values
    valid_mask = ~np.isnan(predictions) & (predictions < 1e5) & (predictions > -50)
    
    n_valid = np.sum(valid_mask)
    n_total = len(predictions)
    
    if n_valid == 0:
        print("  ERROR: Simulation diverged (all values invalid)")
        return None
    
    if n_valid < n_total:
        print(f"  Warning: {n_total - n_valid} invalid points excluded")
    
    valid_preds = predictions[valid_mask]
    valid_truth = ground_truth[valid_mask]
    
    # Compute metrics
    residuals = valid_truth - valid_preds
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    mean_bias = np.mean(residuals)
    std_residual = np.std(residuals)
    max_over = np.min(residuals)  # Most negative = overprediction
    max_under = np.max(residuals)  # Most positive = underprediction
    cv_rmse = (rmse / np.mean(valid_truth)) * 100
    
    # Autocorrelation of residuals
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr = np.nan
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  VALIDATION RESULTS: N={N_sections} RC Model")
    print(f"{'='*70}")
    print(f"  RMSE:           {rmse:.4f}°C")
    print(f"  MAE:            {mae:.4f}°C")
    print(f"  Mean Bias:      {mean_bias:.4f}°C {'BIAS!' if abs(mean_bias) > 0.5 else ''}")
    print(f"  Std Residual:   {std_residual:.4f}°C")
    print(f"  Max Overpred:   {abs(max_over):.4f}°C")
    print(f"  Max Underpred:  {max_under:.4f}°C")
    print(f"  CV(RMSE):       {cv_rmse:.2f}%")
    print(f"  Autocorr:       {autocorr:.3f} {'HIGH!' if abs(autocorr) > 0.5 else ''}")
    print(f"  Valid points:   {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"{'='*70}")
    
    # Comparison with original
    if isinstance(orig_rmse, (int, float)):
        rmse_change = ((rmse - orig_rmse) / orig_rmse) * 100
        print(f"\n  Comparison to original validation:")
        print(f"     Original RMSE: {orig_rmse:.4f}°C")
        print(f"     Current RMSE:  {rmse:.4f}°C")
        print(f"     Change:        {rmse_change:+.1f}%")
    
    # Verdict based on ASHRAE Guideline 14
    print(f"\n  VERDICT (ASHRAE Guideline 14):")
    if cv_rmse < 10:
        verdict = "EXCELLENT - CV(RMSE) < 10%"
        verdict_code = 'excellent'
    elif cv_rmse < 15:
        verdict = "GOOD - CV(RMSE) < 15%"
        verdict_code = 'good'
    elif cv_rmse < 30:
        verdict = "ACCEPTABLE - CV(RMSE) < 30% (ASHRAE threshold)"
        verdict_code = 'acceptable'
    else:
        verdict = "POOR - CV(RMSE) > 30% (exceeds ASHRAE threshold)"
        verdict_code = 'poor'
    
    print(f"     {verdict}")

    # =========================================================================
    # 6. VISUALIZATION
    # =========================================================================
    print("\nCreating validation plots...")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), 
                             gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    fig.suptitle(f'RC Model Validation (N={N_sections}) - Unseen Data (April)\n'
                 f'RMSE: {rmse:.3f}°C | CV(RMSE): {cv_rmse:.2f}% | Bias: {mean_bias:.3f}°C', 
                 fontsize=14, fontweight='bold')
    
    time_vals = df['DateTime'][valid_mask].values
    
    # Plot 1: Temperature comparison
    axes[0].plot(time_vals, valid_truth, 'b-', label='Ground Truth (EnergyPlus)', 
                 linewidth=1.5, alpha=0.8)
    axes[0].plot(time_vals, valid_preds, 'r--', label=f'RC Model (N={N_sections})', 
                 linewidth=1.2)
    axes[0].set_ylabel('Temperature [°C]', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_title('Indoor Air Temperature Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    axes[1].plot(time_vals, residuals, 'g-', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[1].axhline(y=mean_bias, color='orange', linestyle='-', linewidth=1.5, 
                    label=f'Mean Bias: {mean_bias:.3f}°C')
    axes[1].axhline(y=mean_bias + 2*std_residual, color='red', linestyle=':', 
                    linewidth=1, alpha=0.7)
    axes[1].axhline(y=mean_bias - 2*std_residual, color='red', linestyle=':', 
                    linewidth=1, alpha=0.7, label=f'±2σ ({2*std_residual:.2f}°C)')
    axes[1].fill_between(time_vals, residuals, 0, alpha=0.3, color='green')
    axes[1].set_ylabel('Residual [°C]', fontsize=11)
    axes[1].set_title(f'Prediction Residuals (RMSE: {rmse:.3f}°C)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Outdoor temperature context
    axes[2].plot(time_vals, df['T_outdoor'][valid_mask].values, 'purple', linewidth=1)
    axes[2].set_ylabel('T_outdoor [°C]', fontsize=11)
    axes[2].set_title('Outdoor Temperature (Disturbance Context)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Solar radiation
    axes[3].plot(time_vals, df['I_solar_total'][valid_mask].values / 1000, 'orange', linewidth=1)
    axes[3].set_ylabel('Solar [kW]', fontsize=11)
    axes[3].set_xlabel('Date/Time', fontsize=11)
    axes[3].set_title('Solar Radiation (Key Difference from Winter Training)', 
                      fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'validation_N{N_sections}_april_unseen.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {plot_filename}")
    
    plt.show()
    
    # Create parity plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parity plot
    axes2[0].scatter(valid_truth, valid_preds, alpha=0.5, s=10, c='steelblue')
    min_val = min(valid_truth.min(), valid_preds.min())
    max_val = max(valid_truth.max(), valid_preds.max())
    axes2[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    axes2[0].set_xlabel('Ground Truth [°C]', fontsize=11)
    axes2[0].set_ylabel('RC Model [°C]', fontsize=11)
    axes2[0].set_title(f'Parity Plot (R² = {1 - np.var(residuals)/np.var(valid_truth):.3f})', 
                       fontsize=12, fontweight='bold')
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_aspect('equal', adjustable='box')
    
    # Residual histogram
    axes2[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes2[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes2[1].axvline(x=mean_bias, color='orange', linestyle='-', linewidth=2, 
                     label=f'Mean: {mean_bias:.3f}°C')
    axes2[1].set_xlabel('Residual [°C]', fontsize=11)
    axes2[1].set_ylabel('Frequency', fontsize=11)
    axes2[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes2[1].legend(fontsize=10)
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename2 = f'validation_N{N_sections}_april_parity.png'
    plt.savefig(plot_filename2, dpi=150, bbox_inches='tight')
    print(f"  Saved parity plot to {plot_filename2}")
    
    plt.show()

    # =========================================================================
    # 7. SAVE RESULTS
    # =========================================================================
    print("\nSaving validation results...")
    
    validation_results = {
        'validation_period': 'April (first week)' if date_filter else 'Full dataset',
        'date_range': {
            'start': str(df['DateTime'].min()),
            'end': str(df['DateTime'].max())
        },
        'n_timesteps': int(n_valid),
        'model': {
            'N_sections': N_sections,
            'version': model_version,
            'parameters': identified_params
        },
        'metrics': {
            'rmse_C': float(rmse),
            'mae_C': float(mae),
            'mean_bias_C': float(mean_bias),
            'std_residual_C': float(std_residual),
            'cv_rmse_percent': float(cv_rmse),
            'autocorrelation': float(autocorr),
            'max_overprediction_C': float(abs(max_over)),
            'max_underprediction_C': float(max_under)
        },
        'verdict': verdict_code,
        'comparison_to_training': {
            'original_rmse_C': orig_rmse if isinstance(orig_rmse, (int, float)) else None,
            'original_cv_rmse': orig_cv if isinstance(orig_cv, (int, float)) else None
        }
    }
    
    results_filename = f'validation_results_N{N_sections}_april.json'
    with open(results_filename, 'w') as f:
        json.dump(validation_results, f, indent=4)
    print(f"  Saved results to {results_filename}")
    
    # Return results dictionary
    return {
        'rmse': rmse,
        'mae': mae,
        'mean_bias': mean_bias,
        'cv_rmse': cv_rmse,
        'autocorr': autocorr,
        'n_valid': n_valid,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'residuals': residuals,
        'verdict': verdict_code
    }


# =============================================================================
#                    MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    
    print("=" * 70)
    print("  RC MODEL VALIDATION - APRIL SHOULDER SEASON")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    
    # Parameter file (in same folder)
    PARAMETER_FILE = 'results_N3_DE_optimized.json'
    
    # Data file (in same folder) - CSV dates are placeholders, overridden below
    DATA_FILE = 'ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras_processed.csv'
    
    # CORRECT datetime information (overrides CSV DateTime column)
    CORRECT_START_DATE = '2024-04-01 00:20:00'  # Actual start of data
    TIME_FREQUENCY = '10min'  # Time step between samples
    
    # Date filter: First week of April (applied AFTER datetime correction)
    DATE_FILTER = {
        'month': 4,           # April (will work after correction)
        'first_n_days': 7     # First 7 days only
    }
    
    # Alternative date filter options:
    # DATE_FILTER = {'month': 4}  # Entire April
    # DATE_FILTER = {'start_date': '2024-04-01', 'end_date': '2024-04-07'}
    # DATE_FILTER = None  # Use all data (after datetime correction)
    
    # -------------------------------------------------------------------------
    # RUN VALIDATION
    # -------------------------------------------------------------------------
    
    # Check files exist
    files_ok = True
    
    if not os.path.exists(PARAMETER_FILE):
        print(f"\nParameter file not found: {PARAMETER_FILE}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files in directory: {os.listdir('.')}")
        files_ok = False
    
    if not os.path.exists(DATA_FILE):
        print(f"\nData file not found: {DATA_FILE}")
        print(f"   Current directory: {os.getcwd()}")
        files_ok = False
    
    if files_ok:
        # Run validation with datetime correction
        results = validate_model_on_new_data(
            params_file=PARAMETER_FILE,
            data_file=DATA_FILE,
            correct_start_date=CORRECT_START_DATE,
            time_freq=TIME_FREQUENCY,
            date_filter=DATE_FILTER
        )
        
        if results:
            print(f"\n{'='*70}")
            print(f"  VALIDATION COMPLETE!")
            print(f"{'='*70}")
            print(f"  Final RMSE on April data: {results['rmse']:.4f}°C")
            print(f"  CV(RMSE): {results['cv_rmse']:.2f}%")
            print(f"  Verdict: {results['verdict'].upper()}")
            print(f"\n  This model is {'ready' if results['verdict'] in ['excellent', 'good'] else 'marginal'} "
                  f"for use as ground-truth plant in RAMC.")
            print(f"{'='*70}")
        else:
            print("\nValidation failed - check error messages above.")
    else:
        print("\nCannot proceed - required files not found.")
        print("   Make sure both files are in the same folder as this script:")
        print(f"   - {PARAMETER_FILE}")
        print(f"   - {DATA_FILE}")