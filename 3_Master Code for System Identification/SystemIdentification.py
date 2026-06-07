# -*- coding: utf-8 -*-
"""
RC Model Parameter Identification
Author: Nima Monghasemi
Date: August 2025
Description:
Master script with an explicit solar aperture parameter (A_sol) for
year-round applicability across the full heating season (October-April).

Implementation notes:
- workers=1 (class methods cannot be pickled for parallel evaluation)
- Population size tuned for convergence speed
- Progress tracking with ETA estimates

Model Structure:
- Building: 3 states (T_air, T_env, T_int)
- Radiator: N states (T_rad_1, ..., T_rad_N)
- Parameters: 10 (C_air, C_env, C_int, C_rad, R_ex, R_ae, R_ai, K_rad, a_rad, A_sol)
- Inputs: 5 (T_outdoor, Q_internal, I_solar, T_supply, mdot)
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================================================
#                           CONFIGURATION
# ==========================================================================
# --- Adjust these parameters for different experiments ---
N_SECTIONS_TO_RUN = 3          # Radiator discretization
TIMEOUT_MINUTES = 30           # Hard time limit
STAGNATION_PATIENCE = 100      # Iterations without improvement before stopping
STAGNATION_TOLERANCE = 1e-3    # Minimum improvement to reset stagnation counter
# ==========================================================================


# ==========================================================================
#                    PHASE 1: MODEL DEFINITION (PHYSICS)
# ==========================================================================

class RCModelEquationsWithSolar:
    """
    RC model with an explicit solar aperture parameter.
    Maintains the N-section radiator discretization structure.
    
    Model Equations:
    ----------------
    Building thermal network:
        C_air * dT_air/dt = (T_env - T_air)/R_ae + (T_int - T_air)/R_ai 
                            + Q_rad + Q_int + A_sol * I_solar
        C_env * dT_env/dt = (T_out - T_env)/R_ex - (T_env - T_air)/R_ae
        C_int * dT_int/dt = -(T_int - T_air)/R_ai
    
    Radiator sub-model (N sections):
        C_rad/N * dT_rad_n/dt = mdot * cp * (T_in_n - T_rad_n) - Q_n
        Q_n = K_rad * |T_rad_n - T_air|^(a_rad + 1)
        T_in_1 = T_supply, T_in_n = T_rad_(n-1) for n >= 2
    
    States: [T_air, T_env, T_int, T_rad_1, ..., T_rad_N]  (3 + N states)
    Parameters: [C_air, C_env, C_int, C_rad, R_ex, R_ae, R_ai, K_rad, a_rad, A_sol]  (10 params)
    Inputs: [T_outdoor, Q_internal, I_solar, T_supply, mdot]  (5 inputs)
    """
    
    def __init__(self, N_rad_sections=1):
        self.N = N_rad_sections
        self.dt = 600  # 10-minute timestep [s]
        self.Cp_water = 4186.0  # Water specific heat [J/kg·K]
        
        # Parameter names
        self.param_names = [
            'C_air', 'C_env', 'C_int', 'C_rad', 
            'R_ex', 'R_ae', 'R_ai', 
            'K_rad', 'a_rad', 'A_sol'
        ]
        
        # State names
        self.state_names = ['T_air', 'T_env', 'T_int'] + [f'T_rad_{i+1}' for i in range(self.N)]
        
        # Input names
        self.input_names = ['T_outdoor', 'Q_internal', 'I_solar', 'T_supply', 'mdot']
        
    def get_num_states(self):
        """Return total number of states"""
        return 3 + self.N
    
    def get_num_params(self):
        """Return number of parameters"""
        return len(self.param_names)

    def get_derivatives(self, t, x, u, params):
        """
        Calculates the derivatives dx/dt for the ODE solver.
        
        Parameters:
        -----------
        t : float
            Current time [s]
        x : array (3 + N,)
            State vector [T_air, T_env, T_int, T_rad_1, ..., T_rad_N] [°C]
        u : array (5,)
            Input vector [T_outdoor, Q_internal, I_solar, T_supply, mdot]
            Units: [°C, W, W, °C, kg/s]
        params : dict
            Parameter dictionary with keys matching self.param_names
            
        Returns:
        --------
        dxdt : array (3 + N,)
            Derivative of state vector [°C/s]
        """
        # === Unpack parameters ===
        C_air = params['C_air']    # Air thermal capacity [J/K]
        C_env = params['C_env']    # Envelope thermal capacity [J/K]
        C_int = params['C_int']    # Internal mass thermal capacity [J/K]
        C_rad = params['C_rad']    # Total radiator thermal capacity [J/K]
        R_ex = params['R_ex']      # Envelope-outdoor thermal resistance [K/W]
        R_ae = params['R_ae']      # Air-envelope thermal resistance [K/W]
        R_ai = params['R_ai']      # Air-internal mass thermal resistance [K/W]
        K_rad = params['K_rad']    # Radiator heat transfer coefficient [W/K^(a+1)]
        a_rad = params['a_rad']    # Radiator exponent [-]
        A_sol = params['A_sol']    # Solar aperture [-]
        
        # === Unpack states ===
        T_air = x[0]               # Indoor air temperature [°C]
        T_env = x[1]               # Envelope temperature [°C]
        T_int = x[2]               # Internal mass temperature [°C]
        T_rad_sections = x[3:]     # Radiator section temperatures [°C]
        
        # === Unpack inputs ===
        T_outdoor = u[0]           # Outdoor temperature [°C]
        Q_internal = u[1]          # Internal gains (people+lights+equipment) [W]
        I_solar = u[2]             # Transmitted solar radiation [W]
        T_supply = u[3]            # Supply water temperature [°C]
        mdot = u[4]                # Water mass flow rate [kg/s]
        
        # === RADIATOR SUB-MODEL ===
        # Thermal capacity per section
        C_rad_section = C_rad / self.N
        
        # Nonlinear heat emission from each section
        # Q_n = K_rad * |T_rad_n - T_air|^(a_rad + 1)
        temp_diff = np.abs(T_rad_sections - T_air) + 1e-9  # Avoid division by zero
        Q_rad_sections = K_rad * np.power(temp_diff, a_rad + 1)
        Q_rad_total = np.sum(Q_rad_sections)
        
        # Water temperature propagation through sections
        # T_in_1 = T_supply, T_in_n = T_rad_(n-1) for n >= 2
        T_rad_in = np.concatenate(([T_supply], T_rad_sections[:-1]))
        
        # Radiator section dynamics
        # C_rad/N * dT_rad_n/dt = mdot * cp * (T_in_n - T_rad_n) - Q_n
        dT_rad_dt = (mdot * self.Cp_water * (T_rad_in - T_rad_sections) - Q_rad_sections) / (C_rad_section + 1e-9)
        
        # === BUILDING THERMAL NETWORK ===
        # Heat flow rates [W]
        Q_env_air = (T_env - T_air) / R_ae       # Envelope to air
        Q_int_air = (T_int - T_air) / R_ai       # Internal mass to air
        Q_out_env = (T_outdoor - T_env) / R_ex   # Outdoor to envelope
        Q_solar = A_sol * I_solar                 # Solar gain to air node
        
        # Building state derivatives
        # C_air * dT_air/dt = Q_env_air + Q_int_air + Q_rad + Q_int + Q_solar
        dT_air_dt = (Q_env_air + Q_int_air + Q_rad_total + Q_internal + Q_solar) / C_air
        
        # C_env * dT_env/dt = Q_out_env - Q_env_air
        dT_env_dt = (Q_out_env - Q_env_air) / C_env
        
        # C_int * dT_int/dt = -Q_int_air
        dT_int_dt = (-Q_int_air) / C_int
        
        # === Combine all derivatives ===
        return np.concatenate(([dT_air_dt], [dT_env_dt], [dT_int_dt], dT_rad_dt))
    
    def get_radiator_return_temperature(self, x):
        """
        Get predicted return water temperature for validation.
        This is the temperature of the last radiator section.
        """
        return x[3 + self.N - 1]
    
    def get_total_radiator_heat(self, x, T_air, params):
        """
        Calculate total heat output from radiator [W].
        Useful for energy balance validation.
        """
        T_rad_sections = x[3:]
        K_rad, a_rad = params['K_rad'], params['a_rad']
        temp_diff = np.abs(T_rad_sections - T_air) + 1e-9
        Q_rad_sections = K_rad * np.power(temp_diff, a_rad + 1)
        return np.sum(Q_rad_sections)
    
    def calculate_time_constants(self, params):
        """
        Calculate characteristic time constants for physical validation.
        
        Returns dict with time constants in hours.
        """
        tau_air_env = params['C_air'] * params['R_ae'] / 3600   # Air-envelope [h]
        tau_env_out = params['C_env'] * params['R_ex'] / 3600   # Envelope-outdoor [h]
        tau_air_int = params['C_int'] * params['R_ai'] / 3600   # Air-internal mass [h]
        
        # Approximate radiator time constant (at typical operating point)
        dT_typical = 30  # Typical radiator-air temperature difference [K]
        Q_rad_typical = params['K_rad'] * (dT_typical ** (params['a_rad'] + 1))
        tau_rad = params['C_rad'] / (Q_rad_typical / dT_typical + 1e-9) / 60  # [min]
        
        return {
            'tau_air_env_h': tau_air_env,
            'tau_env_out_h': tau_env_out,
            'tau_air_int_h': tau_air_int,
            'tau_rad_min': tau_rad
        }


# ==========================================================================
#              PHASE 2: PARAMETER IDENTIFICATION FRAMEWORK
# ==========================================================================

class MasterSystemIdentifierWithSolar:
    """
    Parameter identification class with solar aperture support.

    Features:
    - Logarithmic scaling for resistance parameters
    - Robust timeout and stagnation handling
    - Best parameter tracking
    - Physical consistency validation
    """
    
    def __init__(self, model_equations, train_df, val_df, parameter_bounds, log_params=None):
        """
        Initialize the identifier.
        
        Parameters:
        -----------
        model_equations : RCModelEquationsWithSolar
            The RC model object
        train_df : pd.DataFrame
            Training data
        val_df : pd.DataFrame
            Validation data
        parameter_bounds : list of tuples
            [(lower, upper), ...] for each parameter
        log_params : list of str, optional
            Parameter names to optimize in log-space (default: R_ex, R_ae, R_ai)
        """
        self.model = model_equations
        self.train_df = train_df
        self.val_df = val_df
        
        # Parameter configuration
        self.param_names = ['C_air', 'C_env', 'C_int', 'C_rad', 'R_ex', 'R_ae', 'R_ai', 'K_rad', 'a_rad', 'A_sol']
        self.log_params = log_params or ['R_ex', 'R_ae', 'R_ai']
        
        # Bounds setup
        self.lower_bounds = np.array([b[0] for b in parameter_bounds])
        self.upper_bounds = np.array([b[1] for b in parameter_bounds])
        
        # Log-space bounds for scaled optimization
        self.log_lower = np.array([
            np.log10(b[0]) if n in self.log_params else b[0] 
            for n, b in zip(self.param_names, parameter_bounds)
        ])
        self.log_upper = np.array([
            np.log10(b[1]) if n in self.log_params else b[1] 
            for n, b in zip(self.param_names, parameter_bounds)
        ])
        
        # Optimization state tracking
        self.timeout_seconds = None
        self.stagnation_patience = 50
        self.stagnation_tolerance = 1e-5
        self.best_rmse = float('inf')
        self.best_params_scaled = None
        self.stagnant_count = 0
        self.start_time = None
        self.iter_count = 0
        self.last_rmse = float('inf')
        self.force_stop = False
        self.final_runtime = 0
        
        # History tracking for diagnostics
        self.rmse_history = []
        
    def _scale_params(self, scaled_p):
        """Convert from [0,1] scaled space to physical parameter space."""
        log_space = self.log_lower + scaled_p * (self.log_upper - self.log_lower)
        return np.array([
            10**log_space[i] if name in self.log_params else log_space[i] 
            for i, name in enumerate(self.param_names)
        ])
    
    def _unscale_params(self, physical_p):
        """Convert from physical parameter space to [0,1] scaled space."""
        log_space = np.array([
            np.log10(physical_p[i]) if name in self.log_params else physical_p[i] 
            for i, name in enumerate(self.param_names)
        ])
        scaled = (log_space - self.log_lower) / (self.log_upper - self.log_lower)
        return np.clip(scaled, 0, 1)
    
    def _params_to_dict(self, p_physical):
        """Convert parameter array to dictionary."""
        return dict(zip(self.param_names, p_physical))
    
    def _callback_function(self, current_params_scaled):
        """Callback for optimizer - additional safety check."""
        if self.force_stop:
            return True
        
        elapsed = time.time() - self.start_time
        if self.timeout_seconds and elapsed > self.timeout_seconds:
            print(f"\n\n!!! CALLBACK TIMEOUT ({elapsed:.0f}s) !!!")
            self.force_stop = True
            raise KeyboardInterrupt("Hard timeout in callback")
        
        return False

    def simulate(self, params_physical, df):
        """
        Run simulation with the given parameters.
        """
        param_dict = self._params_to_dict(params_physical)
        
        # Time vector
        t_data = np.arange(len(df)) * self.model.dt
        t_span = [t_data[0], t_data[-1]]
        
        input_cols = ['T_outdoor', 'Q_internal_no_solar', 'I_solar_total', 'T_supply_avg', 'mdot_water_total']
        
        # Pre-extract numpy arrays (faster than repeated DataFrame access)
        input_arrays = [df[col].values for col in input_cols]
        
        # Create input interpolation functions
        input_funcs = [
            interp1d(t_data, arr, kind='linear', fill_value="extrapolate", assume_sorted=True) 
            for arr in input_arrays
        ]
        
        # Initial conditions
        x0_air = df['T_air_avg'].iloc[0]
        x0_supply = df['T_supply_avg'].iloc[0]
        x0 = np.array([x0_air, x0_air, x0_air] + [x0_supply] * self.model.N)
        
        # ODE wrapper
        def model_wrapper(t, x):
            u_t = [f(t) for f in input_funcs]
            return self.model.get_derivatives(t, x, u_t, param_dict)
        
        # Solve ODE - USE FASTER SETTINGS
        try:
            sol = solve_ivp(
                model_wrapper, 
                t_span, 
                x0, 
                method='RK45',      # Changed from 'Radau' - much faster for non-stiff
                t_eval=t_data, 
                atol=1e-2,          # Relaxed tolerance
                rtol=1e-2,          # Relaxed tolerance
                max_step=1200       # Allow larger steps (20 min max)
            )
            
            if sol.status == 0:
                return sol.y[0]
            else:
                return np.full(len(df), 1e6)
                
        except Exception:
            return np.full(len(df), 1e6)


    def simulate_full_states(self, params_physical, df):
        """
        Run simulation and return all states (for detailed analysis).
        
        Returns:
        --------
        sol : OdeSolution or None
            Full solution object with all states
        """
        param_dict = self._params_to_dict(params_physical)
        t_data = np.arange(len(df)) * self.model.dt
        t_span = [t_data[0], t_data[-1]]
        
        input_cols = ['T_outdoor', 'Q_internal_no_solar', 'I_solar_total', 'T_supply_avg', 'mdot_water_total']
        
        try:
            input_funcs = [
                interp1d(t_data, df[col].values, kind='linear', fill_value="extrapolate") 
                for col in input_cols
            ]
        except ValueError:
            return None
        
        x0_air = df['T_air_avg'].iloc[0]
        x0_supply = df['T_supply_avg'].iloc[0]
        x0 = np.array([x0_air, x0_air, x0_air] + [x0_supply] * self.model.N)
        
        def model_wrapper(t, x):
            u_t = [f(t) for f in input_funcs]
            return self.model.get_derivatives(t, x, u_t, param_dict)
        
        try:
            sol = solve_ivp(model_wrapper, t_span, x0, method='Radau', t_eval=t_data, atol=1e-3, rtol=1e-3)
            return sol if sol.status == 0 else None
        except:
            return None

    def objective_function(self, scaled_params):
        """
        Objective function for optimization (minimizes training RMSE).
        Includes timeout and stagnation checks.
        """
        if self.force_stop:
            return self.best_rmse if self.best_rmse < float('inf') else 1e6
        
        # Convert to physical parameters
        physical_params = self._scale_params(scaled_params)
        
        # Check for invalid parameters
        if any(np.isnan(physical_params)) or any(np.isinf(physical_params)):
            return 1e6
        
        # Run simulation
        preds = self.simulate(physical_params, self.train_df)
        
        # Check for simulation failure
        if np.any(np.isnan(preds)) or np.any(preds > 1e5):
            return 1e6
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((preds - self.train_df['T_air_avg'].values)**2))
        self.last_rmse = rmse
        self.rmse_history.append(rmse)
        
        elapsed = time.time() - self.start_time
        
        # Update best tracking
        improvement = self.best_rmse - rmse
        if improvement > self.stagnation_tolerance:
            self.best_rmse = rmse
            self.best_params_scaled = scaled_params.copy()
            self.stagnant_count = 0
            improvement_marker = f"  NEW BEST! (Δ={improvement:.4f})"
        else:
            self.stagnant_count += 1
            improvement_marker = ""
        
        # Progress output
        print(f"  Iter {self.iter_count:3d} | RMSE: {rmse:.4f}°C | Best: {self.best_rmse:.4f}°C | "
              f"Stag: {self.stagnant_count:2d}/{self.stagnation_patience} | Time: {elapsed:.0f}s{improvement_marker}")
        
        self.iter_count += 1
        
        # Timeout check
        if self.timeout_seconds and elapsed > self.timeout_seconds:
            print(f"\n!!! HARD TIMEOUT ({elapsed:.0f}s > {self.timeout_seconds}s) !!!")
            self.force_stop = True
            raise KeyboardInterrupt("Hard timeout")
        
        # Stagnation check
        if self.stagnant_count >= self.stagnation_patience:
            print(f"\n!!! STAGNATION STOP ({self.stagnant_count} iters without improvement) !!!")
            print(f"    Best RMSE achieved: {self.best_rmse:.6f}°C")
            self.force_stop = True
            raise KeyboardInterrupt("Stagnation timeout")
        
        return rmse

    def identify_parameters(self, initial_guess_physical, timeout_minutes=35, 
                           stagnation_patience=100, stagnation_tolerance=1e-3):
        """
        Run parameter identification.
        
        Parameters:
        -----------
        initial_guess_physical : array
            Initial parameter values in physical units
        timeout_minutes : float
            Hard time limit
        stagnation_patience : int
            Iterations without improvement before stopping
        stagnation_tolerance : float
            Minimum RMSE improvement to reset stagnation counter
            
        Returns:
        --------
        identified_params : dict or None
            Identified parameters, or None if failed
        """
        # Initialize state
        self.timeout_seconds = timeout_minutes * 60
        self.stagnation_patience = stagnation_patience
        self.stagnation_tolerance = stagnation_tolerance
        self.best_rmse = float('inf')
        self.best_params_scaled = None
        self.stagnant_count = 0
        self.iter_count = 0
        self.start_time = time.time()
        self.force_stop = False
        self.rmse_history = []
        
        print(f"\n{'='*70}")
        print(f"  STARTING PARAMETER IDENTIFICATION (N={self.model.N}, 10 parameters)")
        print(f"{'='*70}")
        print(f"  Hard Time Limit: {timeout_minutes} minutes")
        print(f"  Stagnation: Stop after {stagnation_patience} iters with Δ < {stagnation_tolerance}")
        print(f"  Model: {self.model.get_num_states()} states, {self.model.get_num_params()} parameters")
        print(f"{'='*70}\n")
        
        # Scale initial guess
        initial_guess_scaled = np.clip(self._unscale_params(initial_guess_physical), 0.01, 0.99)
        optimizer_bounds = [(0, 1)] * len(initial_guess_scaled)
        
        final_params_scaled = None
        result_message = ""
        
        try:
            result = minimize(
                self.objective_function,
                initial_guess_scaled,
                method='L-BFGS-B',
                bounds=optimizer_bounds,
                callback=self._callback_function,
                options={
                    'disp': False,
                    'maxiter': 1000,
                    'ftol': 1e-7,
                    'gtol': 1e-6,
                    'maxfun': 2000
                }
            )
            
            if result.success:
                final_params_scaled = result.x
                result_message = "Optimization converged successfully!"
            else:
                final_params_scaled = result.x
                result_message = f"Optimization finished: {result.message}"
                
        except KeyboardInterrupt as e:
            result_message = f"Stopped by timeout/stagnation: {e}"
            if self.best_params_scaled is not None:
                final_params_scaled = self.best_params_scaled
                print(f"    Using best parameters (RMSE: {self.best_rmse:.6f}°C)")
            else:
                final_params_scaled = initial_guess_scaled
                
        except Exception as e:
            result_message = f"Optimization failed: {e}"
            final_params_scaled = self.best_params_scaled if self.best_params_scaled is not None else initial_guess_scaled
        
        self.final_runtime = time.time() - self.start_time
        
        print(f"\n{result_message}")
        print(f"Total time: {self.final_runtime:.1f}s ({self.final_runtime/60:.1f} min)")
        print(f"Iterations: {self.iter_count}")
        print(f"Final Training RMSE: {self.best_rmse:.4f}°C")
        
        if final_params_scaled is not None:
            final_physical_params = self._scale_params(final_params_scaled)
            
            # Print parameter summary
            print(f"\n{'='*60}")
            print("IDENTIFIED PARAMETERS:")
            print(f"{'='*60}")
            for i, name in enumerate(self.param_names):
                value = final_physical_params[i]
                pct = ((np.log10(value) if name in self.log_params else value) - self.log_lower[i]) / \
                      (self.log_upper[i] - self.log_lower[i]) * 100
                unit = self._get_param_unit(name)
                print(f"  {name:<10s} = {value:>12.4e} {unit:<8s} ({pct:5.1f}% of range)")
            print(f"{'='*60}")
            
            return self._params_to_dict(final_physical_params)
        else:
            print("\nNo valid parameters identified.")
            return None
    
    def _get_param_unit(self, name):
        """Get unit string for parameter."""
        units = {
            'C_air': 'J/K', 'C_env': 'J/K', 'C_int': 'J/K', 'C_rad': 'J/K',
            'R_ex': 'K/W', 'R_ae': 'K/W', 'R_ai': 'K/W',
            'K_rad': 'W/K^a', 'a_rad': '-', 'A_sol': '-'
        }
        return units.get(name, '')
    
    def identify_parameters_global(self, timeout_minutes=30, 
                                population_size=5, max_generations=25):
        """
        Global parameter identification using Differential Evolution.
        
        FIXES APPLIED:
        - polish=False (prevents silent post-timeout optimization)
        - Hard timeout inside objective function
        - Reduced max_generations to fit within timeout
        """
        from scipy.optimize import differential_evolution
        
        # Initialize state
        self.timeout_seconds = timeout_minutes * 60
        self.best_rmse = float('inf')
        self.best_params_scaled = None
        self.iter_count = 0
        self.start_time = time.time()
        self.force_stop = False
        self.rmse_history = []
        
        n_params = len(self.param_names)
        actual_popsize = population_size * n_params
        
        print(f"\n{'='*70}")
        print(f"  GLOBAL OPTIMIZATION: Differential Evolution (N={self.model.N})")
        print(f"{'='*70}")
        print(f"  Parameters: {n_params}")
        print(f"  Population: {actual_popsize} candidates")
        print(f"  Max generations: {max_generations}")
        print(f"  Time limit: {timeout_minutes} minutes")
        print(f"  Strategy: best1bin (NO polish) - workers=1")
        print(f"{'='*70}\n")
        
        # Bounds in [0, 1] scaled space
        bounds_scaled = [(0.0, 1.0)] * n_params
        
        # Generation counter for callback
        generation_count = [0]
        
        def de_callback(xk, convergence):
            """Callback after each generation."""
            generation_count[0] += 1
            elapsed = time.time() - self.start_time
            
            # Check timeout
            if elapsed > self.timeout_seconds:
                print(f"\n!!! TIMEOUT after {elapsed/60:.1f} minutes !!!")
                self.force_stop = True
                return True
            
            # Progress report every generation
            time_per_eval = elapsed / max(self.iter_count, 1)
            remaining_evals = (max_generations - generation_count[0]) * actual_popsize
            eta_minutes = remaining_evals * time_per_eval / 60
            
            print(f"  Gen {generation_count[0]:3d} | Best: {self.best_rmse:.4f}°C | "
                  f"Conv: {convergence:.4f} | Evals: {self.iter_count} | "
                  f"Time: {elapsed/60:.1f}min | ETA: {eta_minutes:.1f}min")
            
            return False
        
        def objective_wrapper(scaled_params):
            """Objective function wrapper with HARD TIMEOUT."""
            # === HARD TIMEOUT CHECK (works even during polish if enabled) ===
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds or self.force_stop:
                return self.best_rmse if self.best_rmse < float('inf') else 1e6
            
            self.iter_count += 1
            
            # Convert to physical parameters
            physical_params = self._scale_params(scaled_params)
            
            # Quick validity check
            if any(np.isnan(physical_params)) or any(np.isinf(physical_params)):
                return 1e6
            
            # Run simulation
            preds = self.simulate(physical_params, self.train_df)
            
            # Check for simulation failure
            if np.any(np.isnan(preds)) or np.any(preds > 1e5):
                return 1e6
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((preds - self.train_df['T_air_avg'].values)**2))
            self.rmse_history.append(rmse)
            
            # Track best
            if rmse < self.best_rmse:
                improvement = self.best_rmse - rmse
                self.best_rmse = rmse
                self.best_params_scaled = scaled_params.copy()
                if improvement > 0.02:  # Print improvements > 0.02°C
                    print(f"    New best: {rmse:.4f}°C (Δ={improvement:.3f}) @ eval {self.iter_count}")
            
            return rmse
        
        try:
            result = differential_evolution(
                objective_wrapper,
                bounds=bounds_scaled,
                strategy='best1bin',
                maxiter=max_generations,
                popsize=population_size,
                tol=0.001,
                atol=0.001,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                callback=de_callback,
                polish=False,          # No silent L-BFGS-B refinement after DE
                init='latinhypercube',
                updating='immediate',
                workers=1
            )
            
            final_params_scaled = result.x
            self.best_rmse = result.fun
            
            if result.success:
                result_message = f"DE converged: {result.message}"
            else:
                result_message = f"DE finished: {result.message}"
                
        except Exception as e:
            result_message = f"DE stopped: {e}"
            final_params_scaled = self.best_params_scaled
        
        self.final_runtime = time.time() - self.start_time
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"  {result_message}")
        print(f"  Total time: {self.final_runtime/60:.1f} minutes")
        print(f"  Function evaluations: {self.iter_count}")
        print(f"  Final Training RMSE: {self.best_rmse:.4f}°C")
        print(f"{'='*70}")
        
        if final_params_scaled is not None:
            final_physical_params = self._scale_params(final_params_scaled)
            
            print(f"\nIDENTIFIED PARAMETERS (from DE):")
            print(f"{'='*60}")
            for i, name in enumerate(self.param_names):
                value = final_physical_params[i]
                pct = ((np.log10(value) if name in self.log_params else value) - self.log_lower[i]) / \
                      (self.log_upper[i] - self.log_lower[i]) * 100
                unit = self._get_param_unit(name)
                print(f"  {name:<10s} = {value:>12.4e} {unit:<8s} ({pct:5.1f}% of range)")
            print(f"{'='*60}")
            
            return self._params_to_dict(final_physical_params)
        else:
            print("\nNo valid parameters identified.")
            return None



# ==========================================================================
#                    PHASE 3: DIAGNOSTIC FUNCTIONS
# ==========================================================================

def diagnose_data_quality(df, name="Data"):
    """Data quality diagnostics."""
    print(f"\n{'='*50}")
    print(f"  DATA QUALITY: {name}")
    print(f"{'='*50}")
    print(f"  Timesteps: {len(df)}")
    print(f"  Duration: {len(df) * 10 / 60:.1f} hours ({len(df) * 10 / 60 / 24:.1f} days)")
    
    # Temperature ranges
    print(f"\n  Temperature Ranges:")
    print(f"    T_air:     {df['T_air_avg'].min():.1f} to {df['T_air_avg'].max():.1f}°C (σ={df['T_air_avg'].std():.2f})")
    print(f"    T_outdoor: {df['T_outdoor'].min():.1f} to {df['T_outdoor'].max():.1f}°C")
    print(f"    T_supply:  {df['T_supply_avg'].min():.1f} to {df['T_supply_avg'].max():.1f}°C")
    
    # Flow and heat rates
    print(f"\n  Flow and Heat Rates:")
    print(f"    mdot:      {df['mdot_water_total'].min():.4f} to {df['mdot_water_total'].max():.4f} kg/s")
    print(f"    Q_int:     {df['Q_internal_no_solar'].min():.0f} to {df['Q_internal_no_solar'].max():.0f} W")
    print(f"    I_solar:   {df['I_solar_total'].min():.0f} to {df['I_solar_total'].max():.0f} W")
    
    # Warnings
    warnings_found = False
    if df['mdot_water_total'].max() < 0.01:
        print(f"\n  WARNING: Very low mass flow rates!")
        warnings_found = True
    if df['T_supply_avg'].std() < 1.0:
        print(f"\n  WARNING: Low supply temperature variation!")
        warnings_found = True
    if df['T_air_avg'].std() < 0.5:
        print(f"\n  WARNING: Low indoor temperature variation - poor excitation!")
        warnings_found = True
    
    if not warnings_found:
        print(f"\n  Data quality looks good!")
    
    print(f"{'='*50}")


def check_physical_consistency(params, model):
    """Validate identified parameters for physical reasonableness."""
    print(f"\n{'='*60}")
    print("  PHYSICAL CONSISTENCY CHECK")
    print(f"{'='*60}")
    
    # Calculate time constants
    time_constants = model.calculate_time_constants(params)
    
    print(f"\n  Time Constants:")
    print(f"    Air-Envelope (C_air × R_ae):    {time_constants['tau_air_env_h']:.2f} hours")
    print(f"    Envelope-Outdoor (C_env × R_ex): {time_constants['tau_env_out_h']:.1f} hours")
    print(f"    Air-Internal (C_int × R_ai):    {time_constants['tau_air_int_h']:.1f} hours")
    print(f"    Radiator (approx):              {time_constants['tau_rad_min']:.1f} minutes")
    
    # Check expected ranges
    issues = []
    
    if time_constants['tau_air_env_h'] < 0.1 or time_constants['tau_air_env_h'] > 10:
        issues.append(f"Air-envelope time constant ({time_constants['tau_air_env_h']:.2f}h) outside typical 0.1-10h")
    
    if time_constants['tau_env_out_h'] < 10 or time_constants['tau_env_out_h'] > 500:
        issues.append(f"Envelope time constant ({time_constants['tau_env_out_h']:.1f}h) outside typical 10-500h")
    
    if time_constants['tau_rad_min'] < 0.5 or time_constants['tau_rad_min'] > 60:
        issues.append(f"Radiator time constant ({time_constants['tau_rad_min']:.1f}min) outside typical 0.5-60min")
    
    # Heat transfer characteristics
    UA_envelope = 1 / params['R_ex']
    print(f"\n  Heat Transfer:")
    print(f"    Envelope UA: {UA_envelope:.1f} W/K")
    print(f"    Solar aperture: {params['A_sol']*100:.1f}% of transmitted solar")
    
    if UA_envelope < 100 or UA_envelope > 5000:
        issues.append(f"Envelope UA ({UA_envelope:.0f} W/K) outside typical range for commercial buildings")
    
    if params['A_sol'] < 0.3 or params['A_sol'] > 1.0:
        issues.append(f"Solar aperture ({params['A_sol']:.2f}) outside expected 0.3-1.0 range")
    
    # Radiator characteristics
    print(f"\n  Radiator Characteristics:")
    print(f"    K_rad: {params['K_rad']:.2f} W/K^(a+1)")
    print(f"    a_rad: {params['a_rad']:.3f} (typical for convectors: 0.2-0.35)")
    
    if params['a_rad'] < 0.1 or params['a_rad'] > 0.6:
        issues.append(f"Radiator exponent ({params['a_rad']:.3f}) outside typical 0.1-0.6 range")
    
    # Summary
    print(f"\n  {'='*40}")
    if issues:
        print(f"  WARNINGS ({len(issues)}):")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"  All parameters within expected physical ranges!")
    print(f"{'='*60}")
    
    return len(issues) == 0


def analyze_residuals(ground_truth, predictions, name=""):
    """Analyze prediction residuals for model diagnostics."""
    residuals = ground_truth - predictions
    
    print(f"\n{'='*50}")
    print(f"  RESIDUAL ANALYSIS {name}")
    print(f"{'='*50}")
    
    # Basic statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    rmse = np.sqrt(np.mean(residuals**2))
    
    print(f"  RMSE:        {rmse:.4f}°C")
    print(f"  Mean:        {mean_res:.4f}°C {'BIAS!' if abs(mean_res) > 0.3 else ''}")
    print(f"  Std:         {std_res:.4f}°C")
    print(f"  Max over:    {np.min(residuals):.4f}°C")
    print(f"  Max under:   {np.max(residuals):.4f}°C")
    
    # Autocorrelation (indicates missing dynamics)
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        print(f"  Autocorr:    {autocorr:.3f} {'HIGH!' if abs(autocorr) > 0.5 else ''}")
    
    # CV(RMSE) for ASHRAE-style reporting
    mean_temp = np.mean(ground_truth)
    cv_rmse = (rmse / mean_temp) * 100
    print(f"  CV(RMSE):    {cv_rmse:.2f}%")
    
    print(f"{'='*50}")
    
    return residuals


# ==========================================================================
#                    PHASE 4: VISUALIZATION
# ==========================================================================

def create_detailed_validation_plot(val_df, val_preds, identified_params, N, save_path=None):
    """Create the validation visualization."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'RC Model Validation (N={N}, 10 Parameters with Solar)', fontsize=16, fontweight='bold')
    
    time_vals = pd.to_datetime(val_df['DateTime'])
    ground_truth = val_df['T_air_avg'].values
    residuals = ground_truth - val_preds
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Plot 1: Temperature comparison
    axes[0, 0].plot(time_vals, ground_truth, 'b-', label='EnergyPlus (Ground Truth)', linewidth=2, alpha=0.8)
    axes[0, 0].plot(time_vals, val_preds, 'r--', label=f'RC Model (N={N})', linewidth=1.5)
    axes[0, 0].set_ylabel('Temperature [°C]', fontsize=11)
    axes[0, 0].set_title('Indoor Air Temperature Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals over time
    axes[0, 1].plot(time_vals, residuals, 'g-', linewidth=0.8)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[0, 1].fill_between(time_vals, residuals, 0, alpha=0.3, color='green')
    axes[0, 1].set_ylabel('Residual [°C]', fontsize=11)
    axes[0, 1].set_title(f'Prediction Residuals (RMSE: {rmse:.3f}°C)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Supply temperature and flow rate
    ax3a = axes[1, 0]
    ax3b = ax3a.twinx()
    l1 = ax3a.plot(time_vals, val_df['T_supply_avg'], 'r-', label='T_supply', linewidth=1.5)
    l2 = ax3b.plot(time_vals, val_df['mdot_water_total'] * 1000, 'b-', label='mdot', linewidth=1, alpha=0.7)
    ax3a.set_ylabel('Supply Temp [°C]', color='r', fontsize=11)
    ax3b.set_ylabel('Flow Rate [g/s]', color='b', fontsize=11)
    ax3a.set_title('Radiator Inputs', fontsize=12, fontweight='bold')
    ax3a.grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3a.legend(lines, labels, loc='upper right', fontsize=9)
    
    # Plot 4: Outdoor temp, internal gains, and solar
    ax4a = axes[1, 1]
    ax4b = ax4a.twinx()
    l1 = ax4a.plot(time_vals, val_df['T_outdoor'], 'g-', label='T_outdoor', linewidth=1.5)
    l2 = ax4b.plot(time_vals, val_df['Q_internal_no_solar'] / 1000, 'orange', label='Q_int', linewidth=1, alpha=0.7)
    l3 = ax4b.plot(time_vals, val_df['I_solar_total'] / 1000, 'gold', label='I_solar', linewidth=1, alpha=0.7)
    ax4a.set_ylabel('Outdoor Temp [°C]', color='g', fontsize=11)
    ax4b.set_ylabel('Heat Gains [kW]', color='orange', fontsize=11)
    ax4a.set_title('Disturbance Inputs', fontsize=12, fontweight='bold')
    ax4a.grid(True, alpha=0.3)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax4a.legend(lines, labels, loc='upper right', fontsize=9)
    
    # Plot 5: Residual histogram
    axes[2, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[2, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[2, 0].axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals):.3f}°C')
    axes[2, 0].set_xlabel('Residual [°C]', fontsize=11)
    axes[2, 0].set_ylabel('Frequency', fontsize=11)
    axes[2, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[2, 0].legend(fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Scatter plot
    axes[2, 1].scatter(ground_truth, val_preds, alpha=0.5, s=10, c='steelblue')
    min_val = min(ground_truth.min(), val_preds.min())
    max_val = max(ground_truth.max(), val_preds.max())
    axes[2, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    axes[2, 1].set_xlabel('EnergyPlus [°C]', fontsize=11)
    axes[2, 1].set_ylabel('RC Model [°C]', fontsize=11)
    axes[2, 1].set_title('Prediction vs Ground Truth', fontsize=12, fontweight='bold')
    axes[2, 1].legend(fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved validation plot to {save_path}")
    
    plt.show()
    
    return fig


def create_convergence_plot(rmse_history, N, save_path=None):
    """Plot optimization convergence history."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(rmse_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Training RMSE [°C]', fontsize=12)
    ax.set_title(f'Optimization Convergence (N={N})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark best
    best_iter = np.argmin(rmse_history)
    best_rmse = rmse_history[best_iter]
    ax.scatter([best_iter], [best_rmse], color='red', s=100, zorder=5, label=f'Best: {best_rmse:.4f}°C @ iter {best_iter}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig


# ==========================================================================
#                    PHASE 5: MAIN EXECUTION
# ==========================================================================

if __name__ == '__main__':
    print("=" * 80)
    print(f"  RC MODEL PARAMETER IDENTIFICATION - GLOBAL OPTIMIZATION (v5.1)")
    print(f"     Radiator Sections: N = {N_SECTIONS_TO_RUN}")
    print(f"     Optimizer: Differential Evolution (workers=1) + L-BFGS-B polish")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # DATA LOADING (MUST COME FIRST)
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    
    try:
        df = pd.read_csv("ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras_processed.csv")
        
        required_cols = [
            'T_air_avg', 'T_outdoor', 'T_supply_avg', 'mdot_water_total', 'DateTime'
        ]
        
        if 'Q_internal_no_solar' not in df.columns:
            print("  'Q_internal_no_solar' not found - checking for alternatives...")
            
            if 'Q_solar_total' in df.columns and 'Q_internal_total' in df.columns:
                df['Q_internal_no_solar'] = df['Q_internal_total'] - df['Q_solar_total']
                df['I_solar_total'] = df['Q_solar_total']
                print("  Created separated solar columns from existing data")
            elif 'I_solar_total' not in df.columns:
                print("  Solar data not found - assuming Q_internal includes solar")
                if 'Q_internal_total' in df.columns:
                    df['Q_internal_no_solar'] = df['Q_internal_total']
                    df['I_solar_total'] = 0
                    print("  WARNING: Solar gain set to 0 - A_sol will not be identifiable!")
                else:
                    print("  CRITICAL: Neither Q_internal_total nor Q_internal_no_solar found!")
                    exit(1)
        
        all_required = required_cols + ['Q_internal_no_solar', 'I_solar_total']
        missing = [col for col in all_required if col not in df.columns]
        if missing:
            print(f"  CRITICAL: Missing columns: {missing}")
            print(f"     Available: {list(df.columns)}")
            exit(1)
        
        print(f"  All required columns present")
        
        df = df.dropna()
        
        train_df = df.iloc[:2016].reset_index(drop=True)
        val_df = df.iloc[2016:3024].reset_index(drop=True)
        
        print(f"  Data split: {len(train_df)} training, {len(val_df)} validation points")
        
    except FileNotFoundError:
        print("  Data file not found!")
        print("     Expected: ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras_processed.csv")
        exit(1)
    
    # -------------------------------------------------------------------------
    # DATA QUALITY DIAGNOSTICS
    # -------------------------------------------------------------------------
    diagnose_data_quality(train_df, "Training Data")
    diagnose_data_quality(val_df, "Validation Data")
    
    # -------------------------------------------------------------------------
    # MODEL SETUP
    # -------------------------------------------------------------------------
    print(f"\nSetting up RC model with N={N_SECTIONS_TO_RUN} radiator sections...")
    
    rc_model = RCModelEquationsWithSolar(N_rad_sections=N_SECTIONS_TO_RUN)
    print(f"  States: {rc_model.get_num_states()} ({', '.join(rc_model.state_names)})")
    print(f"  Parameters: {rc_model.get_num_params()} ({', '.join(rc_model.param_names)})")
    print(f"  Inputs: {len(rc_model.input_names)} ({', '.join(rc_model.input_names)})")
    
# -------------------------------------------------------------------------
# PARAMETER BOUNDS (TIGHTENED for physical realism and faster solving)
# -------------------------------------------------------------------------
    parameter_bounds = [
        (1e7, 2e8),      # C_air [J/K] 
        (1e8, 1e10),     # C_env [J/K] 
        (5e8, 1.2e10),   # C_int [J/K] 
        (5e5, 5e6),      # C_rad [J/K] - tightened upper bound
        (2e-4, 1e-2),    # R_ex [K/W] - TIGHTENED (was 1e-6 to 1e-3) -> UA 100-5000 W/K
        (5e-5, 5e-3),    # R_ae [K/W] - TIGHTENED (was 1e-5 to 1e-2)
        (5e-5, 5e-3),    # R_ai [K/W] - TIGHTENED (was 1e-6 to 1e-3)
        (10.0, 300.0),   # K_rad [W/K^(a+1)] - raised lower bound
        (0.2, 0.4),      # a_rad [-] - tightened to typical convector range
        (0.4, 0.9)       # A_sol [-] - tightened
    ]

    
    print(f"\n  Parameter bounds configured.")
    
    # -------------------------------------------------------------------------
    # PARAMETER IDENTIFICATION (GLOBAL - Differential Evolution)
    # -------------------------------------------------------------------------
    identifier = MasterSystemIdentifierWithSolar(
        rc_model, train_df, val_df, parameter_bounds
    )
    
    # Use global optimization with the configured settings
    identified_params = identifier.identify_parameters_global(
        timeout_minutes=45,      # I need to change this based on the model complexity
        population_size=5,       # 50 candidates per generation
        max_generations=20       # ~20 gens should fit in 25 min
    )
    
    # -------------------------------------------------------------------------
    # VALIDATION AND ANALYSIS
    # -------------------------------------------------------------------------
    if identified_params:
        print("\n" + "=" * 70)
        print("  VALIDATION AND ANALYSIS")
        print("=" * 70)
        
        is_consistent = check_physical_consistency(identified_params, rc_model)
        
        print("\nRunning validation simulation...")
        val_preds = identifier.simulate(list(identified_params.values()), val_df)
        
        if np.any(np.isnan(val_preds)) or np.any(val_preds > 1e5):
            print("  Validation simulation failed!")
        else:
            val_rmse = np.sqrt(np.mean((val_preds - val_df['T_air_avg'].values)**2))
            cv_rmse = (val_rmse / val_df['T_air_avg'].mean()) * 100
            
            residuals = analyze_residuals(
                val_df['T_air_avg'].values, 
                val_preds, 
                f"(N={N_SECTIONS_TO_RUN})"
            )
            
            # -------------------------------------------------------------------------
            # FINAL RESULTS SUMMARY
            # -------------------------------------------------------------------------
            print(f"\n{'='*70}")
            print(f"  FINAL RESULTS: N={N_SECTIONS_TO_RUN} RADIATOR SECTIONS (DE)")
            print(f"{'='*70}")
            print(f"  Validation RMSE:  {val_rmse:.4f}°C")
            print(f"  CV(RMSE):         {cv_rmse:.2f}%")
            print(f"  Mean Residual:    {np.mean(residuals):.4f}°C")
            print(f"  Runtime:          {identifier.final_runtime:.1f}s ({identifier.final_runtime/60:.1f} min)")
            print(f"  Function Evals:   {identifier.iter_count}")
            print(f"\n  Identified Parameters:")
            for name, value in identified_params.items():
                print(f"    {name:<10s} = {value:.4e}")
            print(f"{'='*70}")
            
            # -------------------------------------------------------------------------
            # SAVE RESULTS
            # -------------------------------------------------------------------------
            print("\nSaving results...")
            
            results_data = {
                "model_version": "v5.1_DE_optimized",
                "optimizer": "differential_evolution_workers1",
                "N_sections": N_SECTIONS_TO_RUN,
                "n_states": rc_model.get_num_states(),
                "n_parameters": rc_model.get_num_params(),
                "validation_rmse_C": float(val_rmse),
                "cv_rmse_percent": float(cv_rmse),
                "mean_residual_C": float(np.mean(residuals)),
                "residual_autocorrelation": float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]),
                "runtime_seconds": float(identifier.final_runtime),
                "function_evaluations": identifier.iter_count,
                "physically_consistent": is_consistent,
                "identified_parameters": {k: float(v) for k, v in identified_params.items()},
                "time_constants": rc_model.calculate_time_constants(identified_params)
            }
            
            json_filename = f"results_N{N_SECTIONS_TO_RUN}_DE_optimized.json"
            with open(json_filename, 'w') as f:
                json.dump(results_data, f, indent=4)
            print(f"  Saved metadata to {json_filename}")
            
            npz_filename = f"predictions_N{N_SECTIONS_TO_RUN}_DE_optimized.npz"
            np.savez(
                npz_filename,
                predictions=val_preds,
                ground_truth=val_df['T_air_avg'].values,
                residuals=residuals,
                timestamps=val_df['DateTime'].values,
                rmse_history=np.array(identifier.rmse_history)
            )
            print(f"  Saved predictions to {npz_filename}")
            
            # -------------------------------------------------------------------------
            # VISUALIZATION
            # -------------------------------------------------------------------------
            print("\nCreating visualizations...")
            
            if len(identifier.rmse_history) > 1:
                create_convergence_plot(
                    identifier.rmse_history, 
                    N_SECTIONS_TO_RUN,
                    f"convergence_N{N_SECTIONS_TO_RUN}_DE_optimized.png"
                )
            
            create_detailed_validation_plot(
                val_df, 
                val_preds, 
                identified_params, 
                N_SECTIONS_TO_RUN,
                f"validation_N{N_SECTIONS_TO_RUN}_DE_optimized.png"
            )
            
            print(f"\nIDENTIFICATION COMPLETE!")
            print(f"   Model: {rc_model.get_num_states()} states, {rc_model.get_num_params()} parameters")
            print(f"   RMSE: {val_rmse:.4f}°C | CV(RMSE): {cv_rmse:.2f}%")
            
    else:
        print("\nParameter identification failed - no valid solution found.")
        print("   Consider adjusting bounds, initial guess, or timeout settings.")