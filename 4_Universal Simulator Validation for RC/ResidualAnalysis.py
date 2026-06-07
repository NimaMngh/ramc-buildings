# -*- coding: utf-8 -*-
"""
Residual Correlation Analysis for the RC Model
Self-contained script that runs simulation + analysis
Author: Nima Monghasemi
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
#                    RC MODEL
# =============================================================================

class RCModelEquationsWithSolar:
    def __init__(self, N_rad_sections=3):
        self.N = N_rad_sections
        self.dt = 600
        self.Cp_water = 4186.0
        
    def get_derivatives(self, t, x, u, params):
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
        
        T_air = x[0]
        T_env = x[1]
        T_int = x[2]
        T_rad_sections = x[3:]
        
        T_outdoor = u[0]
        Q_internal = u[1]
        I_solar = u[2]
        T_supply = u[3]
        mdot = u[4]
        
        C_rad_section = C_rad / self.N
        temp_diff = np.abs(T_rad_sections - T_air) + 1e-9
        Q_rad_sections = K_rad * np.power(temp_diff, a_rad + 1)
        Q_rad_total = np.sum(Q_rad_sections)
        
        T_rad_in = np.concatenate(([T_supply], T_rad_sections[:-1]))
        dT_rad_dt = (mdot * self.Cp_water * (T_rad_in - T_rad_sections) - Q_rad_sections) / (C_rad_section + 1e-9)
        
        Q_env_air = (T_env - T_air) / R_ae
        Q_int_air = (T_int - T_air) / R_ai
        Q_out_env = (T_outdoor - T_env) / R_ex
        Q_solar = A_sol * I_solar
        
        dT_air_dt = (Q_env_air + Q_int_air + Q_rad_total + Q_internal + Q_solar) / C_air
        dT_env_dt = (Q_out_env - Q_env_air) / C_env
        dT_int_dt = (-Q_int_air) / C_int
        
        return np.concatenate(([dT_air_dt], [dT_env_dt], [dT_int_dt], dT_rad_dt))


def simulate_rc_model(params_dict, df, N_sections):
    model = RCModelEquationsWithSolar(N_rad_sections=N_sections)
    t_data = np.arange(len(df)) * model.dt
    t_span = [t_data[0], t_data[-1]]
    
    input_cols = ['T_outdoor', 'Q_internal_no_solar', 'I_solar_total', 'T_supply_avg', 'mdot_water_total']
    input_arrays = [df[col].values for col in input_cols]
    input_funcs = [interp1d(t_data, arr, kind='linear', fill_value="extrapolate", assume_sorted=True) 
                   for arr in input_arrays]
    
    x0_air = df['T_air_avg'].iloc[0]
    x0_supply = df['T_supply_avg'].iloc[0]
    x0 = np.array([x0_air, x0_air, x0_air] + [x0_supply] * N_sections)
    
    def model_wrapper(t, x):
        u_t = [f(t) for f in input_funcs]
        return model.get_derivatives(t, x, u_t, params_dict)
    
    sol = solve_ivp(model_wrapper, t_span, x0, method='RK45', t_eval=t_data, atol=1e-3, rtol=1e-3, max_step=1200)
    return sol.y[0] if sol.status == 0 else np.full(len(df), np.nan)


# =============================================================================
#                    MAIN EXECUTION
# =============================================================================

print("=" * 70)
print("  COMPLETE RESIDUAL CORRELATION ANALYSIS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. LOAD PARAMETERS
# -----------------------------------------------------------------------------
print("\nLoading identified parameters...")
with open('results_N3_DE_optimized.json', 'r') as f:
    results_data = json.load(f)

identified_params = results_data['identified_parameters']
N_sections = results_data['N_sections']
print(f"  Loaded N={N_sections} model")

# -----------------------------------------------------------------------------
# 2. LOAD AND PREPARE DATA
# -----------------------------------------------------------------------------
print("\nLoading April data...")
df = pd.read_csv("ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras_processed.csv")

# Correct DateTime to April
correct_start = pd.to_datetime('2024-04-01 00:20:00')
df['DateTime'] = pd.date_range(start=correct_start, periods=len(df), freq='10min')

# First week of April
df = df[df['DateTime'].dt.month == 4].head(1007).reset_index(drop=True)
print(f"  Loaded {len(df)} timesteps: {df['DateTime'].min()} to {df['DateTime'].max()}")

# -----------------------------------------------------------------------------
# 3. RUN SIMULATION
# -----------------------------------------------------------------------------
print("\nRunning simulation...")
predictions = simulate_rc_model(identified_params, df, N_sections)
ground_truth = df['T_air_avg'].values
residuals = ground_truth - predictions

rmse = np.sqrt(np.mean(residuals**2))
mean_bias = np.mean(residuals)
print(f"  RMSE: {rmse:.4f}°C, Mean Bias: {mean_bias:+.4f}°C")

# -----------------------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PART 1: CORRELATION WITH INPUT VARIABLES")
print("=" * 70)

input_vars = {
    'I_solar_total': 'Solar Radiation [W]',
    'T_outdoor': 'Outdoor Temperature [°C]',
    'T_supply_avg': 'Supply Temperature [°C]',
    'mdot_water_total': 'Mass Flow Rate [kg/s]',
    'Q_internal_no_solar': 'Internal Gains [W]'
}

print(f"\n  {'Variable':<25s} {'Pearson r':<12s} {'p-value':<12s} {'Interpretation'}")
print("  " + "-" * 75)

correlations = {}
for var, label in input_vars.items():
    r, p = stats.pearsonr(df[var].values, residuals)
    correlations[var] = {'r': r, 'p': p}
    
    if abs(r) > 0.5:
        interp = "STRONG - likely bias source!"
    elif abs(r) > 0.3:
        interp = "Moderate - investigate"
    elif abs(r) > 0.1:
        interp = "○  Weak"
    else:
        interp = "Negligible"
    
    print(f"  {label:<25s} {r:>+.4f}      {p:<.2e}    {interp}")

# -----------------------------------------------------------------------------
# 5. SOLAR-SPECIFIC ANALYSIS (KEY DIAGNOSTIC)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PART 2: SOLAR RADIATION ANALYSIS (KEY DIAGNOSTIC)")
print("=" * 70)

# Linear regression
solar = df['I_solar_total'].values
slope, intercept, r_val, p_val, std_err = stats.linregress(solar, residuals)

print(f"\n  Linear Regression: Residual = a × I_solar + b")
print(f"    Slope (a):     {slope:.8f} °C/W  ({slope*1000:.5f} °C/kW)")
print(f"    Intercept (b): {intercept:+.4f} °C")
print(f"    R²:            {r_val**2:.4f}")
print(f"    p-value:       {p_val:.2e}")

# Bin analysis
print(f"\n  Residuals by Solar Radiation Level:")
print(f"  {'Solar Level':<25s} {'Mean Bias':<12s} {'Std':<10s} {'N':<6s}")
print("  " + "-" * 60)

solar_bins = [(0, 100, 'Night/Cloudy (0-100W)'),
              (100, 1000, 'Low (100W-1kW)'),
              (1000, 5000, 'Medium (1-5kW)'),
              (5000, 15000, 'High (5-15kW)'),
              (15000, 40000, 'Very High (15-40kW)')]

for low, high, label in solar_bins:
    mask = (solar >= low) & (solar < high)
    if mask.sum() > 0:
        bin_residuals = residuals[mask]
        print(f"  {label:<25s} {bin_residuals.mean():>+.4f}°C    {bin_residuals.std():.4f}°C  {mask.sum()}")

# -----------------------------------------------------------------------------
# 6. TIME-BASED ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PART 3: DAYTIME vs NIGHTTIME ANALYSIS")
print("=" * 70)

df['hour'] = pd.to_datetime(df['DateTime']).dt.hour
daytime_mask = (df['hour'] >= 6) & (df['hour'] <= 20)

day_residuals = residuals[daytime_mask]
night_residuals = residuals[~daytime_mask]

print(f"\n  Daytime (06:00-20:00):   Mean Bias = {day_residuals.mean():+.4f}°C, Std = {day_residuals.std():.4f}°C, N = {len(day_residuals)}")
print(f"  Nighttime (20:00-06:00): Mean Bias = {night_residuals.mean():+.4f}°C, Std = {night_residuals.std():.4f}°C, N = {len(night_residuals)}")
print(f"\n  Day - Night difference:  {day_residuals.mean() - night_residuals.mean():+.4f}°C")

if day_residuals.mean() - night_residuals.mean() > 0.3:
    print(f"  Significant daytime bias - confirms solar-related issue!")

# Hourly pattern
print(f"\n  Hourly Bias Pattern:")
print(f"  {'Hour':<6s} {'Mean Bias':<12s} {'Std':<10s}")
print("  " + "-" * 35)

hourly_stats = pd.DataFrame({'residual': residuals, 'hour': df['hour']}).groupby('hour').agg(['mean', 'std'])['residual']
for hour in [0, 6, 9, 12, 15, 18, 21]:
    if hour in hourly_stats.index:
        row = hourly_stats.loc[hour]
        marker = "" if 6 <= hour <= 18 else ""
        print(f"  {hour:02d}:00  {row['mean']:>+.4f}°C    {row['std']:.4f}°C  {marker}")

# -----------------------------------------------------------------------------
# 7. DIAGNOSIS AND RECOMMENDATIONS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  DIAGNOSIS AND RECOMMENDATIONS")
print("=" * 70)

r_solar = correlations['I_solar_total']['r']
current_A_sol = identified_params['A_sol']

print(f"\n  Current A_sol: {current_A_sol:.4f}")
print(f"  Solar correlation: r = {r_solar:+.4f}")

if r_solar > 0.25:
    print(f"\n  DIAGNOSIS: Model UNDERPREDICTS temperature when solar is high.")
    print(f"     The positive correlation (r={r_solar:.3f}) indicates A_sol is too low.")
    
    # Estimate correction
    # At high solar (mean of top quartile), what's the bias?
    high_solar_threshold = np.percentile(solar, 75)
    high_solar_mask = solar > high_solar_threshold
    high_solar_bias = residuals[high_solar_mask].mean()
    high_solar_mean = solar[high_solar_mask].mean()
    
    # Rough estimate: if bias is X°C at solar Y W, and Q_solar = A_sol * Y
    # Missing Q ≈ X * C_air / characteristic_time
    # Simplified heuristic: increase A_sol proportionally to bias
    
    estimated_correction = high_solar_bias / 3  # Heuristic scaling
    suggested_A_sol = min(current_A_sol + estimated_correction * 0.1, 0.95)
    
    print(f"\n  At high solar (>{high_solar_threshold/1000:.1f}kW):")
    print(f"    Mean solar: {high_solar_mean/1000:.1f} kW")
    print(f"    Mean bias:  {high_solar_bias:+.4f}°C")
    
    print(f"\n  RECOMMENDATION:")
    print(f"     Consider increasing A_sol from {current_A_sol:.3f} to ~{suggested_A_sol:.3f}")
    print(f"     Or use seasonal A_sol: Winter={current_A_sol:.2f}, Spring/Fall={suggested_A_sol:.2f}")
    
elif abs(r_solar) < 0.15:
    print(f"\n  Solar is NOT the primary bias source.")
    print(f"     The {mean_bias:+.4f}°C bias may be due to:")
    print(f"     - Envelope parameters (R_ex, R_ae)")
    print(f"     - Internal mass dynamics (C_int, R_ai)")
    print(f"     - Different occupancy patterns in April")

# -----------------------------------------------------------------------------
# 8. VISUALIZATION
# -----------------------------------------------------------------------------
print("\nCreating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Residual Correlation Analysis - April Validation\n'
             f'RMSE: {rmse:.3f}°C | Mean Bias: {mean_bias:+.3f}°C | Solar r: {r_solar:+.3f}', 
             fontsize=14, fontweight='bold')

# Plot 1: Residual vs Solar (MOST IMPORTANT)
ax1 = axes[0, 0]
sc = ax1.scatter(solar/1000, residuals, c=df['hour'], cmap='twilight', alpha=0.5, s=15)
x_fit = np.linspace(0, solar.max()/1000, 100)
ax1.plot(x_fit, intercept + slope*1000*x_fit, 'r-', lw=2, label=f'Fit: r={r_val:.3f}')
ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel('Solar Radiation [kW]')
ax1.set_ylabel('Residual [°C]')
ax1.set_title('Residual vs Solar\n(color = hour)', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax1, label='Hour')

# Plot 2: Residual vs Outdoor Temp
ax2 = axes[0, 1]
ax2.scatter(df['T_outdoor'], residuals, alpha=0.5, s=15, c='steelblue')
r_tout = correlations['T_outdoor']['r']
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Outdoor Temperature [°C]')
ax2.set_ylabel('Residual [°C]')
ax2.set_title(f'Residual vs T_outdoor (r={r_tout:.3f})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Hourly bias pattern
ax3 = axes[0, 2]
hourly_mean = pd.DataFrame({'residual': residuals, 'hour': df['hour']}).groupby('hour')['residual'].mean()
colors = ['navy' if h < 6 or h > 20 else 'gold' for h in hourly_mean.index]
ax3.bar(hourly_mean.index, hourly_mean.values, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(0, color='k', linestyle='--')
ax3.axhline(mean_bias, color='red', lw=2, label=f'Mean: {mean_bias:.3f}°C')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Mean Residual [°C]')
ax3.set_title('Hourly Bias Pattern\n(gold=day, navy=night)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Boxplot by solar level
ax4 = axes[1, 0]
df['solar_cat'] = pd.cut(solar, bins=[0, 100, 1000, 5000, 15000, 40000], 
                          labels=['Night', 'Low', 'Med', 'High', 'V.High'])
box_data = [residuals[df['solar_cat'] == cat] for cat in ['Night', 'Low', 'Med', 'High', 'V.High']]
box_data = [d for d in box_data if len(d) > 0]
bp = ax4.boxplot(box_data, labels=['Night', 'Low', 'Med', 'High', 'V.High'][:len(box_data)], patch_artist=True)
colors_box = ['navy', 'steelblue', 'gold', 'orange', 'red']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.axhline(0, color='k', linestyle='--')
ax4.set_xlabel('Solar Level')
ax4.set_ylabel('Residual [°C]')
ax4.set_title('Residual Distribution by Solar', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Time series
ax5 = axes[1, 1]
time_vals = pd.to_datetime(df['DateTime'])
ax5.fill_between(time_vals, residuals, 0, alpha=0.3, color='green')
ax5.plot(time_vals, residuals, 'g-', lw=0.5)
ax5.axhline(0, color='k', linestyle='--')
ax5_twin = ax5.twinx()
ax5_twin.plot(time_vals, solar/1000, 'orange', lw=1, alpha=0.7)
ax5.set_ylabel('Residual [°C]', color='green')
ax5_twin.set_ylabel('Solar [kW]', color='orange')
ax5.set_title('Residual & Solar Time Series', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Parity with solar coloring
ax6 = axes[1, 2]
sc2 = ax6.scatter(ground_truth, predictions, c=solar/1000, cmap='YlOrRd', alpha=0.5, s=15)
lims = [min(ground_truth.min(), predictions.min()), max(ground_truth.max(), predictions.max())]
ax6.plot(lims, lims, 'r--', lw=2)
ax6.set_xlabel('Ground Truth [°C]')
ax6.set_ylabel('Predictions [°C]')
ax6.set_title('Parity Plot (color=solar kW)', fontweight='bold')
ax6.set_aspect('equal')
ax6.grid(True, alpha=0.3)
plt.colorbar(sc2, ax=ax6, label='Solar [kW]')

plt.tight_layout()
plt.savefig('residual_correlation_analysis_complete.png', dpi=150, bbox_inches='tight')
print(f"  Saved: residual_correlation_analysis_complete.png")
plt.show()

# -----------------------------------------------------------------------------
# 9. SAVE ANALYSIS RESULTS
# -----------------------------------------------------------------------------
analysis_results = {
    'rmse_C': float(rmse),
    'mean_bias_C': float(mean_bias),
    'correlations': {k: {'r': float(v['r']), 'p': float(v['p'])} for k, v in correlations.items()},
    'solar_regression': {
        'slope_C_per_W': float(slope),
        'intercept_C': float(intercept),
        'r_squared': float(r_val**2),
        'p_value': float(p_val)
    },
    'daytime_bias_C': float(day_residuals.mean()),
    'nighttime_bias_C': float(night_residuals.mean()),
    'diagnosis': 'solar_underestimation' if r_solar > 0.25 else 'other_factors'
}

with open('residual_analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=4)
print(f"  Saved: residual_analysis_results.json")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE!")
print("=" * 70)
