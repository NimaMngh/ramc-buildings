# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 14:06:12 2026

@author: nmi03
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, 'ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.csv')
output_curve = os.path.join(SCRIPT_DIR, 'Vasteras_Curve_Verification.png')
output_flow = os.path.join(SCRIPT_DIR, 'MassFlow_Analysis.png')

# Load data
print("Loading data...")
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# --- List the relevant columns available in the file ---
print("\n" + "="*50)
print("AVAILABLE COLUMNS")
print("="*50)

# Search for temperature-related columns
print("\nColumns containing 'Node' and 'Temperature':")
for col in df.columns:
    if 'Node' in col and 'Temperature' in col:
        print(f"  - {col}")

print("\nColumns containing 'Supply' or 'Inlet' or 'Outlet':")
for col in df.columns:
    if any(x in col for x in ['Supply', 'Inlet', 'Outlet', 'SUPPLY', 'INLET', 'OUTLET']):
        print(f"  - {col}")

print("\nColumns containing 'District' or 'Heating':")
for col in df.columns:
    if 'District' in col or 'Heating' in col:
        print(f"  - {col}")

print("\nColumns containing 'Flow':")
for col in df.columns:
    if 'Flow' in col:
        print(f"  - {col}")

print("="*50)

# --- 1. Column names used in the analysis ---
# Outdoor air dry-bulb temperature
oadb_col = 'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'

# Supply and return water temperatures
sup_temp_col = 'HW SUPPLY OUTLET NODE:System Node Temperature [C](Hourly)'
ret_temp_col = 'HW SUPPLY INLET NODE:System Node Temperature [C](Hourly)'

# Energy Rate (Power in Watts)
power_col = 'DISTRICT HEATING UTILITY:District Heating Water Rate [W](Hourly)'

# Mass Flow Column
flow_col = 'DISTRICT HEATING UTILITY:District Heating Water Mass Flow Rate [kg/s](Hourly)'

# --- 2. VERIFY COLUMNS EXIST ---
missing_cols = []
for col_name, col_var in [('Supply Temp', sup_temp_col), 
                           ('Return Temp', ret_temp_col),
                           ('Outdoor Temp', oadb_col)]:
    if col_var not in df.columns:
        missing_cols.append(f"  - {col_name}: {col_var}")

if missing_cols:
    print("\nERROR: The following required columns were not found:")
    for msg in missing_cols:
        print(msg)
    print("\nUpdate the column names in the script to match the available columns above.")
    sys.exit(1)

# --- 3. CALCULATE MISSING DATA ---
# Calculate Delta T
df['Delta_T'] = df[sup_temp_col] - df[ret_temp_col]
df['Outdoor_Temp'] = df[oadb_col]

# Determine Flow Rate
if flow_col in df.columns:
    print(f"\nFound explicit Mass Flow column: {flow_col}")
    df['Calculated_Flow'] = df[flow_col]
elif power_col in df.columns:
    print(f"\nMass Flow column missing. Calculating from Heating Power: {power_col}")
    df['Calculated_Flow'] = 0.0
    mask = (df[power_col] > 100) & (df['Delta_T'] > 0.5)
    df.loc[mask, 'Calculated_Flow'] = df.loc[mask, power_col] / (4180 * df.loc[mask, 'Delta_T'])
else:
    print("ERROR: Neither Mass Flow nor Heating Power columns found.")
    sys.exit(1)

# Filter dataset to Heating Season (Active Heating) for cleaner plots
plot_df = df[df['Calculated_Flow'] > 0.01].copy()

if len(plot_df) == 0:
    print("WARNING: No data points with Calculated_Flow > 0.01. Check your data.")
    sys.exit(1)

# --- PLOT 1: VERIFY THE VÄSTERÅS CURVE ---
plt.figure(figsize=(10, 6))
sc = plt.scatter(plot_df['Outdoor_Temp'], plot_df[sup_temp_col], 
                 alpha=0.5, c=plot_df['Delta_T'], cmap='viridis', label='Actual Supply T')
cbar = plt.colorbar(sc)
cbar.set_label('Delta T (°C)')

x_ref = [-20, 20]
y_ref = [60, 19]
plt.plot(x_ref, y_ref, color='red', linestyle='--', linewidth=3, label='Setpoint Curve (60°C to 19°C)')

plt.title('Verification of Supply Temperature Control', fontsize=14, fontweight='bold')
plt.xlabel('Outdoor Temperature (°C)')
plt.ylabel('Supply Water Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(output_curve, dpi=300)
plt.show()

# --- PLOT 2: MASS FLOW vs OUTDOOR TEMP ---
plt.figure(figsize=(10, 6))
plt.scatter(plot_df['Outdoor_Temp'], plot_df['Calculated_Flow'], 
            alpha=0.5, color='#3498db', label='Calculated Mass Flow')

plt.title('Heating Water Mass Flow vs. Outdoor Temperature', fontsize=14, fontweight='bold')
plt.xlabel('Outdoor Temperature (°C)')
plt.ylabel('Mass Flow Rate (kg/s)')
plt.grid(True, alpha=0.3)

max_flow = plot_df['Calculated_Flow'].max()
avg_flow = plot_df['Calculated_Flow'].mean()
avg_delta_t = plot_df['Delta_T'].mean()

plt.axhline(max_flow, color='red', linestyle='--', label=f'Max Flow: {max_flow:.2f} kg/s')
plt.axhline(avg_flow, color='green', linestyle='-', label=f'Avg Flow: {avg_flow:.2f} kg/s')
plt.legend()
plt.savefig(output_flow, dpi=300)
plt.show()

# --- PRINT TEXT REPORT ---
print("\n" + "="*50)
print("DISTRICT HEATING HYDRAULIC ANALYSIS")
print("="*50)
print(f"1. SUPPLY TEMP CHECK:")
print(f"   Set at -20°C Outdoor: 60.0°C")
print(f"   Actual Max Supply T:  {plot_df[sup_temp_col].max():.2f}°C")
print(f"   Status: {'OK' if abs(plot_df[sup_temp_col].max() - 60) < 2 else 'Check Curve'}")

print(f"\n2. MASS FLOW CHECK (The 'Acceptable Range'):")
print(f"   Observed Max Flow:    {max_flow:.3f} kg/s")
print(f"   Observed Avg Flow:    {avg_flow:.3f} kg/s")
print(f"   *Design Note*: Your pump and pipes should be sized for {max_flow:.3f} kg/s.")

print(f"\n3. DELTA-T CHECK (Efficiency):")
print(f"   Average Delta-T:      {avg_delta_t:.1f}°C")
print(f"   Max Delta-T:          {plot_df['Delta_T'].max():.1f}°C")
print(f"   *Interpretation*: ")
print(f"   - >30°C: Excellent Efficiency")
print(f"   - 20-30°C: Good/Standard")
print(f"   - <15°C: Poor (High flow requirement)")

print(f"\n4. RETURN TEMP CHECK:")
print(f"   Average Return Temp:  {plot_df[ret_temp_col].mean():.1f}°C")
print(f"   Max Return Temp:      {plot_df[ret_temp_col].max():.1f}°C")
print("="*50)
