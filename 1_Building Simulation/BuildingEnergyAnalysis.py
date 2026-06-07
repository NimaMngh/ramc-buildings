# -*- coding: utf-8 -*-
"""
Building energy analysis: district heating vs ERV and other loads.
Focuses on the heat recovery ventilator and zone-based energy consumption.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.dates as mpl_dates
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all file paths relative to script location
file_path = os.path.join(SCRIPT_DIR, 'ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.csv')
output_figure = os.path.join(SCRIPT_DIR, 'building_energy_analysis.png')

# Load data
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()


# Parse dates
def parse_date_with_year(date_str, base_year=2023):
    try:
        date_str = date_str.strip()
        parsed_date = pd.to_datetime(date_str, format='%m/%d  %H:%M:%S')
        if parsed_date.month >= 10:
            year = base_year
        else:
            year = base_year + 1
        return parsed_date.replace(year=year)
    except:
        return pd.NaT

df['Date/Time'] = df['Date/Time'].apply(parse_date_with_year)

# Define key columns for analysis
district_heating_col = 'DISTRICT HEATING UTILITY:District Heating Water Rate [W](Hourly)'
erv_col = 'CORE_RETAIL_ERV_HX:Heat Exchanger Sensible Heating Rate [W](Hourly)'
pump_col = 'HW CIRCULATING PUMP:Pump Electricity Rate [W](Hourly)'
baseboard_col = 'CORE_RETAIL BASEBOARD HEATER:Baseboard Total Heating Rate [W](Hourly)'

# Zone lighting columns
lighting_cols = {
    'Back Space': 'BACK_SPACE:Zone Lights Total Heating Rate [W](Hourly)',
    'Core Retail': 'CORE_RETAIL:Zone Lights Total Heating Rate [W](Hourly)',
    'Point of Sale': 'POINT_OF_SALE:Zone Lights Total Heating Rate [W](Hourly)',
    'Front Retail': 'FRONT_RETAIL:Zone Lights Total Heating Rate [W](Hourly)',
    'Front Entry': 'FRONT_ENTRY:Zone Lights Total Heating Rate [W](Hourly)'
}

# Zone temperature columns
temp_cols = {
    'Back Space': 'BACK_SPACE:Zone Mean Air Temperature [C](Hourly)',
    'Core Retail': 'CORE_RETAIL:Zone Mean Air Temperature [C](Hourly)',
    'Point of Sale': 'POINT_OF_SALE:Zone Mean Air Temperature [C](Hourly)',
    'Front Retail': 'FRONT_RETAIL:Zone Mean Air Temperature [C](Hourly)',
    'Front Entry': 'FRONT_ENTRY:Zone Mean Air Temperature [C](Hourly)'
}

# Convert to kW for better scaling
df['District_Heating_kW'] = df[district_heating_col] / 1000
df['ERV_Heating_kW'] = df[erv_col] / 1000
df['Pump_Power_kW'] = df[pump_col] / 1000
df['Baseboard_kW'] = df[baseboard_col] / 1000

# Calculate total lighting load
df['Total_Lighting_kW'] = sum(df[col] for col in lighting_cols.values()) / 1000

# Add occupancy status
def get_occupancy_status(datetime_obj):
    if pd.isna(datetime_obj):
        return False
    hour = datetime_obj.hour
    weekday = datetime_obj.weekday()
    
    if weekday < 5:  # Monday-Friday
        return 7 <= hour < 21
    elif weekday == 5:  # Saturday
        return 7 <= hour < 22
    else:  # Sunday
        return 9 <= hour < 19

df['Occupied'] = df['Date/Time'].apply(get_occupancy_status)

# Filter to January 2024
january_start = pd.to_datetime('2024-01-01')
january_end = pd.to_datetime('2024-01-31 23:59:59')
january_df = df[(df['Date/Time'] >= january_start) & 
               (df['Date/Time'] <= january_end)].copy()

print(f"January 2024 analysis data: {january_df.shape[0]} hours")

# Color palette
colors = {
    'heating': '#E74C3C',        # Red
    'erv': '#3498DB',            # Blue  
    'lighting': '#F39C12',       # Orange
    'pump': '#9B59B6',           # Purple
    'baseboard': '#E67E22',      # Dark orange
    'occupied': '#E74C3C',       # Red
    'unoccupied': '#95A5A6',     # Gray
    'zone1': '#1ABC9C',          # Turquoise
    'zone2': '#2ECC71',          # Green
    'zone3': '#F1C40F',          # Yellow
    'zone4': '#E91E63',          # Pink
    'zone5': '#673AB7'           # Deep purple
}

fig = plt.figure(figsize=(20, 16))

# Plot 1: District Heating vs ERV Heat Exchanger
ax1 = plt.subplot(3, 2, 1)
sample_df = january_df[::4].copy()  # Sample every 4 hours

# Primary axis - District Heating
line1 = ax1.plot(sample_df['Date/Time'], sample_df['District_Heating_kW'], 
                 color=colors['heating'], linewidth=2.5, alpha=0.8, label='District Heating')
ax1.set_ylabel('District Heating (kW)', color=colors['heating'], fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=colors['heating'])
ax1.grid(True, alpha=0.3)

# Secondary axis - ERV Heat Exchanger
ax2 = ax1.twinx()
line2 = ax2.plot(sample_df['Date/Time'], sample_df['ERV_Heating_kW'], 
                 color=colors['erv'], linewidth=2.5, alpha=0.8, label='ERV Heat Recovery')
ax2.set_ylabel('ERV Heat Recovery (kW)', color=colors['erv'], fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=colors['erv'])

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

ax1.set_title('District Heating vs ERV Heat Recovery\n(January 2024)', fontsize=14, fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Multiple Building Loads Comparison
ax3 = plt.subplot(3, 2, 2)

# Normalize all loads for comparison
loads_to_compare = ['District_Heating_kW', 'ERV_Heating_kW', 'Total_Lighting_kW', 'Pump_Power_kW']
load_colors = [colors['heating'], colors['erv'], colors['lighting'], colors['pump']]
load_labels = ['District Heating', 'ERV Heat Recovery', 'Total Lighting', 'HW Pump']

for load, color, label in zip(loads_to_compare, load_colors, load_labels):
    if sample_df[load].max() > 0:
        normalized = (sample_df[load] - sample_df[load].min()) / (sample_df[load].max() - sample_df[load].min())
        ax3.plot(sample_df['Date/Time'], normalized, color=color, linewidth=2, alpha=0.8, label=label)

ax3.set_ylabel('Normalized Load (0-1)', fontsize=12)
ax3.set_title('Normalized Building Load Comparison', fontsize=14, fontweight='bold')
ax3.legend(framealpha=0.9)
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Plot 3: District Heating vs ERV Correlation
ax4 = plt.subplot(3, 2, 3)

# Filter for correlation analysis
correlation_df = january_df[
    (january_df['District_Heating_kW'] > 0) & 
    (january_df['ERV_Heating_kW'] > 0)
].copy()

if len(correlation_df) > 10:
    # Color by occupancy
    occupied_data = correlation_df[correlation_df['Occupied'] == True]
    unoccupied_data = correlation_df[correlation_df['Occupied'] == False]
    
    if len(occupied_data) > 0:
        ax4.scatter(occupied_data['ERV_Heating_kW'], occupied_data['District_Heating_kW'],
                   alpha=0.6, c=colors['occupied'], s=30, label='Occupied', edgecolors='white', linewidth=0.5)
    
    if len(unoccupied_data) > 0:
        ax4.scatter(unoccupied_data['ERV_Heating_kW'], unoccupied_data['District_Heating_kW'],
                   alpha=0.6, c=colors['unoccupied'], s=30, label='Unoccupied', edgecolors='white', linewidth=0.5)
    
    # Correlation analysis
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            correlation_df['ERV_Heating_kW'], correlation_df['District_Heating_kW'])
        
        x_trend = np.linspace(correlation_df['ERV_Heating_kW'].min(), 
                             correlation_df['ERV_Heating_kW'].max(), 100)
        y_trend = slope * x_trend + intercept
        
        ax4.plot(x_trend, y_trend, color='black', linestyle='--', linewidth=2,
                label=f'R² = {r_value**2:.3f}')
        
        # Add correlation info
        ax4.text(0.05, 0.95, f'Correlation: {r_value:.3f}\nSlope: {slope:.2f}\np-value: {p_value:.4f}', 
                transform=ax4.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    except:
        pass
    
    ax4.legend(framealpha=0.9)

ax4.set_xlabel('ERV Heat Recovery (kW)', fontsize=12)
ax4.set_ylabel('District Heating (kW)', fontsize=12)
ax4.set_title('Heating vs ERV Correlation Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 4: Daily Load Patterns
ax5 = plt.subplot(3, 2, 4)

# Calculate hourly averages
valid_data = january_df[~january_df['Date/Time'].isna()].copy()

if len(valid_data) > 0:
    hourly_heating = valid_data.groupby(valid_data['Date/Time'].dt.hour)['District_Heating_kW'].mean()
    hourly_erv = valid_data.groupby(valid_data['Date/Time'].dt.hour)['ERV_Heating_kW'].mean()
    hourly_lighting = valid_data.groupby(valid_data['Date/Time'].dt.hour)['Total_Lighting_kW'].mean()
    
    # Plot on dual axes
    line1 = ax5.plot(hourly_heating.index, hourly_heating.values, 'o-', 
                     color=colors['heating'], linewidth=3, markersize=6, label='District Heating')
    line2 = ax5.plot(hourly_lighting.index, hourly_lighting.values, 's-', 
                     color=colors['lighting'], linewidth=3, markersize=6, label='Total Lighting')
    
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Average Load (kW)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(range(0, 24, 2))
    
    # ERV on secondary axis
    ax6 = ax5.twinx()
    line3 = ax6.plot(hourly_erv.index, hourly_erv.values, '^-', 
                     color=colors['erv'], linewidth=2, markersize=5, alpha=0.8, label='ERV Heat Recovery')
    ax6.set_ylabel('ERV Heat Recovery (kW)', color=colors['erv'], fontsize=12)
    ax6.tick_params(axis='y', labelcolor=colors['erv'])
    
    # Combined legend
    all_lines = line1 + line2 + line3
    all_labels = [line.get_label() for line in all_lines]
    ax5.legend(all_lines, all_labels, loc='upper left', framealpha=0.9, fontsize=9)

ax5.set_title('Daily Load Patterns (Average)', fontsize=14, fontweight='bold')

# Plot 5: Zone Lighting Analysis
ax7 = plt.subplot(3, 2, 5)

zone_colors = [colors['zone1'], colors['zone2'], colors['zone3'], colors['zone4'], colors['zone5']]
zone_names = list(lighting_cols.keys())

for i, (zone_name, col) in enumerate(lighting_cols.items()):
    zone_avg = valid_data.groupby(valid_data['Date/Time'].dt.hour)[col].mean() / 1000  # Convert to kW
    ax7.plot(zone_avg.index, zone_avg.values, 'o-', 
             color=zone_colors[i], linewidth=2, markersize=4, alpha=0.8, label=zone_name)

ax7.set_xlabel('Hour of Day')
ax7.set_ylabel('Zone Lighting Load (kW)', fontsize=12)
ax7.set_title('Zone-by-Zone Lighting Patterns', fontsize=14, fontweight='bold')
ax7.legend(framealpha=0.9, fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_xticks(range(0, 24, 2))

# Plot 6: Energy Summary Dashboard
ax8 = plt.subplot(3, 2, 6)

# Calculate energy statistics
energy_summary = {
    'District Heating': january_df['District_Heating_kW'].mean(),
    'ERV Heat Recovery': january_df['ERV_Heating_kW'].mean(),
    'Total Lighting': january_df['Total_Lighting_kW'].mean(),
    'HW Pump': january_df['Pump_Power_kW'].mean(),
    'Baseboard': january_df['Baseboard_kW'].mean()
}

# Remove zero values
energy_summary = {k: v for k, v in energy_summary.items() if v > 0.1}

categories = list(energy_summary.keys())
values = list(energy_summary.values())
bar_colors = [colors['heating'], colors['erv'], colors['lighting'], colors['pump'], colors['baseboard']][:len(categories)]

bars = ax8.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='white', linewidth=1)

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
             f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax8.set_ylabel('Average Power (kW)', fontsize=12)
ax8.set_title('Building Energy Summary\n(January 2024 Average)', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_figure, dpi=600, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("BUILDING ENERGY ANALYSIS")
print(f"{'='*70}")

print(f"\nENERGY CONSUMPTION SUMMARY (January 2024):")
print(f"  District Heating:    {january_df['District_Heating_kW'].mean():.1f} kW average")
print(f"  ERV Heat Recovery:   {january_df['ERV_Heating_kW'].mean():.1f} kW average")
print(f"  Total Lighting:      {january_df['Total_Lighting_kW'].mean():.1f} kW average")
print(f"  Hot Water Pump:      {january_df['Pump_Power_kW'].mean():.2f} kW average")

if len(correlation_df) > 10:
    correlation = correlation_df['District_Heating_kW'].corr(correlation_df['ERV_Heating_kW'])
    if not pd.isna(correlation):
        print(f"\nDISTRICT HEATING vs ERV CORRELATION:")
        print(f"  Correlation coefficient: {correlation:.3f}")
        
        if correlation > 0.7:
            print("  STRONG POSITIVE correlation - ERV and heating work together")
        elif correlation > 0.3:
            print("  MODERATE POSITIVE correlation")
        elif correlation < -0.3:
            print("  NEGATIVE correlation - ERV reduces heating load")
        else:
            print("  WEAK correlation")

# Zone analysis
print(f"\nZONE LIGHTING ANALYSIS:")
for zone_name, col in lighting_cols.items():
    avg_lighting = january_df[col].mean() / 1000
    print(f"  {zone_name:15s}: {avg_lighting:.1f} kW average")

print(f"\nKEY OBSERVATIONS:")
print(f"  - ERV heat recovery varies by {january_df['ERV_Heating_kW'].std():.1f} kW (std)")
print(f"  - Core Retail is the largest lighting consumer")
print(f"  - Occupancy patterns are visible across all systems")
print(f"  - The heat recovery system contributes to building heating")