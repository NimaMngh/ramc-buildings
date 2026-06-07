# -*- coding: utf-8 -*-
"""
Energy breakdown analysis for January 2024.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mpl_dates
import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all file paths relative to script location
file_path = os.path.join(SCRIPT_DIR, 'ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.csv')
output_figure = os.path.join(SCRIPT_DIR, 'Energy_Breakdown.png')

# Load data
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Custom date parsing function
def parse_date_with_year(date_str, base_year=2023):
    """Parse date string and add year, handling the Oct-Apr heating season"""
    try:
        date_str = date_str.strip()
        parsed_date = pd.to_datetime(date_str, format='%m/%d  %H:%M:%S')
        
        if parsed_date.month >= 10:  # Oct, Nov, Dec
            year = base_year
        else:  # Jan, Feb, Mar, Apr
            year = base_year + 1
            
        return parsed_date.replace(year=year)
    except:
        return pd.NaT

# Parse dates
print("Parsing dates...")
df['Date/Time'] = df['Date/Time'].apply(parse_date_with_year)

# Define occupancy schedule
def get_occupancy_status(datetime_obj):
    """
    Determine if building is occupied based on retail schedule
    Weekdays: 7:00-21:00, Saturdays: 7:00-22:00, Sundays: 9:00-19:00
    """
    hour = datetime_obj.hour
    weekday = datetime_obj.weekday()  # 0=Monday, 6=Sunday
    
    if weekday < 5:  # Monday-Friday
        return 7 <= hour < 21
    elif weekday == 5:  # Saturday
        return 7 <= hour < 22
    else:  # Sunday
        return 9 <= hour < 19

# Add occupancy status
df['Occupied'] = df['Date/Time'].apply(get_occupancy_status)

# Define energy columns (check which ones exist in your data)
energy_columns = {
    'Facility': 'Electricity:Facility [J](Hourly)',
    'Fans': 'Fans:Electricity [J](Hourly)', 
    'Interior_Lights': 'InteriorLights:Electricity [J](Hourly)',
    'Heating': 'Heating:Electricity [J](Hourly)',  # if exists
    'Cooling': 'Cooling:Electricity [J](Hourly)'   # if exists
}

# Check which energy columns exist
existing_energy_cols = {}
for key, col_name in energy_columns.items():
    if col_name in df.columns:
        existing_energy_cols[key] = col_name
        print(f"Found energy column: {col_name}")

if len(existing_energy_cols) == 0:
    print("No energy columns found! Please check column names.")
    print("Available columns containing 'Electricity':")
    for col in df.columns:
        if 'Electricity' in col or 'Electric' in col:
            print(f"  - {col}")
else:
    # Convert J to kWh (1 J = 1/3,600,000 kWh)
    for key, col_name in existing_energy_cols.items():
        df[f'{key}_Electricity_kWh'] = df[col_name] / 3600000

    # Calculate other electricity if we have facility total
    if 'Facility' in existing_energy_cols:
        other_components = []
        for key in ['Fans', 'Interior_Lights', 'Heating', 'Cooling']:
            if key in existing_energy_cols:
                other_components.append(f'{key}_Electricity_kWh')
        
        if other_components:
            df['Other_Electricity_kWh'] = df['Facility_Electricity_kWh'] - df[other_components].sum(axis=1)
        else:
            df['Other_Electricity_kWh'] = df['Facility_Electricity_kWh']

    # Filter to Full January 2024
    january_start = pd.to_datetime('2024-01-01')
    january_end = pd.to_datetime('2024-01-31 23:59:59')
    january_df = df[(df['Date/Time'] >= january_start) & 
                   (df['Date/Time'] <= january_end)].copy()

    print(f"\nFull January 2024 energy data shape: {january_df.shape}")

    # Color palette (high-contrast, colorblind-friendly)
    colors = {
        'Facility': '#E74C3C',      # Red
        'Fans': '#F39C12',          # Orange  
        'Interior_Lights': '#F1C40F', # Yellow
        'Heating': '#8E44AD',       # Purple
        'Cooling': '#3498DB',       # Blue
        'Other': '#27AE60',         # Green
        'Occupied': '#E74C3C',      # Red
        'Unoccupied': '#95A5A6',    # Gray
        'Weekdays': '#E74C3C',      # Red
        'Weekends': '#3498DB'       # Blue
    }

    fig = plt.figure(figsize=(20, 16))

    # Plot 1: Stacked Bar Chart - Weekly Energy Breakdown
    ax1 = plt.subplot(3, 2, 1)
    
    # Prepare data for stacking - sample every 24 hours for readability
    sample_data = january_df[::24].copy()  # Every 24th hour for monthly view
    
    # Build the stacked components
    stack_data = []
    stack_labels = []
    stack_colors = []
    
    component_order = ['Fans', 'Interior_Lights', 'Heating', 'Cooling', 'Other']  # Logical order
    
    for component in component_order:
        col_name = f'{component}_Electricity_kWh'
        if col_name in sample_data.columns:
            stack_data.append(sample_data[col_name].values)
            stack_labels.append(component.replace('_', ' '))
            stack_colors.append(colors.get(component, '#95A5A6'))
    
    if stack_data:
        bottom = np.zeros(len(sample_data))
        for i, (data, label, color) in enumerate(zip(stack_data, stack_labels, stack_colors)):
            plt.bar(range(len(sample_data)), data, bottom=bottom, 
                   label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
            bottom += data
    
    plt.title('Daily Energy Use Breakdown - January 2024', fontsize=14, fontweight='bold')
    plt.xlabel('Day of Month')
    plt.ylabel('Energy (kWh)')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis labels to show dates
    date_labels = [d.strftime('%m/%d') for d in sample_data['Date/Time']]
    plt.xticks(range(0, len(date_labels), 3), date_labels[::3], rotation=45)

    # Plot 2: Hourly Energy Profile (Average by Hour of Day)
    ax2 = plt.subplot(3, 2, 2)
    
    hourly_profiles = {}
    for key in existing_energy_cols.keys():
        col_name = f'{key}_Electricity_kWh'
        if col_name in january_df.columns:
            hourly_avg = january_df.groupby(january_df['Date/Time'].dt.hour)[col_name].mean()
            hourly_profiles[key.replace('_', ' ')] = hourly_avg
    
    for component, profile in hourly_profiles.items():
        color = colors.get(component.replace(' ', '_'), '#95A5A6')
        plt.plot(profile.index, profile.values, 'o-', 
                label=component, color=color, linewidth=3, markersize=6, alpha=0.8)
    
    plt.title('Average Hourly Energy Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Energy (kWh)')
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))

    # Plot 3: Occupied vs Unoccupied Energy Comparison
    ax3 = plt.subplot(3, 2, 3)
    
    occupied_data = january_df[january_df['Occupied'] == True]
    unoccupied_data = january_df[january_df['Occupied'] == False]
    
    components = []
    occupied_values = []
    unoccupied_values = []
    
    for key in existing_energy_cols.keys():
        col_name = f'{key}_Electricity_kWh'
        if col_name in january_df.columns:
            components.append(key.replace('_', ' '))
            occupied_values.append(occupied_data[col_name].mean())
            unoccupied_values.append(unoccupied_data[col_name].mean())
    
    x = np.arange(len(components))
    width = 0.35
    
    plt.bar(x - width/2, occupied_values, width, label='Occupied', 
           color=colors['Occupied'], alpha=0.8, edgecolor='white', linewidth=0.5)
    plt.bar(x + width/2, unoccupied_values, width, label='Unoccupied', 
           color=colors['Unoccupied'], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    plt.title('Occupied vs Unoccupied Energy Consumption', fontsize=14, fontweight='bold')
    plt.xlabel('Energy Component')
    plt.ylabel('Average Energy (kWh/hour)')
    plt.xticks(x, components, rotation=45)
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # Plot 4: Daily Energy Totals
    ax4 = plt.subplot(3, 2, 4)
    
    if 'Facility' in existing_energy_cols:
        daily_totals = january_df.groupby(january_df['Date/Time'].dt.date)['Facility_Electricity_kWh'].sum()
        
        plt.plot(daily_totals.index, daily_totals.values, 'o-', 
                color=colors['Facility'], linewidth=3, markersize=8, alpha=0.8)
        plt.title('Daily Total Energy Consumption', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Daily Energy (kWh)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add average line
        avg_daily = daily_totals.mean()
        plt.axhline(avg_daily, color='#34495E', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Average: {avg_daily:.1f} kWh/day')
        plt.legend(framealpha=0.9)

    # Plot 5: Energy Intensity by Occupancy
    ax5 = plt.subplot(3, 2, 5)
    
    if 'Facility' in existing_energy_cols:
        # Calculate energy per occupied hour
        daily_occupied_hours = january_df.groupby(january_df['Date/Time'].dt.date)['Occupied'].sum()
        daily_energy = january_df.groupby(january_df['Date/Time'].dt.date)['Facility_Electricity_kWh'].sum()
        
        plt.scatter(daily_occupied_hours, daily_energy, alpha=0.8, 
                   color=colors['Other'], s=80, edgecolors='white', linewidth=1)
        plt.title('Energy vs Occupied Hours', fontsize=14, fontweight='bold')
        plt.xlabel('Occupied Hours per Day')
        plt.ylabel('Daily Energy (kWh)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(daily_occupied_hours) > 1:
            z = np.polyfit(daily_occupied_hours, daily_energy, 1)
            p = np.poly1d(z)
            plt.plot(daily_occupied_hours, p(daily_occupied_hours), 
                    color='#34495E', linestyle='--', alpha=0.8, linewidth=2)

    # Plot 6: Weekend vs Weekday Energy Pattern
    ax6 = plt.subplot(3, 2, 6)
    
    january_df['Weekday'] = january_df['Date/Time'].dt.weekday
    january_df['Is_Weekend'] = january_df['Weekday'].isin([5, 6])  # Saturday, Sunday
    
    if 'Facility' in existing_energy_cols:
        weekday_profile = january_df[~january_df['Is_Weekend']].groupby(
            january_df['Date/Time'].dt.hour)['Facility_Electricity_kWh'].mean()
        weekend_profile = january_df[january_df['Is_Weekend']].groupby(
            january_df['Date/Time'].dt.hour)['Facility_Electricity_kWh'].mean()
        
        plt.plot(weekday_profile.index, weekday_profile.values, 'o-', 
                label='Weekdays', color=colors['Weekdays'], linewidth=3, markersize=6)
        plt.plot(weekend_profile.index, weekend_profile.values, 's-', 
                label='Weekends', color=colors['Weekends'], linewidth=3, markersize=6)
        
        plt.title('Weekday vs Weekend Energy Profile', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Energy (kWh)')
        plt.legend(framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(output_figure, dpi=600, bbox_inches='tight')
    plt.show()

    # Print color legend for reference
    print(f"\n{'='*50}")
    print("COLOR LEGEND")
    print(f"{'='*50}")
    for component, color in colors.items():
        print(f"{component:<15}: {color}")
