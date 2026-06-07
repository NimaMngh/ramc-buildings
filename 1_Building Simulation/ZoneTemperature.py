import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mpl_dates
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all file paths relative to script location
file_path = os.path.join(SCRIPT_DIR, 'ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.csv')
output_january = os.path.join(SCRIPT_DIR, 'zone_temps_vs_setpoints_january.png')
output_weekly = os.path.join(SCRIPT_DIR, 'zone_temps_vs_setpoints.png')

# Define zones explicitly
zones = ['BACK_SPACE', 'CORE_RETAIL']

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

# Select relevant columns
temp_cols = [f'{zone}:Zone Mean Air Temperature [C](Hourly)' for zone in zones]
setpoint_cols = [f'{zone}:Zone Thermostat Heating Setpoint Temperature [C](Hourly)' for zone in zones]

# Filter data to January only
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-01-31 23:59:59')
january_df = df[(df['Date/Time'] >= start_date) & (df['Date/Time'] <= end_date)].copy()

print(f"January data shape: {january_df.shape}")
print(f"Date range: {january_df['Date/Time'].min()} to {january_df['Date/Time'].max()}")

# Create the plot
plt.figure(figsize=(16, 10))
colors = ['blue', 'red']

for i, (zone, temp_col, setpoint_col) in enumerate(zip(zones, temp_cols, setpoint_cols)):
    color = colors[i]
    
    # Plot temperature
    plt.plot(january_df['Date/Time'], january_df[temp_col], 
             label=f'{zone} Temperature', color=color, alpha=0.8, linewidth=1)
    
    # Plot setpoint
    plt.plot(january_df['Date/Time'], january_df[setpoint_col], 
             label=f'{zone} Heating Setpoint', color=color, linestyle='--', 
             alpha=0.9, linewidth=2)

plt.title('Zone Temperatures vs. Heating Setpoints - January 2024', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature [°C]', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)

# Format x-axis for better readability
plt.gca().xaxis.set_major_formatter(mpl_dates.DateFormatter('%m-%d'))  # MM-DD format
plt.gca().xaxis.set_major_locator(mpl_dates.DayLocator(interval=2))  # Every 2 days
plt.gca().xaxis.set_minor_locator(mpl_dates.DayLocator())  # Every day

# Rotate dates for better readability
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(output_january, dpi=600, bbox_inches='tight')
plt.show()

# Print January statistics
print("\n=== January 2024 Statistics ===")
for zone in zones:
    temp_col = f'{zone}:Zone Mean Air Temperature [C](Hourly)'
    setpoint_col = f'{zone}:Zone Thermostat Heating Setpoint Temperature [C](Hourly)'
    
    print(f"\n{zone}:")
    print(f"  Temperature - Mean: {january_df[temp_col].mean():.2f}°C, "
          f"Min: {january_df[temp_col].min():.2f}°C, "
          f"Max: {january_df[temp_col].max():.2f}°C")
    print(f"  Setpoint - Mean: {january_df[setpoint_col].mean():.2f}°C, "
          f"Min: {january_df[setpoint_col].min():.2f}°C, "
          f"Max: {january_df[setpoint_col].max():.2f}°C")

# Optional: Create a weekly view for even more detail
plt.figure(figsize=(16, 8))

# First week of January
first_week_start = pd.to_datetime('2024-01-01')
first_week_end = pd.to_datetime('2024-01-07 23:59:59')
first_week_df = january_df[(january_df['Date/Time'] >= first_week_start) & 
                          (january_df['Date/Time'] <= first_week_end)]

for i, (zone, temp_col, setpoint_col) in enumerate(zip(zones, temp_cols, setpoint_cols)):
    color = colors[i]
    
    plt.plot(first_week_df['Date/Time'], first_week_df[temp_col], 
             label=f'{zone} Temperature', color=color, alpha=0.8, linewidth=1.5)
    
    plt.plot(first_week_df['Date/Time'], first_week_df[setpoint_col], 
             label=f'{zone} Heating Setpoint', color=color, linestyle='--', 
             alpha=0.9, linewidth=2)

plt.title('Zone Temperatures vs. Heating Setpoints - First Week of January 2024', 
          fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature [°C]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Format x-axis to show days and hours
plt.gca().xaxis.set_major_formatter(mpl_dates.DateFormatter('%m/%d %H:%M'))
plt.gca().xaxis.set_major_locator(mpl_dates.HourLocator(interval=12))  # Every 12 hours
plt.gca().xaxis.set_minor_locator(mpl_dates.HourLocator(interval=6))   # Every 6 hours

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(output_weekly, dpi=600, bbox_inches='tight')
plt.show()
