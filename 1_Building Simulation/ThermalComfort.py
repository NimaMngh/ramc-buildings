# -*- coding: utf-8 -*-
"""
Thermal comfort analysis with occupancy schedule integration for January 2024.
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
output_figure = os.path.join(SCRIPT_DIR, 'thermal_comfort_analysis.png')

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

# Define occupancy schedule based on BLDG_OCC_SCH
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

# Add occupancy status to dataframe
df['Occupied'] = df['Date/Time'].apply(get_occupancy_status)

# Define PPD columns
ppd_cols = [
    'BACK_SPACE:Zone Thermal Comfort Fanger Model PPD [%](Hourly)',
    'CORE_RETAIL:Zone Thermal Comfort Fanger Model PPD [%](Hourly)',
    'POINT_OF_SALE:Zone Thermal Comfort Fanger Model PPD [%](Hourly)',
    'FRONT_RETAIL:Zone Thermal Comfort Fanger Model PPD [%](Hourly)',
    'FRONT_ENTRY:Zone Thermal Comfort Fanger Model PPD [%](Hourly)'
]

# Filter to only columns that exist in the data
existing_ppd_cols = [col for col in ppd_cols if col in df.columns]
print(f"\nFound {len(existing_ppd_cols)} PPD columns in data")

if existing_ppd_cols:
    # Filter to FULL JANUARY 2024
    january_start = pd.to_datetime('2024-01-01')
    january_end = pd.to_datetime('2024-01-31 23:59:59')
    january_df = df[(df['Date/Time'] >= january_start) & 
                   (df['Date/Time'] <= january_end)].copy()

    print(f"\nFull January 2024 data shape: {january_df.shape}")
    
    # ASHRAE 55 Comfort Thresholds Analysis
    print("\n" + "="*70)
    print("ASHRAE 55 THERMAL COMFORT VIOLATION ANALYSIS")
    print("FULL JANUARY 2024 MONTHLY REPORT")
    print("="*70)
    
    # Define comfort categories based on ASHRAE 55
    comfort_thresholds = {
        'Excellent': 10,    # PPD ≤ 10%
        'Acceptable': 20,   # PPD ≤ 20% (ASHRAE 55 limit)
        'Poor': float('inf')  # PPD > 20%
    }
    
    # Initialize summary data for monthly analysis
    monthly_summary = {}
    
    for col in existing_ppd_cols:
        zone_name = col.split(':')[0]
        print(f"\n{'='*50}")
        print(f"ZONE: {zone_name}")
        print(f"{'='*50}")
        
        # Filter to occupied hours only
        occupied_data = january_df[january_df['Occupied'] == True]
        zone_ppd = occupied_data[col]
        
        if len(occupied_data) == 0:
            print("No occupied hours found in this period")
            continue
            
        print(f"Total occupied hours analyzed: {len(occupied_data)}")
        
        # Find violations of ASHRAE 55 (PPD > 20%)
        violations_20 = occupied_data[occupied_data[col] > 20]
        violations_10 = occupied_data[occupied_data[col] > 10]
        
        compliance_rate = ((len(occupied_data) - len(violations_20))/len(occupied_data)*100)
        
        print(f"\nASHRAE 55 COMPLIANCE (PPD ≤ 20%):")
        print(f"  Violation hours: {len(violations_20)}")
        print(f"  Compliance rate: {compliance_rate:.1f}%")
        
        # Weekly breakdown
        print(f"\nWEEKLY BREAKDOWN:")
        for week in range(1, 6):  # 5 weeks in January
            week_start = january_start + pd.Timedelta(days=(week-1)*7)
            week_end = min(week_start + pd.Timedelta(days=6), january_end)
            
            week_data = occupied_data[(occupied_data['Date/Time'] >= week_start) & 
                                    (occupied_data['Date/Time'] <= week_end)]
            
            if len(week_data) > 0:
                week_violations = len(week_data[week_data[col] > 20])
                week_compliance = ((len(week_data) - week_violations)/len(week_data)*100)
                print(f"  Week {week} ({week_start.strftime('%m/%d')}-{week_end.strftime('%m/%d')}): "
                      f"{week_compliance:.1f}% compliance ({week_violations} violations)")
        
        # Daily pattern analysis
        print(f"\nDAILY PATTERN ANALYSIS:")
        daily_violations = violations_20.groupby(violations_20['Date/Time'].dt.hour).size()
        if len(daily_violations) > 0:
            print(f"  Most problematic hours:")
            for hour in daily_violations.nlargest(5).index:
                count = daily_violations[hour]
                print(f"    {hour:02d}:00 - {count} violations")
        
        print(f"\nCOMFORT QUALITY ANALYSIS (PPD ≤ 10%):")
        print(f"  Hours with PPD > 10%: {len(violations_10)}")
        print(f"  High comfort rate: {((len(occupied_data) - len(violations_10))/len(occupied_data)*100):.1f}%")
        
        # Statistical summary
        print(f"\nSTATISTICAL SUMMARY (Occupied Hours Only):")
        print(f"  Mean PPD: {zone_ppd.mean():.2f}%")
        print(f"  Median PPD: {zone_ppd.median():.2f}%")
        print(f"  Max PPD: {zone_ppd.max():.2f}%")
        print(f"  Min PPD: {zone_ppd.min():.2f}%")
        print(f"  Std Dev: {zone_ppd.std():.2f}%")
        print(f"  95th Percentile: {zone_ppd.quantile(0.95):.2f}%")
        print(f"  5th Percentile: {zone_ppd.quantile(0.05):.2f}%")
        
        # Store summary data
        monthly_summary[zone_name] = {
            'total_hours': len(occupied_data),
            'violations': len(violations_20),
            'compliance_rate': compliance_rate,
            'mean_ppd': zone_ppd.mean(),
            'max_ppd': zone_ppd.max(),
            'std_ppd': zone_ppd.std()
        }

    # Build the figure
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    
    # Plot 1: Full month PPD trends
    ax1 = plt.subplot(3, 1, 1)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, col in enumerate(existing_ppd_cols):
        zone = col.split(':')[0]
        color = colors[i % len(colors)]
        
        # Plot occupied hours only for clarity
        occupied_data = january_df[january_df['Occupied'] == True]
        plt.plot(occupied_data['Date/Time'], occupied_data[col], 
                label=f'{zone}', color=color, alpha=0.7, linewidth=1.5)

    plt.axhline(10, color='green', linestyle='--', alpha=0.7, 
                label='Excellent Comfort (10%)')
    plt.axhline(20, color='red', linestyle='--', alpha=0.7, 
                label='ASHRAE 55 Limit (20%)')
    
    plt.title('Thermal Comfort PPD - Full January 2024 (Occupied Hours Only)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('PPD [%]', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Weekly compliance rates
    ax2 = plt.subplot(3, 2, 3)
    zones = list(monthly_summary.keys())
    compliance_rates = [monthly_summary[zone]['compliance_rate'] for zone in zones]
    
    bars = plt.bar(zones, compliance_rates, color=colors[:len(zones)], alpha=0.7)
    plt.axhline(80, color='red', linestyle='--', alpha=0.7, label='Target 80%')
    plt.title('Monthly Compliance Rates by Zone', fontweight='bold')
    plt.ylabel('Compliance Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, compliance_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: PPD distribution box plot
    ax3 = plt.subplot(3, 2, 4)
    ppd_data = []
    zone_labels = []
    
    for col in existing_ppd_cols:
        zone = col.split(':')[0]
        occupied_data = january_df[january_df['Occupied'] == True]
        ppd_data.append(occupied_data[col].dropna())
        zone_labels.append(zone)
    
    plt.boxplot(ppd_data, labels=zone_labels)
    plt.axhline(20, color='red', linestyle='--', alpha=0.7, label='ASHRAE 55 Limit')
    plt.title('PPD Distribution by Zone', fontweight='bold')
    plt.ylabel('PPD [%]')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Hourly violation patterns
    ax4 = plt.subplot(3, 2, 5)
    hourly_violations = {}
    
    for col in existing_ppd_cols:
        zone = col.split(':')[0]
        occupied_data = january_df[january_df['Occupied'] == True]
        violations = occupied_data[occupied_data[col] > 20]
        hourly_count = violations.groupby(violations['Date/Time'].dt.hour).size()
        hourly_violations[zone] = hourly_count
    
    for i, (zone, data) in enumerate(hourly_violations.items()):
        if len(data) > 0:
            plt.plot(data.index, data.values, 'o-', label=zone, 
                    color=colors[i], alpha=0.7, linewidth=2)
    
    plt.title('Hourly Violation Patterns', fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Daily violation trends
    ax5 = plt.subplot(3, 2, 6)
    for i, col in enumerate(existing_ppd_cols):
        zone = col.split(':')[0]
        occupied_data = january_df[january_df['Occupied'] == True]
        violations = occupied_data[occupied_data[col] > 20]
        daily_violations = violations.groupby(violations['Date/Time'].dt.date).size()
        
        if len(daily_violations) > 0:
            plt.plot(daily_violations.index, daily_violations.values, 'o-', 
                    label=zone, color=colors[i], alpha=0.7)
    
    plt.title('Daily Violation Trends', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Violations per Day')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_figure, dpi=600, bbox_inches='tight')
    plt.show()

    # Monthly summary report
    print(f"\n{'='*70}")
    print("MONTHLY COMPLIANCE SUMMARY - JANUARY 2024")
    print(f"{'='*70}")
    
    total_occupied_hours = len(january_df[january_df['Occupied'] == True])
    print(f"Total occupied hours in January: {total_occupied_hours}")
    
    print(f"\n{'Zone':<15} | {'Compliance':<10} | {'Violations':<10} | {'Mean PPD':<9} | {'Max PPD':<8} | {'Status'}")
    print("-" * 70)
    
    overall_violations = 0
    for col in existing_ppd_cols:
        zone_name = col.split(':')[0]
        data = monthly_summary[zone_name]
        
        status = "GOOD" if data['compliance_rate'] >= 80 else "POOR" if data['compliance_rate'] >= 60 else "CRITICAL"
        overall_violations += data['violations']
        
        print(f"{zone_name:<15} | {data['compliance_rate']:>8.1f}% | {data['violations']:>9} | "
              f"{data['mean_ppd']:>7.1f}% | {data['max_ppd']:>6.1f}% | {status}")
    
    overall_compliance = ((total_occupied_hours * len(existing_ppd_cols) - overall_violations) / 
                         (total_occupied_hours * len(existing_ppd_cols)) * 100)
    
    print("-" * 70)
    print(f"{'OVERALL':<15} | {overall_compliance:>8.1f}% | {overall_violations:>9} | {'':>9} | {'':>8} | "
          f"{'GOOD' if overall_compliance >= 80 else 'POOR' if overall_compliance >= 60 else 'CRITICAL'}")

else:
    print("No PPD columns found in the dataset!")
