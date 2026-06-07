# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 11:15:34 2025

@author: nmi03

EnergyPlus Retail Building Data Processor for Gray-Box RC Model Identification
Uses EnergyPlus 25.1 variable naming conventions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RetailBuildingDataProcessor:
    def __init__(self, csv_file_path, zone_volumes=None):
        """
        Initialize the data processor for retail building EnergyPlus output
        
        Parameters:
        -----------
        csv_file_path : str
            Path to the EnergyPlus CSV output file
        zone_volumes : dict, optional
            Exact zone volumes from .eio file. If None, will prompt user to extract them.
        """
        self.df = pd.read_csv(csv_file_path)
        self.zones = ['BACK_SPACE', 'CORE_RETAIL', 'POINT_OF_SALE', 'FRONT_RETAIL', 'FRONT_ENTRY']
        
        # Zone volumes - MUST be extracted from .eio file
        if zone_volumes is None:
            print("CRITICAL: Zone volumes not provided!")
            print("Extract exact volumes from your .eio file:")
            print("   1. Open ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.eio")
            print("   2. Search for 'Zone Summary' or 'Zone Internal Gains Nominal'")
            print("   3. Find the Volume [m3] column for each zone")
            print("   4. Update the zone_volumes parameter")
            print("\nUsing placeholder values for now - REPLACE THESE!")
            
            # Placeholder values - MUST BE REPLACED WITH ACTUAL VALUES
            self.zone_volumes = {
                'BACK_SPACE': 2317.33,     # FROM .eio
                'CORE_RETAIL': 9762.95,    # FROM .eio
                'POINT_OF_SALE': 919.94,   # FROM .eio
                'FRONT_RETAIL': 919.94,    # FROM .eio
                'FRONT_ENTRY': 73.20       # FROM .eio
            }
        else:
            self.zone_volumes = zone_volumes
            print("Using provided exact zone volumes")
        
        self.total_volume = sum(self.zone_volumes.values())
        
        # DO NOT define thermal capacities - these are parameters to be identified!
        
        print(f"Loaded data: {len(self.df)} timesteps")
        print(f"Date range: {self.df['Date/Time'].iloc[0]} to {self.df['Date/Time'].iloc[-1]}")
        print(f"Total building volume: {self.total_volume:.0f} m³")
        
        # Check if mass flow rate data is available
        self.has_flow_data = self._check_flow_data_availability()
    
    def _check_flow_data_availability(self):
        """Check if mass flow rate columns are available in the CSV"""
        # EnergyPlus 25.1 names this variable "Hot Water" (not just "Water")
        flow_cols = [f'{zone} BASEBOARD HEATER:Baseboard Hot Water Mass Flow Rate [kg/s](TimeStep)'
                    for zone in self.zones]
        
        available_cols = [col for col in flow_cols if col in self.df.columns]
        
        if len(available_cols) == len(self.zones):
            print("Mass flow rate data available - will use direct values")
            return True
        else:
            print("Mass flow rate data not available - will calculate from energy balance")
            print(f"   Found {len(available_cols)} of {len(self.zones)} expected columns")
            return False

    
    def clean_and_validate_data(self):
        """Clean the dataset and handle any missing values or anomalies"""
        print("\n=== DATA CLEANING ===")
        
        # Handle EnergyPlus datetime format with error handling
        try:
            # Clean and standardize the date format
            date_strings = self.df['Date/Time'].str.strip()
            # Remove extra spaces and add year
            cleaned_dates = date_strings.str.replace(r'\s+', ' ', regex=True)
            date_strings_with_year = '2024/' + cleaned_dates
            self.df['DateTime'] = pd.to_datetime(date_strings_with_year, format='%Y/%m/%d %H:%M:%S')
            print("Successfully parsed EnergyPlus datetime format")
        except Exception as e:
            print(f"Date parsing error: {e}")
            # Fallback: create sequential timestamps starting from Jan 1, 2024
            start_date = pd.Timestamp('2024-01-01 00:10:00')
            self.df['DateTime'] = pd.date_range(
                start=start_date, 
                periods=len(self.df), 
                freq='10min'
            )
            print("Using sequential 10-minute timestamps as fallback")
        
        # Check for missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values found:")
            print(missing_counts[missing_counts > 0])
        else:
            print("No missing values detected")
        
        # Check data ranges for physical reasonableness
        outdoor_temp = self.df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
        print(f"Outdoor temperature range: {outdoor_temp.min():.1f}°C to {outdoor_temp.max():.1f}°C")
        
        # Check zone temperatures
        for zone in self.zones:
            temp_col = f'{zone}:Zone Mean Air Temperature [C](TimeStep)'
            if temp_col in self.df.columns:
                temp_data = self.df[temp_col]
                print(f"{zone} temperature range: {temp_data.min():.1f}°C to {temp_data.max():.1f}°C")
        
        # Verify PRBS signal in supply temperature
        for zone in self.zones:
            supply_temp_col = f'{zone} BASEBOARD HEATER:Baseboard Water Inlet Temperature [C](TimeStep)'
            if supply_temp_col in self.df.columns:
                supply_temp = self.df[supply_temp_col]
                unique_temps = sorted(supply_temp.unique())
                if len(unique_temps) > 2:
                    print(f"PRBS signal detected in supply temperature: {len(unique_temps)} levels")
                    print(f"   Temperature levels: {unique_temps}")
                else:
                    print(f"Limited temperature variation detected: {unique_temps}")
                break
        
        return self.df
    
    def aggregate_zone_data(self):
        """Aggregate multi-zone data to building-level variables"""
        print("\n=== DATA AGGREGATION ===")
        
        # Initialize aggregated dataset
        agg_data = pd.DataFrame()
        agg_data['DateTime'] = self.df['DateTime']
        
        # 1. CONTROLLED VARIABLE: Volume-weighted average zone temperature
        zone_temp_weighted = []
        for zone in self.zones:
            temp_col = f'{zone}:Zone Mean Air Temperature [C](TimeStep)'
            if temp_col in self.df.columns:
                weighted_temp = self.df[temp_col] * self.zone_volumes[zone]
                zone_temp_weighted.append(weighted_temp)
            else:
                print(f"Warning: Temperature column not found for {zone}")
        
        if zone_temp_weighted:
            agg_data['T_air_avg'] = sum(zone_temp_weighted) / self.total_volume
            print(f"Volume-weighted air temperature calculated")
        else:
            print("Error: No zone temperature data found")
            return None
        
        # 2. MANIPULATED VARIABLES: Total heating system
        heating_rates = []
        for zone in self.zones:
            heating_col = f'{zone} BASEBOARD HEATER:Baseboard Total Heating Rate [W](TimeStep)'
            if heating_col in self.df.columns:
                heating_rates.append(self.df[heating_col])
            else:
                print(f"Warning: Heating rate column not found for {zone}")
                heating_rates.append(pd.Series([0] * len(self.df)))
        
        agg_data['Q_heating_total'] = sum(heating_rates)
        print(f"Total heating rate aggregated")
        
        # 3. Water temperatures and flow rates
        supply_temps = []
        return_temps = []
        flow_rates = []
        
        for zone in self.zones:
            supply_col = f'{zone} BASEBOARD HEATER:Baseboard Water Inlet Temperature [C](TimeStep)'
            return_col = f'{zone} BASEBOARD HEATER:Baseboard Water Outlet Temperature [C](TimeStep)'
            # EnergyPlus 25.1 names this variable "Hot Water"
            flow_col = f'{zone} BASEBOARD HEATER:Baseboard Hot Water Mass Flow Rate [kg/s](TimeStep)'
            
            if supply_col in self.df.columns:
                supply_temps.append(self.df[supply_col])
            else:
                print(f"Warning: Supply temperature column not found for {zone}")
                supply_temps.append(pd.Series([20] * len(self.df)))
            
            if return_col in self.df.columns:
                return_temps.append(self.df[return_col])
            else:
                print(f"Warning: Return temperature column not found for {zone}")
                return_temps.append(pd.Series([20] * len(self.df)))
            
            if self.has_flow_data and flow_col in self.df.columns:
                flow_rates.append(self.df[flow_col])
            else:
                flow_rates.append(pd.Series([0] * len(self.df)))
        
        if self.has_flow_data and flow_rates:
            # Use actual flow rates for flow-weighted averages
            mdot_total = sum(flow_rates)
            
            # Calculate flow-weighted average temperatures
            weighted_supply = sum(temp * flow for temp, flow in zip(supply_temps, flow_rates))
            weighted_return = sum(temp * flow for temp, flow in zip(return_temps, flow_rates))
            
            # Handle zero flow conditions
            agg_data['mdot_water_total'] = mdot_total
            agg_data['T_supply_avg'] = np.where(mdot_total > 1e-6, 
                                               weighted_supply / mdot_total,
                                               sum(supply_temps) / len(supply_temps))
            agg_data['T_return_avg'] = np.where(mdot_total > 1e-6,
                                               weighted_return / mdot_total, 
                                               sum(return_temps) / len(return_temps))
            print(f"Flow-weighted water temperatures calculated")
        else:
            # Fallback: simple averages and calculated flow rate
            agg_data['T_supply_avg'] = sum(supply_temps) / len(supply_temps)
            agg_data['T_return_avg'] = sum(return_temps) / len(return_temps)
            
            # Calculate mass flow rate from energy balance (with safeguards)
            Cp_water = 4186  # J/kg·K
            delta_T_water = agg_data['T_supply_avg'] - agg_data['T_return_avg']
            
            # Only calculate when there's significant temperature difference and heating
            valid_mask = (np.abs(delta_T_water) > 0.5) & (agg_data['Q_heating_total'] > 100)
            
            agg_data['mdot_water_total'] = 0.0  # Initialize with zeros
            agg_data.loc[valid_mask, 'mdot_water_total'] = (
                agg_data.loc[valid_mask, 'Q_heating_total'] / 
                (Cp_water * delta_T_water.loc[valid_mask])
            )
            print(f"Flow rate calculated from energy balance (not directly measured)")
        
        # 4. DISTURBANCE VARIABLES
        outdoor_temp_col = 'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'
        wind_speed_col = 'Environment:Site Wind Speed [m/s](TimeStep)'
        
        if outdoor_temp_col in self.df.columns:
            agg_data['T_outdoor'] = self.df[outdoor_temp_col]
        else:
            print("Warning: Outdoor temperature column not found, using default values")
            agg_data['T_outdoor'] = pd.Series([0] * len(self.df))
        
        if wind_speed_col in self.df.columns:
            agg_data['wind_speed'] = self.df[wind_speed_col]
        else:
            print("Warning: Wind speed column not found, using default values")
            agg_data['wind_speed'] = pd.Series([2] * len(self.df))
        
        # Internal gains (people, lights, equipment)
        people_gains = []
        light_gains = []
        equipment_gains = []
        
        for zone in self.zones:
            people_col = f'{zone}:Zone People Total Heating Rate [W](TimeStep)'
            light_col = f'{zone}:Zone Lights Total Heating Rate [W](TimeStep)'
            equipment_col = f'{zone}:Zone Electric Equipment Total Heating Rate [W](TimeStep)'
            
            # People gains
            if people_col in self.df.columns:
                people_gains.append(self.df[people_col])
            else:
                people_gains.append(pd.Series([0] * len(self.df)))
            
            # Light gains
            if light_col in self.df.columns:
                light_gains.append(self.df[light_col])
            else:
                light_gains.append(pd.Series([0] * len(self.df)))
            
            # Equipment gains
            if equipment_col in self.df.columns:
                equipment_gains.append(self.df[equipment_col])
            else:
                equipment_gains.append(pd.Series([0] * len(self.df)))
        
        agg_data['Q_people_total'] = sum(people_gains)
        agg_data['Q_lights_total'] = sum(light_gains)
        agg_data['Q_equipment_total'] = sum(equipment_gains)
        
        # Solar radiation gains (EnergyPlus 25.1 uses "Enclosure" rather than "Zone")
        solar_gains = []
        for zone in self.zones:
            solar_col = f'{zone}:Enclosure Windows Total Transmitted Solar Radiation Rate [W](TimeStep)'
            
            if solar_col in self.df.columns:
                solar_gains.append(self.df[solar_col])
            else:
                # Try old format as fallback
                solar_col_old = f'{zone}:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)'
                if solar_col_old in self.df.columns:
                    solar_gains.append(self.df[solar_col_old])
                    print(f"Using old solar variable format for {zone}")
                else:
                    print(f"Warning: Solar radiation column not found for {zone}")
                    solar_gains.append(pd.Series([0] * len(self.df)))
        
        agg_data['Q_solar_total'] = sum(solar_gains)
        
        # --- Solar separation ---
        # Separate internal gains (without solar) for gray-box modeling
        agg_data['Q_internal_no_solar'] = (agg_data['Q_people_total'] + 
                                            agg_data['Q_lights_total'] + 
                                            agg_data['Q_equipment_total'])
        
        # Solar radiation as separate input (renamed for clarity)
        agg_data['I_solar_total'] = agg_data['Q_solar_total']
        
        # Combined total for reference/validation only
        agg_data['Q_internal_with_solar'] = agg_data['Q_internal_no_solar'] + agg_data['I_solar_total']
        
        print(f"Internal gains aggregated:")
        print(f"   - Q_internal_no_solar: {agg_data['Q_internal_no_solar'].mean():.0f} W (mean)")
        print(f"   - I_solar_total: {agg_data['I_solar_total'].mean():.0f} W (mean)")
        print(f"   - Combined: {agg_data['Q_internal_with_solar'].mean():.0f} W (mean)")

        # 5. TIME FEATURES
        agg_data['hour'] = agg_data['DateTime'].dt.hour
        agg_data['day_of_week'] = agg_data['DateTime'].dt.dayofweek
        agg_data['day_of_year'] = agg_data['DateTime'].dt.dayofyear
        
        print(f"Aggregated data created: {len(agg_data)} timesteps")
        print(f"Variables created: {len(agg_data.columns)} columns")
        
        return agg_data
    
    def calculate_derived_variables(self, agg_data):
        """Calculate additional variables needed for gray-box modeling"""
        print("\n=== DERIVED VARIABLES ===")
        
        # Time step (10-minute intervals)
        dt = 600  # seconds
        
        # Temperature differences for heat transfer calculations
        agg_data['dT_outdoor'] = agg_data['T_air_avg'] - agg_data['T_outdoor']
        agg_data['dT_supply'] = agg_data['T_supply_avg'] - agg_data['T_air_avg']
        
        # Rate of change of air temperature (for dynamic modeling)
        agg_data['dT_air_dt'] = agg_data['T_air_avg'].diff() / (dt / 3600)  # °C/hour
        
        # Moving averages for visualization only (NOT for parameter identification)
        window = 6  # 1-hour moving average
        agg_data['T_air_avg_viz'] = agg_data['T_air_avg'].rolling(window=window, center=True).mean()
        agg_data['Q_heating_total_viz'] = agg_data['Q_heating_total'].rolling(window=window, center=True).mean()
        
        # Power density for benchmarking
        building_area = self.total_volume / 3.0  # Assuming 3m ceiling height
        agg_data['heating_power_density'] = agg_data['Q_heating_total'] / building_area
        agg_data['internal_gains_density'] = agg_data['Q_internal_no_solar'] / building_area
        agg_data['solar_gains_density'] = agg_data['I_solar_total'] / building_area
        
        print(f"Temperature difference range: {agg_data['dT_outdoor'].min():.1f} to {agg_data['dT_outdoor'].max():.1f} °C")
        print(f"Heating power range: {agg_data['Q_heating_total'].min():.0f} to {agg_data['Q_heating_total'].max():.0f} W")
        print(f"Flow rate range: {agg_data['mdot_water_total'].min():.3f} to {agg_data['mdot_water_total'].max():.3f} kg/s")
        
        return agg_data
    
    def verify_rc_model_inputs(self, model_data):
        """
        Verify that all required inputs for the gray-box RC model are present
        Based on the model structure:
        - Building: T_air, T_env, T_int (states)
        - Radiator: T_rad,1...T_rad,N (states)  
        - Inputs: T_out, Q_int, T_sup, mdot
        """
        print("\n=== RC MODEL INPUT VERIFICATION ===")
        
        required_inputs = {
            'T_air_avg': 'Building air temperature (output/state)',
            'T_outdoor': 'Outdoor temperature (input)',
            'Q_internal_no_solar': 'Internal gains WITHOUT solar (input)',
            'I_solar_total': 'Solar radiation - separate input (input)',
            'T_supply_avg': 'Radiator supply temperature (input)',
            'mdot_water_total': 'Water mass flow rate (input)',
            'Q_heating_total': 'Heating rate for validation'
        }
        
        all_present = True
        for var, description in required_inputs.items():
            if var in model_data.columns:
                data = model_data[var]
                print(f"{var}:")
                print(f"   Description: {description}")
                print(f"   Range: {data.min():.2f} to {data.max():.2f}")
                print(f"   Mean: {data.mean():.2f}, Std: {data.std():.2f}")
            else:
                print(f"{var}: MISSING")
                print(f"   {description}")
                all_present = False
        
        if all_present:
            print("\nAll required inputs for RC model identification are present!")
        else:
            print("\nSome required inputs are missing - check your EnergyPlus outputs")
        
        return all_present
    
    def create_zone_level_dataset(self):
        """
        Create zone-level dataset for potential multi-zone RC modeling
        Useful if you later want to identify individual zone parameters
        """
        print("\n=== ZONE-LEVEL DATASET CREATION ===")
        
        zone_data = {}
        
        for zone in self.zones:
            zone_df = pd.DataFrame()
            zone_df['DateTime'] = self.df['DateTime']
            
            # Zone temperature
            temp_col = f'{zone}:Zone Mean Air Temperature [C](TimeStep)'
            if temp_col in self.df.columns:
                zone_df['T_air'] = self.df[temp_col]
            
            # Zone heating
            heating_col = f'{zone} BASEBOARD HEATER:Baseboard Total Heating Rate [W](TimeStep)'
            if heating_col in self.df.columns:
                zone_df['Q_heating'] = self.df[heating_col]
            
            # Zone supply/return temperatures
            supply_col = f'{zone} BASEBOARD HEATER:Baseboard Water Inlet Temperature [C](TimeStep)'
            return_col = f'{zone} BASEBOARD HEATER:Baseboard Water Outlet Temperature [C](TimeStep)'
            if supply_col in self.df.columns:
                zone_df['T_supply'] = self.df[supply_col]
            if return_col in self.df.columns:
                zone_df['T_return'] = self.df[return_col]
            
            # Zone mass flow rate
            flow_col = f'{zone} BASEBOARD HEATER:Baseboard Hot Water Mass Flow Rate [kg/s](TimeStep)'
            if flow_col in self.df.columns:
                zone_df['mdot_water'] = self.df[flow_col]
            
            # Zone internal gains
            people_col = f'{zone}:Zone People Total Heating Rate [W](TimeStep)'
            lights_col = f'{zone}:Zone Lights Total Heating Rate [W](TimeStep)'
            equip_col = f'{zone}:Zone Electric Equipment Total Heating Rate [W](TimeStep)'
            # EnergyPlus 25.1 uses "Enclosure" for solar
            solar_col = f'{zone}:Enclosure Windows Total Transmitted Solar Radiation Rate [W](TimeStep)'
            
            zone_df['Q_people'] = self.df[people_col] if people_col in self.df.columns else 0
            zone_df['Q_lights'] = self.df[lights_col] if lights_col in self.df.columns else 0
            zone_df['Q_equipment'] = self.df[equip_col] if equip_col in self.df.columns else 0
            zone_df['Q_solar'] = self.df[solar_col] if solar_col in self.df.columns else 0
            zone_df['Q_internal'] = zone_df['Q_people'] + zone_df['Q_lights'] + zone_df['Q_equipment'] + zone_df['Q_solar']
            
            # Common outdoor conditions
            zone_df['T_outdoor'] = self.df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
            
            # Zone volume
            zone_df['volume'] = self.zone_volumes[zone]
            
            zone_data[zone] = zone_df
            print(f"  {zone}: {len(zone_df)} timesteps, {len(zone_df.columns)} variables")
        
        print(f"Zone-level datasets created for {len(zone_data)} zones")
        return zone_data
    
    def create_model_ready_dataset(self, agg_data):
        """Prepare final dataset for gray-box parameter identification"""
        print("\n=== MODEL-READY DATASET ===")
        
        # Core variables for gray-box modeling (NO smoothed versions)
        model_vars = [
            'DateTime',
            # Outputs (controlled variables)
            'T_air_avg',
            
            # Inputs (manipulated variables) 
            'Q_heating_total',
            'T_supply_avg',
            'T_return_avg',
            'mdot_water_total',
            
            # Inputs (disturbances) - SOLAR SEPARATED
            'T_outdoor',
            'Q_internal_no_solar',  # Internal gains WITHOUT solar
            'I_solar_total',         # Solar radiation SEPARATE
            'Q_people_total',        # Individual components for analysis
            'Q_lights_total',
            'Q_equipment_total',
            'Q_solar_total',         # Keep original for reference
            
            # Derived variables
            'dT_outdoor',
            'dT_supply', 
            'dT_air_dt',
            
            # Additional context
            'hour',
            'day_of_week'
        ]
        
        # Only include variables that exist
        available_vars = [var for var in model_vars if var in agg_data.columns]
        model_data = agg_data[available_vars].copy()
        
        # Remove rows with NaN (from diff calculation and flow issues)
        initial_length = len(model_data)
        model_data = model_data.dropna()
        final_length = len(model_data)
        
        if initial_length != final_length:
            print(f"Removed {initial_length - final_length} rows with NaN values")
        
        print(f"Final dataset shape: {model_data.shape}")
        print(f"Time range: {model_data['DateTime'].min()} to {model_data['DateTime'].max()}")
        
        # Final data quality check
        numeric_cols = model_data.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_cols).sum().sum()
        
        if inf_count > 0:
            print(f"Warning: {inf_count} infinite values detected")
            model_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            model_data.dropna(inplace=True)
            print(f"Infinite values handled, final shape: {model_data.shape}")
        else:
            print("Dataset clean and ready for parameter identification")
        
        # Verify RC model inputs
        self.verify_rc_model_inputs(model_data)
        
        return model_data

def extract_zone_volumes_guide():
    """Guide for extracting exact zone volumes from .eio file"""
    print("=" * 60)
    print("HOW TO EXTRACT EXACT ZONE VOLUMES FROM .eio FILE")
    print("=" * 60)
    print("1. Look for file: ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.eio")
    print("2. Open it in a text editor")
    print("3. Search for 'Zone Summary' or 'Zone Internal Gains Nominal'")
    print("4. Look for a table with columns including 'Volume [m3]'")
    print("5. Extract volume for each zone:")
    print("   - BACK_SPACE: ___ m³")
    print("   - CORE_RETAIL: ___ m³") 
    print("   - POINT_OF_SALE: ___ m³")
    print("   - FRONT_RETAIL: ___ m³")
    print("   - FRONT_ENTRY: ___ m³")
    print("6. Use these values in the zone_volumes parameter")
    print("=" * 60)

# Usage function with exact volumes
def process_energyplus_data(csv_file_path, zone_volumes=None, save_zone_data=False):
    """
    Complete data processing pipeline for EnergyPlus output
    
    Parameters:
    -----------
    csv_file_path : str
        Path to EnergyPlus CSV output file
    zone_volumes : dict, optional
        Exact zone volumes from .eio file
        Format: {'BACK_SPACE': vol1, 'CORE_RETAIL': vol2, ...}
    save_zone_data : bool, optional
        If True, also save zone-level datasets
    """
    
    if zone_volumes is None:
        extract_zone_volumes_guide()
        print("\nUsing placeholder volumes - extract exact values for final analysis!")
    
    # Initialize processor
    processor = RetailBuildingDataProcessor(csv_file_path, zone_volumes)
    
    # Processing pipeline
    df_clean = processor.clean_and_validate_data()
    df_aggregated = processor.aggregate_zone_data()
    
    if df_aggregated is None:
        print("Error: Data aggregation failed. Check your CSV column names.")
        return None, None
    
    df_derived = processor.calculate_derived_variables(df_aggregated)
    model_data = processor.create_model_ready_dataset(df_derived)
    
    # Save processed data
    output_file = csv_file_path.replace('.csv', '_processed.csv')
    model_data.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Optionally save zone-level data
    if save_zone_data:
        zone_data = processor.create_zone_level_dataset()
        for zone, zone_df in zone_data.items():
            zone_output = csv_file_path.replace('.csv', f'_zone_{zone}.csv')
            zone_df.to_csv(zone_output, index=False)
            print(f"Zone data saved to: {zone_output}")
    
    return model_data, processor

# Example usage with exact volumes (replace with your actual values)
if __name__ == "__main__":
    # STEP 1: Extract exact zone volumes from .eio file first!
    # Replace these placeholder values with actual values from your .eio file
    exact_volumes = {
        'BACK_SPACE': 2317.33,     # FROM .eio
        'CORE_RETAIL': 9762.95,    # FROM .eio
        'POINT_OF_SALE': 919.94,   # FROM .eio
        'FRONT_RETAIL': 919.94,    # FROM .eio
        'FRONT_ENTRY': 73.20       # FROM .eio
    }
    
    csv_file_path = "ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.csv"
    
    # Process with exact volumes (or None to use placeholders)
    model_data, processor = process_energyplus_data(
        csv_file_path, 
        exact_volumes,
        save_zone_data=False  # Set to True if you want zone-level datasets
    )
    
    if model_data is not None:
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETE!")
        print("="*60)
        print(f"Ready for gray-box parameter identification")
        print(f"Dataset: {len(model_data)} data points")
        print(f"Timestep: 10 minutes")
        print(f"Duration: {len(model_data) * 10 / 60 / 24:.1f} days")
        print("="*60)
    else:
        print("\nDATA PROCESSING FAILED!")
        print("Please check your CSV file and column names.")


def analyze_excitation_quality(csv_file_path):
    """
    Analyze the quality of PRBS excitation for system identification
    """
    # Load the processed data
    df = pd.read_csv(csv_file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Build the figure
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('PRBS Excitation Quality Assessment for System Identification', fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature profiles
    axes[0,0].plot(df['DateTime'], df['T_air_avg'], 'b-', label='Indoor Air (T_air_avg)', linewidth=2)
    axes[0,0].plot(df['DateTime'], df['T_outdoor'], 'g-', label='Outdoor', alpha=0.7)
    axes[0,0].plot(df['DateTime'], df['T_supply_avg'], 'r-', label='Supply Water (PRBS)', linewidth=2)
    axes[0,0].set_ylabel('Temperature [°C]', fontsize=11)
    axes[0,0].set_title('Indoor air temperature response to PRBS supply changes', fontsize=11, fontweight='bold')
    axes[0,0].legend(fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Temperature Response Detail (First 7 days)
    week_mask = df['DateTime'] <= df['DateTime'].iloc[0] + pd.Timedelta(days=7)
    axes[0,1].plot(df.loc[week_mask, 'DateTime'], df.loc[week_mask, 'T_air_avg'], 'b-', label='Indoor Air', linewidth=2)
    axes[0,1].plot(df.loc[week_mask, 'DateTime'], df.loc[week_mask, 'T_supply_avg'], 'r-', label='Supply (PRBS)', linewidth=2)
    axes[0,1].set_ylabel('Temperature [°C]', fontsize=11)
    axes[0,1].set_title('First Week Detail: Air Temp Response to PRBS', fontsize=11)
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: PRBS Signal Verification
    axes[1,0].plot(df['DateTime'], df['T_supply_avg'], 'r-', linewidth=2)
    axes[1,0].set_ylabel('Supply Temp [°C]', fontsize=11)
    axes[1,0].set_title('Supply Temperature: Follows Heating Curve (Expected)',fontsize=11)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=55, color='orange', linestyle='--', alpha=0.5, label='Low level')
    axes[1,0].axhline(y=65, color='red', linestyle='--', alpha=0.5, label='High level')
    axes[1,0].legend(fontsize=9)
    
    # Plot 4: Heating Power Response
    axes[1,1].plot(df['DateTime'], df['Q_heating_total']/1000, 'orange', linewidth=2)
    axes[1,1].set_ylabel('Heating Power [kW]', fontsize=11)
    axes[1,1].set_title('Heating System Response', fontsize=11)
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 5: Temperature Differences
    axes[2,0].plot(df['DateTime'], df['dT_outdoor'], 'b-', alpha=0.8)
    axes[2,0].set_ylabel('Indoor - Outdoor [°C]', fontsize=11)
    axes[2,0].set_title('Temperature Difference (Building Heat Loss)', fontsize=11)
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 6: Rate of Change
    axes[2,1].plot(df['DateTime'], df['dT_air_dt'], 'purple', alpha=0.8)
    axes[2,1].set_ylabel('dT/dt [°C/h]', fontsize=11)
    axes[2,1].set_title('Air Temperature Rate of Change', fontsize=11)
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 7: Temperature Correlation
    axes[3,0].scatter(df['T_supply_avg'], df['T_air_avg'], alpha=0.3, s=1, c='blue')
    axes[3,0].set_xlabel('Supply Temperature [°C]', fontsize=11)
    axes[3,0].set_ylabel('Air Temperature [°C]', fontsize=11)
    axes[3,0].set_title('Supply vs Air Temperature Correlation', fontsize=11)
    axes[3,0].grid(True, alpha=0.3)
    
    # Plot 8: Excitation Statistics
    axes[3,1].hist(df['dT_air_dt'].dropna(), bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[3,1].set_xlabel('dT/dt [°C/h]', fontsize=11)
    axes[3,1].set_ylabel('Frequency', fontsize=11)
    axes[3,1].set_title('Distribution of Temperature Rate Changes', fontsize=11)
    axes[3,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_output = csv_file_path.replace('_processed.csv', '_excitation_analysis.png')
    plt.savefig(fig_output, dpi=150, bbox_inches='tight')
    print(f"\nExcitation analysis plot saved to: {fig_output}")
    plt.show()
    
    # Quantitative Analysis
    print("\n" + "="*60)
    print("EXCITATION QUALITY ASSESSMENT")
    print("="*60)
    
    # Temperature variation statistics
    T_air_range = df['T_air_avg'].max() - df['T_air_avg'].min()
    T_air_std = df['T_air_avg'].std()
    T_supply_range = df['T_supply_avg'].max() - df['T_supply_avg'].min()
    
    print(f"\nIndoor Air Temperature:")
    print(f"  Range: {T_air_range:.2f}°C")
    print(f"  Standard Deviation: {T_air_std:.2f}°C")
    print(f"  Mean: {df['T_air_avg'].mean():.1f}°C")
    
    print(f"\nSupply Temperature (PRBS):")
    print(f"  Range: {T_supply_range:.2f}°C")
    unique_levels = sorted(df['T_supply_avg'].unique())
    print(f"  Unique levels: {len(unique_levels)}")
    print(f"  Level values: {[f'{x:.1f}' for x in unique_levels[:10]]}")  # Show first 10
    
    # Rate of change statistics
    dT_dt_nonzero = df['dT_air_dt'].dropna()
    dT_dt_range = dT_dt_nonzero.max() - dT_dt_nonzero.min()
    dT_dt_std = dT_dt_nonzero.std()
    
    print(f"\nTemperature Rate of Change:")
    print(f"  Range: {dT_dt_range:.3f}°C/h")
    print(f"  Standard Deviation: {dT_dt_std:.3f}°C/h")
    print(f"  Max heating rate: {dT_dt_nonzero.max():.3f}°C/h")
    print(f"  Max cooling rate: {dT_dt_nonzero.min():.3f}°C/h")
    
    # Excitation quality assessment
    print(f"\n" + "="*60)
    print("EXCITATION QUALITY VERDICT")
    print("="*60)
    
    if T_air_range > 2.0:
        print(f"EXCELLENT: Air temperature range ({T_air_range:.1f}°C) > 2°C")
        quality = "EXCELLENT"
    elif T_air_range > 1.0:
        print(f"GOOD: Air temperature range ({T_air_range:.1f}°C) > 1°C")
        quality = "GOOD"
    elif T_air_range > 0.5:
        print(f"MARGINAL: Air temperature range ({T_air_range:.1f}°C) is small")
        quality = "MARGINAL"
    else:
        print(f"POOR: Air temperature range ({T_air_range:.1f}°C) < 0.5°C")
        quality = "POOR"
    
    if T_air_std > 0.5:
        print(f"Good variability: Standard deviation {T_air_std:.2f}°C")
    else:
        print(f"Low variability: Standard deviation {T_air_std:.2f}°C")
    
    if abs(dT_dt_std) > 0.1:
        print(f"Good dynamics: Rate change std {dT_dt_std:.3f}°C/h")
    else:
        print(f"Slow dynamics: Rate change std {dT_dt_std:.3f}°C/h")
    
    # Recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if quality in ["EXCELLENT", "GOOD"]:
        print("Dataset is ready for parameter identification!")
        print("Proceed with gray-box modeling.")
        print("PRBS excitation successfully captured system dynamics.")
    elif quality == "MARGINAL":
        print("Dataset may work but consider improvements:")
        print("   - Increase PRBS amplitude (e.g., 50°C to 70°C)")
        print("   - Reduce building heating setpoints to increase sensitivity")
        print("   - Try running with current data first")
        print("   - If results are poor, re-run with stronger excitation")
    else:
        print("Dataset needs improvement:")
        print("   - Increase PRBS amplitude significantly")
        print("   - Check if thermostats are overriding the PRBS signal")
        print("   - Consider longer pulse durations (2-4 hours)")
        print("   - Re-run simulation with stronger excitation")
    
    print("="*60)
    
    return quality, {
        'T_air_range': T_air_range,
        'T_air_std': T_air_std,
        'dT_dt_std': dT_dt_std,
        'T_supply_range': T_supply_range
    }

# Run the analysis
if __name__ == "__main__":
    # First process the data
    csv_file = "ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.csv"
    
    exact_volumes = {
        'BACK_SPACE': 2317.33,
        'CORE_RETAIL': 9762.95,
        'POINT_OF_SALE': 919.94,
        'FRONT_RETAIL': 919.94,
        'FRONT_ENTRY': 73.20
    }
    
    model_data, processor = process_energyplus_data(csv_file, exact_volumes)
    
    # Then analyze excitation quality
    if model_data is not None:
        processed_file = csv_file.replace('.csv', '_processed.csv')
        quality, stats = analyze_excitation_quality(processed_file)
        
        print(f"\nFINAL ASSESSMENT: {quality}")
        if quality in ["EXCELLENT", "GOOD"]:
            print("Ready to proceed with gray-box parameter identification!")
        else:
            print("Consider improving excitation before parameter identification.")