# visualize_existing_scenarios.py
"""
Visualize pre-generated weather scenarios WITHOUT regenerating data.
Safe to run after co-simulation - only reads CSVs.

Updated 2026-02-15: Aligned with training-data-based scenario generation.
Now shows all three disturbance channels and uses correct output directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict
import json

# Constants (matching generate_scenarios_from_training.py)
COLD_SNAP_START_DAY = 2
COLD_SNAP_DURATION_HOURS = 48
COLD_SNAP_OFFSET_C = -10.0
FORECAST_BIAS_C = 1.5
FORECAST_NOISE_STD_C = 1.0
FORECAST_AR_COEF = 0.9
STEPS_PER_DAY = 144
STEPS_PER_WEEK = 1008


def load_existing_scenarios(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load pre-generated CSV files."""
    data_dir = Path(data_dir)

    scenarios = {}
    for name in ['nominal', 'cold_snap', 'forecast_error']:
        truth_file = data_dir / f"{name}_truth.csv"
        forecast_file = data_dir / f"{name}_forecast.csv"

        if not truth_file.exists():
            raise FileNotFoundError(f"Missing file: {truth_file}")

        scenarios[f'{name}_truth'] = pd.read_csv(truth_file, parse_dates=['timestamp'])
        scenarios[f'{name}_forecast'] = pd.read_csv(forecast_file, parse_dates=['timestamp'])

        t = scenarios[f'{name}_truth']
        print(f"  Loaded {name}: {len(t)} steps, "
              f"T_out=[{t['T_out_C'].min():.1f}, {t['T_out_C'].max():.1f}]°C, "
              f"Q_solar=[{t['Q_solar_W'].min():.0f}, {t['Q_solar_W'].max():.0f}] W")

    return scenarios


def plot_temperature_comparison(scenarios: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Panel (a)-(c): Temperature truth vs forecast for all three scenarios.
    This is the primary figure for the paper.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)

    scenario_names = ['nominal', 'cold_snap', 'forecast_error']
    titles = [
        '(a) Nominal',
        '(b) Cold Snap',
        '(c) Forecast Error',
    ]

    for idx, (name, title) in enumerate(zip(scenario_names, titles)):
        truth = scenarios[f'{name}_truth']
        forecast = scenarios[f'{name}_forecast']

        ax = axes[idx]

        x = np.arange(len(truth))
        hours = x * 10.0 / 60.0

        ax.plot(hours, truth['T_out_C'],
                label='Truth', linewidth=1.8, color='#1f77b4', zorder=3)
        ax.plot(hours, forecast['T_out_C'],
                label='Forecast', linewidth=1.4, color='#d62728',
                linestyle='--', dashes=(5, 3), alpha=0.85, zorder=2)

        ax.set_ylabel(r'$T_{\mathrm{out}}$ (°C)', fontsize=13)
        # Replace the legend call with this:
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray',
                  fontsize=11, fancybox=False)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylim(-30, 10)

        if idx == 2:
            ax.set_xlabel('Time (hours)', fontsize=13)

    axes[-1].set_xlim(0, hours[-1])

    fig.tight_layout(h_pad=1.2)
    output_file = output_dir / 'Evaluation_scenarios_temperature.png'
    fig.savefig(output_file, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")



def plot_all_disturbances(scenarios: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Full 3×3 panel: T_out, Q_solar, Q_internal for all scenarios.
    Shows that disturbance magnitudes are within training range.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    scenario_names = ['nominal', 'cold_snap', 'forecast_error']
    scenario_labels = ['Nominal', 'Cold Snap', 'Forecast Error']

    for col_idx, (name, slabel) in enumerate(zip(scenario_names, scenario_labels)):
        truth = scenarios[f'{name}_truth']
        forecast = scenarios[f'{name}_forecast']

        hours = np.arange(len(truth)) * 10.0 / 60.0

        # Row 0: T_out
        ax = axes[0, col_idx]
        ax.plot(hours, truth['T_out_C'], color='steelblue', linewidth=1.2, label='Truth')
        ax.plot(hours, forecast['T_out_C'], color='coral', linewidth=1.0,
                linestyle='--', label='Forecast', alpha=0.8)
        ax.set_ylabel('T$_{out}$ (°C)', fontsize=13)
        if col_idx == 0:
            ax.legend(fontsize=10, loc='upper right')
        ax.set_title(slabel, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')

        # Row 1: Q_solar
        ax = axes[1, col_idx]
        ax.fill_between(hours, 0, truth['Q_solar_W'], color='orange', alpha=0.6)
        ax.set_ylabel('Q$_{solar}$ (W)', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(bottom=0)

        # Row 2: Q_internal (if present)
        ax = axes[2, col_idx]
        if 'Q_internal_W' in truth.columns:
            ax.plot(hours, truth['Q_internal_W'], color='purple', linewidth=1.0, alpha=0.8)
            ax.set_ylabel('Q$_{internal}$ (W)', fontsize=13)
        else:
            ax.text(0.5, 0.5, 'Q_internal not in CSV\n(computed by simulator)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12,
                    color='gray')
            ax.set_ylabel('Q$_{internal}$ (W)', fontsize=13, color='gray')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlabel('Time (hours)', fontsize=13)

    plt.tight_layout()
    output_file = output_dir / 'Evaluation_scenarios_all_disturbances.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")


def plot_training_range_check(scenarios: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Overlay scenario disturbance ranges against training data ranges.
    Visual confirmation that scenarios stay within NN's trained domain.
    """
    # Training ranges (from inspection script output)
    training_ranges = {
        'T_out_C': (-26.0, 8.0),
        'Q_solar_W': (0.0, 27311.16),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenario_names = ['nominal', 'cold_snap', 'forecast_error']
    colors = {'nominal': 'steelblue', 'cold_snap': 'crimson', 'forecast_error': 'darkorange'}

    for ax, (col, (train_lo, train_hi)) in zip(axes, training_ranges.items()):
        # Training range as shaded band
        ax.axhspan(train_lo, train_hi, color='green', alpha=0.1, label='Training range')
        ax.axhline(train_lo, color='green', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.axhline(train_hi, color='green', linewidth=1.5, linestyle='--', alpha=0.7)

        for name in scenario_names:
            truth = scenarios[f'{name}_truth']
            hours = np.arange(len(truth)) * 10.0 / 60.0
            ax.plot(hours, truth[col], color=colors[name], linewidth=1.0,
                    alpha=0.7, label=f'{name} truth')

        ax.set_xlabel('Time (hours)', fontsize=13)
        ax.set_ylabel(col, fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_title(f'{col}: Scenarios vs Training Range', fontsize=13)

    plt.tight_layout()
    output_file = output_dir / 'Evaluation_scenarios_vs_training_range.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")


def print_scenario_statistics(scenarios: Dict[str, pd.DataFrame]):
    """Print detailed statistics for verification."""
    print("\n" + "-" * 60)
    print("SCENARIO STATISTICS")
    print("-" * 60)

    for name in ['nominal', 'cold_snap', 'forecast_error']:
        truth = scenarios[f'{name}_truth']
        forecast = scenarios[f'{name}_forecast']

        print(f"\n  {name.upper()}:")
        print(f"    Truth   T_out:  mean={truth['T_out_C'].mean():.1f}, "
              f"min={truth['T_out_C'].min():.1f}, max={truth['T_out_C'].max():.1f}")
        print(f"    Fcst    T_out:  mean={forecast['T_out_C'].mean():.1f}, "
              f"min={forecast['T_out_C'].min():.1f}, max={forecast['T_out_C'].max():.1f}")

        delta_T = forecast['T_out_C'].values - truth['T_out_C'].values
        print(f"    Fcst-Truth T:   mean={delta_T.mean():.2f}, "
              f"std={delta_T.std():.2f}, max|Δ|={np.max(np.abs(delta_T)):.2f}")

        print(f"    Q_solar:        max={truth['Q_solar_W'].max():.0f} W")
        if 'Q_internal_W' in truth.columns:
            print(f"    Q_internal:     [{truth['Q_internal_W'].min():.0f}, "
                  f"{truth['Q_internal_W'].max():.0f}] W")


def main():
    # Path to weather data (now from training-data extraction)
    DATA_DIR = (
        r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
        r"\New Project for Risk Aware Model then Control\7_Close Loop Performance"
        r"\M1_Weather_Data\data_ramc_epw"
    )

    OUTPUT_DIR = Path(DATA_DIR) / "visualizations"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("VISUALIZING WEATHER SCENARIOS (Training-Data Based)")
    print("=" * 70)
    print(f"\nReading from: {DATA_DIR}")
    print(f"Saving to:    {OUTPUT_DIR}\n")

    # Load metadata if available
    metadata_file = Path(DATA_DIR) / "scenario_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
        print(f"Scenario metadata:")
        print(f"  Source episode: {meta.get('episode_id', '?')}")
        print(f"  Coldness percentile: {meta.get('coldness_percentile', '?')}%")
        print(f"  Mean T_out: {meta.get('mean_T_out_C', '?')}°C")
        print(f"  Solar column: {meta.get('solar_column_used', '?')}")
        print()

    # Load CSVs
    scenarios = load_existing_scenarios(DATA_DIR)

    # Print statistics
    print_scenario_statistics(scenarios)

    # Generate all figures
    print("\nGenerating figures...")
    plot_temperature_comparison(scenarios, OUTPUT_DIR)
    plot_all_disturbances(scenarios, OUTPUT_DIR)
    plot_training_range_check(scenarios, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE — NO DATA REGENERATED")
    print("=" * 70)
    print(f"\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
