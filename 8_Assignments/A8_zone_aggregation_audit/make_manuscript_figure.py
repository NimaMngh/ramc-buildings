"""
make_manuscript_figure.py

Single panel figure for the manuscript subsection on the choice of
zone aggregation. Plots the volume weighted aggregate temperature
T_vol against the minimum zone temperature T_minzone over the
nominal evaluation week.

Yellow shading indicates occupied steps where the aggregate stays
above the 20 degC comfort lower bound while the coldest zone falls
below it.

Reads results/raw/nominal_week_timeseries.csv
Writes results/figures/manuscript_figure.pdf and .png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "results" / "raw" / "nominal_week_timeseries.csv"
FIG_DIR = HERE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

T_MIN_COMFORT = 20.0 

def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    t = df["datetime"]

    line_vol, = ax.plot(t, df["T_vol"], lw=1.3, color="#2C3E50",
                        label=r"$T_{\rm vol}$ (volume weighted)")
    line_min, = ax.plot(t, df["T_minzone"], lw=1.3, color="#C0392B",
                        label=r"$T_{\rm minzone}$ (coldest zone)")
    

    occ = df["occupied"].astype(bool).values
    cold = (df["T_minzone"] < T_MIN_COMFORT).values
    agg_warm = (df["T_vol"] >= T_MIN_COMFORT).values
    
    hidden = occ & cold & agg_warm
    diff = np.diff(hidden.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for s, e in zip(starts, ends):
        if e - s > 1:
            ax.axvspan(t.iloc[s], t.iloc[min(e, len(t) - 1)],
                       color="#F1C40F", alpha=0.25, zorder=0)

    ax.set_ylabel("Indoor air temperature (°C)", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    
    ax.tick_params(labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(alpha=0.25)
    
    ax.set_ylim(17.5, 24.5)

    yellow_patch = mpatches.Patch(color="#F1C40F", alpha=0.3, 
                                  label="Hidden cold violations")
    
    ax.legend(handles=[line_vol, line_min, yellow_patch], 
              loc="upper center", bbox_to_anchor=(0.5, -0.28), 
              ncol=3, fontsize=9, frameon=False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"manuscript_figure.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved {out}")

if __name__ == "__main__":
    main()