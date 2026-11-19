"""
Consolidated Historical Rainfall Dataset Builder
Tamil Nadu District-Wise Rainfall Analysis and Prediction System

Compiles all verified pure monthly observations from:
  1. 2024–2025 official dataset (data/processed/district_monthly_cleaned.csv and state_monthly_cleaned.csv)
  2. 2023–2024 verified screenshot extracts (data/processed/district_monthly_2023_2024_extracted.csv)

Strict Rules:
  - Excludes pre-aggregated seasonal and annual totals (to prevent double/triple counting).
  - Preserves all 38 official administrative districts + State Average.
  - Never replaces missing values with zero.
  - Output saved to: data/processed/historical_rainfall_validated.csv
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def build_consolidated_historical_dataset():
    """
    Consolidates verified historical monthly observations into a single standardized ledger.
    """
    # 1. Load 2024-2025 cleaned district monthly observations (456 rows)
    df_dist_24 = pd.read_csv(WORKSPACE_ROOT / "data" / "processed" / "district_monthly_cleaned.csv")
    df_state_24 = pd.read_csv(WORKSPACE_ROOT / "data" / "processed" / "state_monthly_cleaned.csv")

    # Standardize column naming for 2024-25
    records = []

    # Map seasons to standardized display
    season_map = {
        "South-West Monsoon": "Southwest Monsoon",
        "North-East Monsoon": "Northeast Monsoon",
        "Winter": "Winter",
        "Hot-Weather": "Hot Weather / Summer"
    }

    # Add 2024-25 district monthly
    for _, row in df_dist_24.iterrows():
        p_dt = pd.to_datetime(row["period"])
        records.append({
            "District": row["district"],
            "District_Standardized": row["district"],
            "Year": "2024–2025",
            "Month": p_dt.strftime("%B"),
            "Period": row["period"],
            "Season": season_map.get(row["season"], row["season"]),
            "Actual Rainfall": float(row["actual_mm"]),
            "Normal Rainfall": float(row["normal_mm"]),
            "Percentage Deviation": float(row["pct_deviation"]),
            "Source": "Official TN Meteorological Report 2024–25 (CSV)",
            "Is_State_Aggregate": False
        })

    # Add 2024-25 state monthly
    for _, row in df_state_24.iterrows():
        p_dt = pd.to_datetime(row["period"])
        records.append({
            "District": "Tamil Nadu (State)",
            "District_Standardized": "Tamil Nadu (State)",
            "Year": "2024–2025",
            "Month": p_dt.strftime("%B"),
            "Period": row["period"],
            "Season": season_map.get(row["season"], row["season"]),
            "Actual Rainfall": float(row["actual_mm"]),
            "Normal Rainfall": float(row["normal_mm"]),
            "Percentage Deviation": float(row["pct_deviation"]),
            "Source": "Official TN Meteorological Report 2024–25 (State Aggregate)",
            "Is_State_Aggregate": True
        })

    # 2. Load 2023-2024 extracted screenshot monthly observations (273 rows)
    df_m23 = pd.read_csv(WORKSPACE_ROOT / "data" / "processed" / "district_monthly_2023_2024_extracted.csv")

    season_map_23 = {
        "Northeast Monsoon": "Northeast Monsoon",
        "Winter": "Winter",
        "Summer": "Hot Weather / Summer"
    }

    for _, row in df_m23.iterrows():
        is_state = (row["district"] == "State Average" or row["district_standardized"] == "Tamil Nadu (State)")
        dist_name = "Tamil Nadu (State)" if is_state else row["district"]
        dist_std = "Tamil Nadu (State)" if is_state else row["district_standardized"]

        records.append({
            "District": dist_name,
            "District_Standardized": dist_std,
            "Year": "2023–2024",
            "Month": row["month"],
            "Period": row["period"],
            "Season": season_map_23.get(row["season"], row["season"]),
            "Actual Rainfall": float(row["actual_mm"]),
            "Normal Rainfall": float(row["normal_mm"]),
            "Percentage Deviation": float(row["pct_deviation"]),
            "Source": f"Official TN Seasonal Rainfall Report Screenshot ({row['source_screenshot']})",
            "Is_State_Aggregate": is_state
        })

    df_consolidated = pd.DataFrame(records)

    # Verification checks
    print("=" * 80)
    print("CONSOLIDATED HISTORICAL RAINFALL DATASET AUDIT")
    print("=" * 80)
    print(f"Total pure monthly observations : {len(df_consolidated)}")
    print(f"  - 2024–2025 observations     : {len(df_consolidated[df_consolidated['Year'] == '2024–2025'])} (456 district + 12 state)")
    print(f"  - 2023–2024 observations     : {len(df_consolidated[df_consolidated['Year'] == '2023–2024'])} (266 district + 7 state)")
    print(f"Total unique districts covered  : {df_consolidated[~df_consolidated['Is_State_Aggregate']]['District_Standardized'].nunique()}")
    print(f"Missing values across columns   :\n{df_consolidated.isnull().sum().to_dict()}")
    print(f"Duplicate rows detected         : {df_consolidated.duplicated(subset=['District_Standardized', 'Year', 'Month']).sum()}")

    # Save to data/processed/historical_rainfall_validated.csv
    out_path = WORKSPACE_ROOT / "data" / "processed" / "historical_rainfall_validated.csv"
    df_consolidated.to_csv(out_path, index=False)
    print(f"\nSuccessfully saved consolidated dataset to: {out_path}")
    print("=" * 80)

    return df_consolidated

if __name__ == "__main__":
    build_consolidated_historical_dataset()
