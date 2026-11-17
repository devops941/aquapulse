"""
Multi-Year Analysis and Year-over-Year (YoY) Comparison Engine
Compares historical observations (2023-24) against current reporting year (2024-25).
"""

import os
import pandas as pd
import numpy as np

def run_multiyear_analysis():
    """
    Executes full multi-year and Year-over-Year (YoY) comparative analysis:
    - Annual whole-year comparisons across all 38 districts + State Average
    - Common months (Nov, Dec, Jan, Feb, Mar, Apr, May) comparative deltas
    - Seasonal aggregates (Winter and Hot-Weather seasons)
    Outputs:
      - data/processed/multiyear_district_annual_comparison.csv
      - data/processed/multiyear_district_monthly_comparison.csv
      - data/processed/multiyear_seasonal_comparison.csv
      - reports/multiyear_yoy_report.txt
    """
    # 1. Load Datasets
    raw_24 = pd.read_csv("data/raw/rainfall_by_district_2024_25.csv")
    clean_24 = pd.read_csv("data/processed/district_monthly_cleaned.csv")
    annual_23 = pd.read_csv("data/processed/annual_totals_2023_2024_extracted.csv")
    monthly_23 = pd.read_csv("data/processed/district_monthly_2023_2024_extracted.csv")
    state_24 = pd.read_csv("data/processed/state_monthly_cleaned.csv")

    # 2. District-Level Annual Comparison
    annual_24_raw = raw_24[raw_24["season"] == "Whole year"][["district", "actual_mm", "normal_mm", "pct_deviation"]].copy()
    annual_24_raw.columns = ["district", "actual_2024_25", "normal_2024_25", "pct_dev_2024_25"]

    annual_23_clean = annual_23[["district_standardized", "actual_mm", "normal_mm", "pct_deviation"]].copy()
    annual_23_clean.columns = ["district", "actual_2023_24", "normal_2023_24", "pct_dev_2023_24"]

    df_annual = annual_23_clean.merge(annual_24_raw, on="district")
    df_annual["yoy_growth_mm"] = df_annual["actual_2024_25"] - df_annual["actual_2023_24"]
    df_annual["yoy_growth_pct"] = np.where(
        df_annual["actual_2023_24"] > 0,
        (df_annual["yoy_growth_mm"] / df_annual["actual_2023_24"]) * 100,
        0.0
    )

    def categorize_yoy(pct):
        if pct >= 20.0:
            return "Substantial Increase (>+20%)"
        elif pct >= 5.0:
            return "Moderate Increase (+5% to +20%)"
        elif pct >= -5.0:
            return "Stable (-5% to +5%)"
        elif pct >= -20.0:
            return "Moderate Decrease (-20% to -5%)"
        else:
            return "Substantial Decrease (<-20%)"

    df_annual["yoy_trend"] = df_annual["yoy_growth_pct"].apply(categorize_yoy)

    # Save annual comparison
    os.makedirs("data/processed", exist_ok=True)
    df_annual.to_csv("data/processed/multiyear_district_annual_comparison.csv", index=False)

    # 3. Common Months Matching (Nov, Dec, Jan, Feb, Mar, Apr, May)
    # 2024-25 monthly data
    clean_24["month_name"] = pd.to_datetime(clean_24["period"]).dt.strftime("%B")
    m24_dist = clean_24[["district", "month_name", "actual_mm", "normal_mm", "pct_deviation"]].copy()
    m24_dist.columns = ["district", "month", "actual_2024_25", "normal_2024_25", "pct_dev_2024_25"]

    # Also add state average for 2024-25
    state_24["month_name"] = pd.to_datetime(state_24["period"]).dt.strftime("%B")
    m24_state = state_24[["district", "month_name", "actual_mm", "normal_mm", "pct_deviation"]].copy()
    m24_state.columns = ["district", "month", "actual_2024_25", "normal_2024_25", "pct_dev_2024_25"]
    m24_combined = pd.concat([m24_dist, m24_state], ignore_index=True)

    # 2023-24 monthly data
    m23_dist = monthly_23[["district_standardized", "month", "actual_mm", "normal_mm", "pct_deviation"]].copy()
    m23_dist.columns = ["district", "month", "actual_2023_24", "normal_2023_24", "pct_dev_2023_24"]

    df_monthly = m23_dist.merge(m24_combined, on=["district", "month"], how="inner")
    df_monthly["yoy_growth_mm"] = df_monthly["actual_2024_25"] - df_monthly["actual_2023_24"]
    df_monthly["yoy_growth_pct"] = np.where(
        df_monthly["actual_2023_24"] > 0,
        (df_monthly["yoy_growth_mm"] / df_monthly["actual_2023_24"]) * 100,
        np.where(df_monthly["actual_2024_25"] > 0, 100.0, 0.0)
    )
    df_monthly.to_csv("data/processed/multiyear_district_monthly_comparison.csv", index=False)

    # 4. Seasonal Comparisons (Winter & Summer)
    seasonal_records = []
    for d in df_annual["district"].unique():
        d_sub23 = monthly_23[monthly_23["district_standardized"] == d]
        d_sub24 = m24_combined[m24_combined["district"] == d]

        # Winter (Jan + Feb)
        win23 = d_sub23[d_sub23["month"].isin(["January", "February"])]["actual_mm"].sum()
        win24 = d_sub24[d_sub24["month"].isin(["January", "February"])]["actual_2024_25"].sum()
        win_norm24 = d_sub24[d_sub24["month"].isin(["January", "February"])]["normal_2024_25"].sum()
        win_delta = win24 - win23
        win_pct = (win_delta / win23 * 100) if win23 > 0 else 0.0

        seasonal_records.append({
            "district": d, "season": "Winter Monsoon",
            "actual_2023_24": win23, "actual_2024_25": win24,
            "normal_mm": win_norm24, "yoy_growth_mm": win_delta, "yoy_growth_pct": win_pct
        })

        # Summer / Hot Weather (Mar + Apr + May)
        sum23 = d_sub23[d_sub23["month"].isin(["March", "April", "May"])]["actual_mm"].sum()
        sum24 = d_sub24[d_sub24["month"].isin(["March", "April", "May"])]["actual_2024_25"].sum()
        sum_norm24 = d_sub24[d_sub24["month"].isin(["March", "April", "May"])]["normal_2024_25"].sum()
        sum_delta = sum24 - sum23
        sum_pct = (sum_delta / sum23 * 100) if sum23 > 0 else 0.0

        seasonal_records.append({
            "district": d, "season": "Hot-Weather Season",
            "actual_2023_24": sum23, "actual_2024_25": sum24,
            "normal_mm": sum_norm24, "yoy_growth_mm": sum_delta, "yoy_growth_pct": sum_pct
        })

    df_seasonal = pd.DataFrame(seasonal_records)
    df_seasonal.to_csv("data/processed/multiyear_seasonal_comparison.csv", index=False)

    # 5. Generate Comprehensive Multi-Year & YoY Report
    state_annual = df_annual[df_annual["district"] == "Tamil Nadu (State)"].iloc[0]
    district_only = df_annual[df_annual["district"] != "Tamil Nadu (State)"]
    top_increase = district_only.sort_values("yoy_growth_mm", ascending=False).head(5)
    top_decrease = district_only.sort_values("yoy_growth_mm", ascending=True).head(5)

    state_monthly = df_monthly[df_monthly["district"] == "Tamil Nadu (State)"]
    month_order = ["November", "December", "January", "February", "March", "April", "May"]
    state_monthly["month"] = pd.Categorical(state_monthly["month"], categories=month_order, ordered=True)
    state_monthly = state_monthly.sort_values("month")

    report_lines = [
        "=" * 90,
        "TAMIL NADU MULTI-YEAR RAINFALL & YEAR-OVER-YEAR (YoY) COMPARATIVE REPORT",
        "Comparative Baseline: 2023-2024 (Historical Prior) vs 2024-2025 (Reporting Year)",
        "=" * 90,
        "",
        "1. STATE-LEVEL ANNUAL EXECUTIVE SUMMARY",
        "-" * 90,
        f"State Total Rainfall (2023-24): {state_annual['actual_2023_24']:.1f} mm (Normal: {state_annual['normal_2023_24']:.1f} mm, Dev: {state_annual['pct_dev_2023_24']:+.1f}%)",
        f"State Total Rainfall (2024-25): {state_annual['actual_2024_25']:.1f} mm (Normal: {state_annual['normal_2024_25']:.1f} mm, Dev: {state_annual['pct_dev_2024_25']:+.1f}%)",
        f"Year-over-Year Delta:           {state_annual['yoy_growth_mm']:+.1f} mm ({state_annual['yoy_growth_pct']:+.2f}%)",
        f"Macro Trend:                    {state_annual['yoy_trend']}",
        "",
        "2. TOP 5 DISTRICTS WITH HIGHEST RAINFALL SURGE IN 2024-25",
        "-" * 90,
        f"{'District':<20} {'2023-24 (mm)':<15} {'2024-25 (mm)':<15} {'YoY Delta (mm)':<18} {'YoY Growth (%)':<15}"
    ]

    for _, r in top_increase.iterrows():
        report_lines.append(f"{r['district']:<20} {r['actual_2023_24']:<15.1f} {r['actual_2024_25']:<15.1f} {r['yoy_growth_mm']:<+18.1f} {r['yoy_growth_pct']:<+15.2f}%")

    report_lines.extend([
        "",
        "3. TOP 5 DISTRICTS WITH HIGHEST RAINFALL DEFICIT IN 2024-25",
        "-" * 90,
        f"{'District':<20} {'2023-24 (mm)':<15} {'2024-25 (mm)':<15} {'YoY Delta (mm)':<18} {'YoY Growth (%)':<15}"
    ])

    for _, r in top_decrease.iterrows():
        report_lines.append(f"{r['district']:<20} {r['actual_2023_24']:<15.1f} {r['actual_2024_25']:<15.1f} {r['yoy_growth_mm']:<+18.1f} {r['yoy_growth_pct']:<+15.2f}%")

    report_lines.extend([
        "",
        "4. COMMON METEOROLOGICAL MONTHS COMPARISON (STATE AVERAGE)",
        "-" * 90,
        f"{'Month':<15} {'2023-24 (mm)':<15} {'2024-25 (mm)':<15} {'Delta (mm)':<15} {'YoY Change (%)':<15} {'Meteorological Note':<25}"
    ])

    notes = {
        "November": "NE Monsoon peak shift",
        "December": "Heavy cyclonic systems",
        "January": "Post-monsoon drydown",
        "February": "Standard dry winter",
        "March": "Early summer convective showers",
        "April": "Intense pre-monsoon activity",
        "May": "Robust pre-monsoon thunderstorms"
    }

    for _, r in state_monthly.iterrows():
        m_note = notes.get(r['month'], "")
        report_lines.append(f"{r['month']:<15} {r['actual_2023_24']:<15.1f} {r['actual_2024_25']:<15.1f} {r['yoy_growth_mm']:<+15.1f} {r['yoy_growth_pct']:<+15.2f}% {m_note:<25}")

    report_lines.extend([
        "",
        "5. SEASONAL AGGREGATE COMPARISONS (STATE AVERAGE)",
        "-" * 90
    ])

    for s_name in ["Winter Monsoon", "Hot-Weather Season"]:
        s_row = df_seasonal[(df_seasonal["district"] == "Tamil Nadu (State)") & (df_seasonal["season"] == s_name)].iloc[0]
        report_lines.append(f"{s_name:<22}: 2023-24 = {s_row['actual_2023_24']:>6.1f} mm | 2024-25 = {s_row['actual_2024_25']:>6.1f} mm | Delta = {s_row['yoy_growth_mm']:>+6.1f} mm ({s_row['yoy_growth_pct']:>+6.2f}%)")

    report_lines.extend([
        "",
        "=" * 90,
        "END OF MULTI-YEAR & YoY REPORT",
        "=" * 90
    ])

    report_text = "\n".join(report_lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/multiyear_yoy_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Saved Multi-Year & YoY Comparative Report to reports/multiyear_yoy_report.txt")

    return df_annual, df_monthly, df_seasonal

if __name__ == "__main__":
    run_multiyear_analysis()
