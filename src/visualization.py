"""
Exploratory Data Analysis (EDA) & Data Visualization Engine
Tamil Nadu District-wise Rainfall Analysis and Prediction (2024–2025)

Author: Senior Python Data Scientist & Data Visualization Engineer
Path: src/visualization.py
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Set workspace-local matplotlib cache directory to avoid sandboxed permissions issues
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(WORKSPACE_ROOT / ".cache" / "matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Professional styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

DEFAULT_DISTRICT_PATH = WORKSPACE_ROOT / "data" / "processed" / "district_monthly_cleaned.csv"
DEFAULT_STATE_PATH = WORKSPACE_ROOT / "data" / "processed" / "state_monthly_cleaned.csv"
DEFAULT_FIGURES_DIR = WORKSPACE_ROOT / "reports" / "figures"
DEFAULT_REPORT_PATH = WORKSPACE_ROOT / "reports" / "eda_report.txt"

SEASON_DISPLAY_NAMES = {
    "South-West Monsoon": "Southwest Monsoon",
    "North-East Monsoon": "Northeast Monsoon",
    "Winter": "Winter",
    "Hot-Weather": "Hot Weather / Summer",
}


def load_cleaned_data(
    district_path: Path = DEFAULT_DISTRICT_PATH,
    state_path: Path = DEFAULT_STATE_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned monthly datasets and parse temporal fields."""
    if not district_path.exists():
        raise FileNotFoundError(f"District dataset not found at: {district_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State dataset not found at: {state_path}")

    df_dist = pd.read_csv(district_path)
    df_state = pd.read_csv(state_path)

    df_dist["period"] = pd.to_datetime(df_dist["period"])
    df_state["period"] = pd.to_datetime(df_state["period"])

    df_dist["month_name"] = df_dist["period"].dt.strftime("%b %Y")
    df_state["month_name"] = df_state["period"].dt.strftime("%b %Y")
    df_dist["season_display"] = df_dist["season"].map(SEASON_DISPLAY_NAMES).fillna(df_dist["season"])
    df_state["season_display"] = df_state["season"].map(SEASON_DISPLAY_NAMES).fillna(df_state["season"])

    return df_dist, df_state


# ==============================================================================
# 1. ANALYSIS & AGGREGATION FUNCTIONS
# ==============================================================================

def analyze_district_rainfall(df_dist: pd.DataFrame) -> pd.DataFrame:
    """Calculate district-wise annual aggregations, deviations, and rankings."""
    dist_totals = df_dist.groupby(["district_code", "district"]).agg(
        total_actual=("actual_mm", "sum"),
        total_normal=("normal_mm", "sum"),
        avg_monthly_actual=("actual_mm", "mean"),
        std_monthly_actual=("actual_mm", "std"),
        min_monthly_actual=("actual_mm", "min"),
        max_monthly_actual=("actual_mm", "max"),
    ).reset_index()

    dist_totals["abs_deviation"] = dist_totals["total_actual"] - dist_totals["total_normal"]
    dist_totals["pct_deviation"] = (
        (dist_totals["abs_deviation"] / dist_totals["total_normal"]) * 100
    ).round(2)
    dist_totals["surplus_flag"] = dist_totals["pct_deviation"].apply(
        lambda x: "Surplus" if x >= 0 else "Deficit"
    )

    dist_totals = dist_totals.sort_values(by="total_actual", ascending=False).reset_index(drop=True)
    dist_totals["rank_by_actual"] = dist_totals.index + 1
    return dist_totals


def analyze_monthly_trends(df_state: pd.DataFrame, df_dist: pd.DataFrame) -> pd.DataFrame:
    """Analyze monthly progression at state and district-pooled levels."""
    state_monthly = df_state.sort_values(by="period").copy()
    state_monthly["abs_deviation"] = state_monthly["actual_mm"] - state_monthly["normal_mm"]

    dist_month_agg = df_dist.groupby("period").agg(
        dist_avg_actual=("actual_mm", "mean"),
        dist_total_actual=("actual_mm", "sum"),
        dist_avg_normal=("normal_mm", "mean"),
    ).reset_index()

    monthly_summary = pd.merge(state_monthly, dist_month_agg, on="period")
    return monthly_summary


def analyze_seasonal_rainfall(df_dist: pd.DataFrame, df_state: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate monthly observations into seasonal comparisons."""
    season_order = ["Southwest Monsoon", "Northeast Monsoon", "Winter", "Hot Weather / Summer"]

    dist_seasonal = df_dist.groupby("season_display").agg(
        total_actual=("actual_mm", "sum"),
        total_normal=("normal_mm", "sum"),
        avg_district_actual=("actual_mm", "mean"),
        count=("actual_mm", "count"),
    ).reindex(season_order).reset_index()

    dist_seasonal["abs_deviation"] = dist_seasonal["total_actual"] - dist_seasonal["total_normal"]
    dist_seasonal["pct_deviation"] = (
        (dist_seasonal["abs_deviation"] / dist_seasonal["total_normal"]) * 100
    ).round(2)

    state_seasonal = df_state.groupby("season_display").agg(
        actual_mm=("actual_mm", "sum"),
        normal_mm=("normal_mm", "sum"),
        month_count=("actual_mm", "count"),
    ).reindex(season_order).reset_index()

    state_seasonal["abs_deviation"] = state_seasonal["actual_mm"] - state_seasonal["normal_mm"]
    state_seasonal["pct_deviation"] = (
        (state_seasonal["abs_deviation"] / state_seasonal["normal_mm"]) * 100
    ).round(2)

    total_state_actual = state_seasonal["actual_mm"].sum()
    state_seasonal["contribution_pct"] = (
        (state_seasonal["actual_mm"] / total_state_actual) * 100
    ).round(2)

    return dist_seasonal, state_seasonal


def calculate_statistical_summary(df_dist: pd.DataFrame, df_state: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute detailed summary statistics across observation layers."""
    def stats_dict(s: pd.Series) -> Dict[str, float]:
        return {
            "count": int(s.count()),
            "mean": float(round(s.mean(), 2)),
            "std": float(round(s.std(), 2)),
            "median": float(round(s.median(), 2)),
            "min": float(round(s.min(), 2)),
            "max": float(round(s.max(), 2)),
            "sum": float(round(s.sum(), 2)),
        }

    dist_annual = df_dist.groupby("district")["actual_mm"].sum()
    dist_annual_norm = df_dist.groupby("district")["normal_mm"].sum()

    return {
        "district_monthly_actual": stats_dict(df_dist["actual_mm"]),
        "district_monthly_normal": stats_dict(df_dist["normal_mm"]),
        "district_annual_actual": stats_dict(dist_annual),
        "district_annual_normal": stats_dict(dist_annual_norm),
        "state_monthly_actual": stats_dict(df_state["actual_mm"]),
        "state_monthly_normal": stats_dict(df_state["normal_mm"]),
    }


# ==============================================================================
# 2. REQUIRED STEP 4 VISUALIZATION MODULES (10 CORE CHARTS)
# ==============================================================================

def plot_01_monthly_trend(monthly_summary: pd.DataFrame, out_dir: Path) -> Path:
    """1. Tamil Nadu Monthly Rainfall Trend."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = monthly_summary["month_name"]

    ax.plot(x_labels, monthly_summary["actual_mm"], marker="o", linewidth=2.5,
            color="#1E88E5", label="Actual Rainfall (mm)")
    ax.plot(x_labels, monthly_summary["normal_mm"], marker="s", linewidth=2.0,
            linestyle="--", color="#FF7043", label="Normal Expected Benchmark (mm)")

    wettest_idx = monthly_summary["actual_mm"].idxmax()
    driest_idx = monthly_summary["actual_mm"].idxmin()

    ax.annotate(
        f"Peak: {monthly_summary.loc[wettest_idx, 'actual_mm']:.1f} mm\n({monthly_summary.loc[wettest_idx, 'month_name']})",
        xy=(wettest_idx, monthly_summary.loc[wettest_idx, "actual_mm"]),
        xytext=(wettest_idx - 1.2, monthly_summary.loc[wettest_idx, "actual_mm"] + 20),
        arrowprops=dict(facecolor="#1565C0", arrowstyle="->", lw=1.5),
        weight="bold", fontsize=9, color="#0D47A1"
    )
    ax.annotate(
        f"Lowest: {monthly_summary.loc[driest_idx, 'actual_mm']:.1f} mm\n({monthly_summary.loc[driest_idx, 'month_name']})",
        xy=(driest_idx, monthly_summary.loc[driest_idx, "actual_mm"]),
        xytext=(driest_idx + 0.3, monthly_summary.loc[driest_idx, "actual_mm"] + 35),
        arrowprops=dict(facecolor="#C62828", arrowstyle="->", lw=1.5),
        weight="bold", fontsize=9, color="#B71C1C"
    )

    ax.set_title("1. Tamil Nadu State Monthly Rainfall Trend (Jun 2024 – May 2025)", weight="bold", pad=15)
    ax.set_xlabel("Meteorological Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend(frameon=True, loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = out_dir / "01_tamil_nadu_monthly_rainfall_trend.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_02_district_totals(dist_summary: pd.DataFrame, out_dir: Path) -> Path:
    """2. District-wise Total Rainfall Comparison."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ordered = dist_summary.sort_values(by="total_actual", ascending=True)

    colors = sns.color_palette("Blues_r", n_colors=len(ordered))
    bars = ax.barh(ordered["district"], ordered["total_actual"], color=colors, edgecolor="black", linewidth=0.4)

    state_avg = ordered["total_actual"].mean()
    ax.axvline(state_avg, color="#D32F2F", linestyle="--", linewidth=1.8, label=f"State Mean: {state_avg:.1f} mm")

    ax.set_title("2. District-wise Total Actual Rainfall Comparison (2024–2025)", weight="bold", pad=15)
    ax.set_xlabel("Total Annual Rainfall (mm)")
    ax.set_ylabel("District")
    ax.legend(loc="lower right")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 15, bar.get_y() + bar.get_height() / 2, f"{w:.0f}",
                va="center", ha="left", fontsize=8, color="#333333")

    plt.tight_layout()
    out_path = out_dir / "02_district_wise_total_rainfall_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_03_actual_vs_normal(dist_summary: pd.DataFrame, out_dir: Path) -> Path:
    """3. Actual Rainfall vs Normal Rainfall (Grouped Bar Chart per district)."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ordered = dist_summary.sort_values(by="total_actual", ascending=False)

    x = np.arange(len(ordered))
    width = 0.38

    ax.bar(x - width/2, ordered["total_actual"], width, label="Actual Rainfall", color="#1976D2", edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, ordered["total_normal"], width, label="Normal Benchmark", color="#FFA726", edgecolor="black", linewidth=0.4)

    ax.set_title("3. Actual Rainfall vs Normal Rainfall by District (2024–2025)", weight="bold", pad=15)
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["district"], rotation=75, ha="right", fontsize=9)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_path = out_dir / "03_actual_rainfall_vs_normal_rainfall.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_04_monthly_distribution(df_dist: pd.DataFrame, out_dir: Path) -> Path:
    """4. Monthly Rainfall Distribution (Boxplot across all 38 districts)."""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Order months chronologically
    month_order = sorted(df_dist["month_name"].unique(), key=lambda m: pd.to_datetime(m, format="%b %Y"))

    sns.boxplot(
        data=df_dist,
        x="month_name",
        y="actual_mm",
        hue="month_name",
        legend=False,
        order=month_order,
        palette="crest",
        fliersize=4,
        linewidth=1.2,
        ax=ax
    )
    sns.stripplot(
        data=df_dist,
        x="month_name",
        y="actual_mm",
        order=month_order,
        color="#1A237E",
        alpha=0.35,
        jitter=0.2,
        size=4,
        ax=ax
    )

    ax.set_title("4. Monthly Rainfall Distribution Across All 38 Districts (2024–2025)", weight="bold", pad=15)
    ax.set_xlabel("Meteorological Month")
    ax.set_ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = out_dir / "04_monthly_rainfall_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_05_seasonal_analysis(state_seasonal: pd.DataFrame, out_dir: Path) -> Path:
    """5. Seasonal Rainfall Analysis (Actual vs Normal across 4 meteorological seasons)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(state_seasonal))
    width = 0.35

    ax.bar(x - width/2, state_seasonal["actual_mm"], width, label="Actual Rainfall (mm)", color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, state_seasonal["normal_mm"], width, label="Normal Benchmark (mm)", color="#FB8C00", edgecolor="black", linewidth=0.5)

    ax.set_title("5. Seasonal Rainfall Analysis: Actual vs Normal Performance", weight="bold", pad=15)
    ax.set_ylabel("State Total Rainfall (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(state_seasonal["season_display"], weight="bold")
    ax.legend(frameon=True)

    for i, row in state_seasonal.iterrows():
        pct = row["pct_deviation"]
        sign = "+" if pct >= 0 else ""
        ax.text(i, max(row["actual_mm"], row["normal_mm"]) + 25, f"{sign}{pct:.1f}%",
                ha="center", weight="bold", color="#2E7D32" if pct >= 0 else "#C62828", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "05_seasonal_rainfall_analysis.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_06_top10_highest(dist_summary: pd.DataFrame, out_dir: Path) -> Path:
    """6. Top 10 Highest Rainfall Districts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = dist_summary.head(10).sort_values(by="total_actual", ascending=True)

    colors = sns.color_palette("Greens_d", n_colors=10)
    bars = ax.barh(top10["district"], top10["total_actual"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_title("6. Top 10 Highest Rainfall Districts (2024–2025)", weight="bold", pad=15)
    ax.set_xlabel("Total Actual Rainfall (mm)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 20, bar.get_y() + bar.get_height() / 2, f"{w:.1f} mm",
                va="center", ha="left", fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = out_dir / "06_top_10_highest_rainfall_districts.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_07_top10_lowest(dist_summary: pd.DataFrame, out_dir: Path) -> Path:
    """7. Top 10 Lowest Rainfall Districts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom10 = dist_summary.tail(10).sort_values(by="total_actual", ascending=True)

    colors = sns.color_palette("Reds_d", n_colors=10)
    bars = ax.barh(bottom10["district"], bottom10["total_actual"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_title("7. Top 10 Lowest Rainfall Districts (2024–2025)", weight="bold", pad=15)
    ax.set_xlabel("Total Actual Rainfall (mm)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height() / 2, f"{w:.1f} mm",
                va="center", ha="left", fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = out_dir / "07_top_10_lowest_rainfall_districts.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_08_rainfall_heatmap(df_dist: pd.DataFrame, out_dir: Path) -> Path:
    """8. District-wise Rainfall Heatmap (Districts x Months matrix)."""
    fig, ax = plt.subplots(figsize=(14, 11))

    # Pivot: district on rows, month on columns
    month_order = sorted(df_dist["month_name"].unique(), key=lambda m: pd.to_datetime(m, format="%b %Y"))

    # Order districts by total actual rainfall
    district_order = df_dist.groupby("district")["actual_mm"].sum().sort_values(ascending=False).index

    pivot_df = df_dist.pivot(index="district", columns="month_name", values="actual_mm")
    pivot_df = pivot_df.reindex(index=district_order, columns=month_order)

    sns.heatmap(
        pivot_df,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=0.4,
        cbar_kws={"label": "Actual Rainfall (mm)"},
        ax=ax,
        annot_kws={"size": 7}
    )

    ax.set_title("8. District-wise Monthly Rainfall Heatmap (Jun 2024 – May 2025)", weight="bold", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("District (Ranked by Total Rainfall)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = out_dir / "08_district_wise_rainfall_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_09_deviation_analysis(dist_summary: pd.DataFrame, out_dir: Path) -> Path:
    """9. Rainfall Deviation Analysis (Diverging bar chart of surplus/deficit percentage)."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ordered = dist_summary.sort_values(by="pct_deviation", ascending=True)

    colors = ["#2E7D32" if val >= 0 else "#C62828" for val in ordered["pct_deviation"]]
    bars = ax.barh(ordered["district"], ordered["pct_deviation"], color=colors, edgecolor="black", linewidth=0.4)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title("9. District Rainfall Deviation Analysis (% Departure from Normal)", weight="bold", pad=15)
    ax.set_xlabel("Percentage Deviation (%)")
    ax.set_ylabel("District")

    for bar in bars:
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 2 if w >= 0 else -2
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}%",
                va="center", ha=ha, fontsize=8, color="#222222")

    plt.tight_layout()
    out_path = out_dir / "09_rainfall_deviation_analysis.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_10_seasonal_contribution(state_seasonal: pd.DataFrame, out_dir: Path) -> Path:
    """10. Seasonal Contribution Percentage (Donut Chart)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["#1976D2", "#00897B", "#7E57C2", "#FFA000"]
    wedges, texts, autotexts = ax.pie(
        state_seasonal["actual_mm"],
        labels=state_seasonal["season_display"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )

    for at in autotexts:
        at.set_color("white")
        at.set_weight("bold")
        at.set_fontsize(11)

    total_rf = state_seasonal["actual_mm"].sum()
    ax.text(0, 0, f"Total\n{total_rf:.1f} mm\n(100%)", ha="center", va="center", fontsize=12, weight="bold", color="#333333")

    ax.set_title("10. Seasonal Contribution to Total Annual Rainfall (2024–2025)", weight="bold", pad=15)
    plt.tight_layout()

    out_path = out_dir / "10_seasonal_contribution_percentage.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def generate_interactive_plotly(dist_summary: pd.DataFrame, monthly_summary: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Generate interactive HTML charts for dashboard exploration."""
    fig_dist = px.bar(
        dist_summary,
        x="district",
        y=["total_actual", "total_normal"],
        barmode="group",
        title="Interactive District Comparison: Actual vs Normal Rainfall (2024–2025)",
        labels={"value": "Rainfall (mm)", "district": "District", "variable": "Metric"},
        color_discrete_map={"total_actual": "#1E88E5", "total_normal": "#FFB74D"},
        hover_data={"pct_deviation": ":.2f%"}
    )
    fig_dist.update_layout(xaxis_tickangle=-45, legend_title_text="")
    p1 = out_dir / "interactive_district_comparison.html"
    fig_dist.write_html(str(p1))

    fig_month = go.Figure()
    fig_month.add_trace(go.Scatter(
        x=monthly_summary["month_name"],
        y=monthly_summary["actual_mm"],
        mode="lines+markers",
        name="Actual Rainfall (mm)",
        line=dict(color="#1E88E5", width=3)
    ))
    fig_month.add_trace(go.Scatter(
        x=monthly_summary["month_name"],
        y=monthly_summary["normal_mm"],
        mode="lines+markers",
        name="Normal Rainfall (mm)",
        line=dict(color="#FF7043", width=2, dash="dash")
    ))
    fig_month.update_layout(
        title="Interactive Monthly Rainfall Progression (Jun 2024 – May 2025)",
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        hovermode="x unified"
    )
    p2 = out_dir / "interactive_monthly_trend.html"
    fig_month.write_html(str(p2))

    return p1, p2


# ==============================================================================
# 3. MASTER VISUALIZATION EXECUTION
# ==============================================================================

# ==============================================================================
# MULTI-YEAR & YEAR-OVER-YEAR (YoY) VISUALIZATIONS (FIGURES 14, 15, 16)
# ==============================================================================

def plot_14_yoy_district_annual_comparison(df_annual: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 14: District-Wise Year-over-Year (YoY) Total Rainfall Comparison (2023-24 vs 2024-25)."""
    fig, ax = plt.subplots(figsize=(14, 18), dpi=300)
    
    # Sort by 2024-25 rainfall descending
    df_sorted = df_annual.sort_values("actual_2024_25", ascending=True).copy()
    
    y = np.arange(len(df_sorted))
    bar_height = 0.38

    ax.barh(y - bar_height / 2, df_sorted["actual_2023_24"], height=bar_height,
            label="2023–2024 Actual Rainfall (mm)", color="#3498db", alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.barh(y + bar_height / 2, df_sorted["actual_2024_25"], height=bar_height,
            label="2024–2025 Actual Rainfall (mm)", color="#2ecc71", alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["district"], fontsize=10, fontweight="medium")
    ax.set_xlabel("Total Annual Rainfall (mm)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Tamil Nadu Districts: Multi-Year Annual Rainfall Comparison\n2023–2024 vs 2024–2025 (Official Government Report Data)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    # Highlight State Average
    for idx, d_name in enumerate(df_sorted["district"]):
        if "State" in d_name:
            ax.get_yticklabels()[idx].set_color("#e74c3c")
            ax.get_yticklabels()[idx].set_fontweight("bold")

    fig.tight_layout()
    output_path = figures_dir / "14_yoy_district_annual_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_15_yoy_common_months_trend(df_monthly: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 15: Common Months YoY Comparison (Nov, Dec, Jan, Feb, Mar, Apr, May) for State Average."""
    state_m = df_monthly[df_monthly["district"] == "Tamil Nadu (State)"].copy()
    month_order = ["November", "December", "January", "February", "March", "April", "May"]
    state_m["month"] = pd.Categorical(state_m["month"], categories=month_order, ordered=True)
    state_m = state_m.sort_values("month")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    x = np.arange(len(state_m))

    ax.plot(x, state_m["actual_2023_24"], marker="o", linewidth=2.5, markersize=8,
            color="#2980b9", linestyle="--", label="2023–2024 Actual Rainfall (mm)")
    ax.plot(x, state_m["actual_2024_25"], marker="s", linewidth=2.5, markersize=8,
            color="#27ae60", linestyle="-", label="2024–2025 Actual Rainfall (mm)")

    # Fill difference
    ax.fill_between(x, state_m["actual_2023_24"], state_m["actual_2024_25"],
                    where=(state_m["actual_2024_25"] >= state_m["actual_2023_24"]),
                    interpolate=True, color="#2ecc71", alpha=0.25, label="2024–25 Higher (Convective/Summer Gain)")
    ax.fill_between(x, state_m["actual_2023_24"], state_m["actual_2024_25"],
                    where=(state_m["actual_2024_25"] < state_m["actual_2023_24"]),
                    interpolate=True, color="#e74c3c", alpha=0.25, label="2023–24 Higher (Nov/Jan Flood Peaks)")

    for i, (_, row) in enumerate(state_m.iterrows()):
        ax.annotate(f"{row['actual_2023_24']:.1f}", (i, row['actual_2023_24']),
                    textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color="#2980b9", fontweight="bold")
        ax.annotate(f"{row['actual_2024_25']:.1f}", (i, row['actual_2024_25']),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#27ae60", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(state_m["month"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Monthly Actual Rainfall (mm)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Tamil Nadu State Average: Common Months Year-over-Year (YoY) Trajectory\nComparison Across 7 Consecutive Meteorological Months",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    fig.tight_layout()
    output_path = figures_dir / "15_yoy_common_months_trend.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_16_district_yoy_growth_deficit(df_annual: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 16: District YoY Percentage Growth/Deficit Diverging Bar Chart."""
    df_dist = df_annual[df_annual["district"] != "Tamil Nadu (State)"].sort_values("yoy_growth_pct", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(14, 16), dpi=300)
    y = np.arange(len(df_dist))
    colors = ["#27ae60" if val >= 0 else "#e74c3c" for val in df_dist["yoy_growth_pct"]]

    bars = ax.barh(y, df_dist["yoy_growth_pct"], color=colors, height=0.65, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-")

    # Add State Average reference line
    state_pct = df_annual[df_annual["district"] == "Tamil Nadu (State)"]["yoy_growth_pct"].values[0]
    ax.axvline(state_pct, color="#8e44ad", linewidth=1.5, linestyle="--", label=f"State Average YoY: {state_pct:+.1f}%")

    ax.set_yticks(y)
    ax.set_yticklabels(df_dist["district"], fontsize=10, fontweight="medium")
    ax.set_xlabel("Year-over-Year (YoY) Rainfall Growth / Deficit (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Tamil Nadu Districts: YoY Rainfall Percentage Deviation (2024–25 vs 2023–24)\nGreen: Rainfall Surge | Red: Rainfall Deficit",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    # Value labels
    for bar, val in zip(bars, df_dist["yoy_growth_pct"]):
        x_pos = bar.get_width() + (1.0 if val >= 0 else -1.0)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}%",
                va="center", ha=ha, fontsize=8.5, fontweight="bold",
                color="#1b5e20" if val >= 0 else "#b71c1c")

    fig.tight_layout()
    output_path = figures_dir / "16_district_yoy_growth_deficit.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_interactive_yoy_plotly(df_annual: pd.DataFrame, df_monthly: pd.DataFrame, figures_dir: Path) -> Path:
    """Generates standalone interactive Plotly YoY comparison visualization."""
    df_sorted = df_annual.sort_values("actual_2024_25", ascending=False).copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted["district"],
        y=df_sorted["actual_2023_24"],
        name="2023–24 Actual (mm)",
        marker_color="#3498db"
    ))
    fig.add_trace(go.Bar(
        x=df_sorted["district"],
        y=df_sorted["actual_2024_25"],
        name="2024–25 Actual (mm)",
        marker_color="#2ecc71"
    ))

    fig.update_layout(
        title="<b>Tamil Nadu Multi-Year Rainfall Interactive Explorer (2023–24 vs 2024–25)</b>",
        xaxis_title="District",
        yaxis_title="Total Annual Rainfall (mm)",
        barmode="group",
        template="plotly_white",
        legend=dict(x=0.8, y=0.98),
        hovermode="x unified",
        height=650
    )

    out_file = figures_dir / "interactive_yoy_district_comparison.html"
    fig.write_html(str(out_file))
    return out_file


def generate_multiyear_visualizations(figures_dir: Path = DEFAULT_FIGURES_DIR) -> Dict[str, Path]:
    """Generates all multi-year YoY comparative figures (14, 15, 16 + Interactive)."""
    annual_path = Path("data/processed/multiyear_district_annual_comparison.csv")
    monthly_path = Path("data/processed/multiyear_district_monthly_comparison.csv")
    if not annual_path.exists() or not monthly_path.exists():
        from src.multiyear_analysis import run_multiyear_analysis
        run_multiyear_analysis()

    df_annual = pd.read_csv(annual_path)
    df_monthly = pd.read_csv(monthly_path)

    gen = {}
    gen["14_yoy_annual"] = plot_14_yoy_district_annual_comparison(df_annual, figures_dir)
    gen["15_yoy_monthly"] = plot_15_yoy_common_months_trend(df_monthly, figures_dir)
    gen["16_yoy_growth"] = plot_16_district_yoy_growth_deficit(df_annual, figures_dir)
    gen["interactive_yoy"] = generate_interactive_yoy_plotly(df_annual, df_monthly, figures_dir)

    print("\n" + "=" * 80)
    print("MULTI-YEAR: ALL YoY COMPARATIVE VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 80)
    for name, p in gen.items():
        print(f"  • {name:<25}: {p.name}")
    print(f"Output Directory: {figures_dir.resolve()}\n")
    return gen


def generate_all_step4_visualizations(
    district_path: Path = DEFAULT_DISTRICT_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    figures_dir: Path = DEFAULT_FIGURES_DIR
) -> Dict[str, Path]:
    """Generate all 10 required Step 4 visualizations plus interactive artifacts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    df_dist, df_state = load_cleaned_data(district_path, state_path)

    dist_summary = analyze_district_rainfall(df_dist)
    monthly_summary = analyze_monthly_trends(df_state, df_dist)
    dist_seasonal, state_seasonal = analyze_seasonal_rainfall(df_dist, df_state)

    generated = {}
    generated["01_monthly_trend"] = plot_01_monthly_trend(monthly_summary, figures_dir)
    generated["02_district_totals"] = plot_02_district_totals(dist_summary, figures_dir)
    generated["03_actual_vs_normal"] = plot_03_actual_vs_normal(dist_summary, figures_dir)
    generated["04_monthly_distribution"] = plot_04_monthly_distribution(df_dist, figures_dir)
    generated["05_seasonal_analysis"] = plot_05_seasonal_analysis(state_seasonal, figures_dir)
    generated["06_top10_highest"] = plot_06_top10_highest(dist_summary, figures_dir)
    generated["07_top10_lowest"] = plot_07_top10_lowest(dist_summary, figures_dir)
    generated["08_rainfall_heatmap"] = plot_08_rainfall_heatmap(df_dist, figures_dir)
    generated["09_deviation_analysis"] = plot_09_deviation_analysis(dist_summary, figures_dir)
    generated["10_seasonal_contribution"] = plot_10_seasonal_contribution(state_seasonal, figures_dir)
    generated["interactive_dist"], generated["interactive_month"] = generate_interactive_plotly(dist_summary, monthly_summary, figures_dir)

    print("\n" + "=" * 80)
    print("STEP 4: ALL 10 REQUIRED VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 80)
    for name, p in generated.items():
        print(f"  • {name:<30}: {p.name}")
    print(f"Output Directory: {figures_dir.resolve()}\n")

    return generated


if __name__ == "__main__":
    generate_all_step4_visualizations()
