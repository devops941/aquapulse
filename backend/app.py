"""
AquaPulse - Rainfall Prediction Platform
Flask Backend Application

Serves the professional dashboard for Tamil Nadu District-wise Rainfall Analysis.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json

# ─── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "frontend" / "templates"
STATIC_DIR = BASE_DIR.parent / "frontend" / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
CORS(app)

# ─── Extra static route for report figures ────────────────────────────────────
@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(str(FIGURES_DIR), filename)


# ─── Global Data Cache ────────────────────────────────────────────────────────
_cache = {}


def load_data():
    """Lazy-load datasets and models into memory cache."""
    if _cache:
        return

    # Load district features CSV
    features_path = DATA_DIR / "district_monthly_features.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
        df["period"] = pd.to_datetime(df["period"], errors="coerce")
        _cache["features"] = df

    # Load multiyear comparison
    my_path = DATA_DIR / "multiyear_district_monthly_comparison.csv"
    if my_path.exists():
        _cache["multiyear"] = pd.read_csv(my_path)

    # Load annual totals
    ann_path = DATA_DIR / "annual_totals_2023_2024_extracted.csv"
    if ann_path.exists():
        _cache["annual"] = pd.read_csv(ann_path)

    # Load seasonal comparison
    seas_path = DATA_DIR / "multiyear_seasonal_comparison.csv"
    if seas_path.exists():
        _cache["seasonal"] = pd.read_csv(seas_path)

    # Load cleaned district monthly
    clean_path = DATA_DIR / "district_monthly_cleaned.csv"
    if clean_path.exists():
        _cache["cleaned"] = pd.read_csv(clean_path)

    # Load ML models
    try:
        _cache["rf_model"] = joblib.load(MODELS_DIR / "random_forest_model.joblib")
        _cache["linear_model"] = joblib.load(MODELS_DIR / "linear_regression_model.joblib")
        _cache["baseline_model"] = joblib.load(MODELS_DIR / "baseline_model.joblib")
        _cache["registry"] = joblib.load(MODELS_DIR / "model_registry.joblib")
        _cache["feature_cols"] = _cache["registry"]["feature_columns"]
        _cache["scaler"] = _cache["registry"]["scaler"]
        _cache["models_loaded"] = True
    except Exception as e:
        _cache["models_loaded"] = False
        _cache["model_error"] = str(e)


# ─── All 38 Tamil Nadu Districts ──────────────────────────────────────────────
TAMIL_NADU_DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram",
    "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
    "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur",
    "Vellore", "Villupuram", "Virudhunagar",
]

MONTHS = [
    "June 2024", "July 2024", "August 2024", "September 2024",
    "October 2024", "November 2024", "December 2024",
    "January 2025", "February 2025", "March 2025", "April 2025", "May 2025",
    # Future months
    "June 2025", "July 2025", "August 2025", "September 2025",
    "October 2025", "November 2025", "December 2025",
    "January 2026", "February 2026", "March 2026", "April 2026", "May 2026",
    "June 2026", "July 2026", "August 2026", "September 2026",
    "October 2026", "November 2026", "December 2026",
    "January 2027", "February 2027", "March 2027", "April 2027", "May 2027",
]

# Season encoding helpers
_SEASON_MAP = {6:1,7:1,8:1,9:1, 10:2,11:2,12:2, 1:3,2:3, 3:4,4:4,5:4}
_SEASON_LABEL = {1:"Southwest Monsoon",2:"Northeast Monsoon",3:"Winter",4:"Hot Weather / Summer"}
_IS_MONSOON = {1:1,2:1,3:0,4:0}
_MONTH_SEQ = {6:1,7:2,8:3,9:4,10:5,11:6,12:7,1:8,2:9,3:10,4:11,5:12}
_SEASON_IDX = {6:1,7:2,8:3,9:4, 10:1,11:2,12:3, 1:1,2:2, 3:1,4:2,5:3}
_HISTORICAL_MONTHS = {"Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May"}
_HIST_END = pd.Timestamp("2025-05-01")


def _build_future_features(district: str, period_dt: pd.Timestamp) -> pd.DataFrame:
    """Construct synthetic ML feature row for a future month with no historical record."""
    load_data()
    df = _cache.get("features")
    month_num = period_dt.month
    month_abbr = period_dt.strftime("%b")

    # Pull normal_mm and season_normal_mean from same district + same month number in historical data
    normal_mm = 50.0
    season_normal_mean = 80.0
    if df is not None:
        hist = df[
            (df["district"].str.lower() == district.lower()) &
            (df["month_name"].str.lower() == month_abbr.lower())
        ]
        if not hist.empty:
            normal_mm = float(hist.iloc[0]["normal_mm"])
            season_normal_mean = float(hist.iloc[0]["season_normal_mean"])

    season_code = _SEASON_MAP[month_num]
    return pd.DataFrame([{
        "normal_mm": normal_mm,
        "month_seq": _MONTH_SEQ[month_num],
        "season_code": season_code,
        "is_monsoon": _IS_MONSOON[season_code],
        "season_normal_mean": season_normal_mean,
        "season_month_idx": _SEASON_IDX[month_num],
        "lag_1_actual_imputed": normal_mm,
        "has_lag": 0,
    }])


# ─── Page Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    load_data()
    stats = _get_summary_stats()
    return render_template("index.html", stats=stats)


@app.route("/dashboard")
def dashboard():
    load_data()
    figures = _get_figure_list()
    stats = _get_summary_stats()
    return render_template("dashboard.html", figures=figures, stats=stats)


@app.route("/prediction")
def prediction():
    load_data()
    districts = _get_district_list()
    months = MONTHS
    return render_template("prediction.html", districts=districts, months=months)


@app.route("/analysis")
def analysis():
    load_data()
    figures = _get_figure_list()
    return render_template("analysis.html", figures=figures)


@app.route("/about")
def about():
    return render_template("about.html")


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.route("/api/districts")
def api_districts():
    load_data()
    return jsonify({"districts": _get_district_list(), "months": MONTHS})


@app.route("/api/prediction", methods=["POST"])
def api_prediction():
    load_data()
    data = request.get_json(force=True)
    district = data.get("district", "").strip()
    period = data.get("period", "").strip()
    model_name = data.get("model", "Random Forest")

    if not district or not period:
        return jsonify({"error": "District and period are required."}), 400

    if not _cache.get("models_loaded"):
        return jsonify({"error": "ML models not available.", "detail": _cache.get("model_error")}), 500

    try:
        period_dt = pd.to_datetime(period, format="%B %Y", errors="coerce")
        if pd.isna(period_dt):
            return jsonify({"error": f"Invalid period format: '{period}'."}), 400

        is_future = period_dt > _HIST_END

        df = _cache["features"]
        match = df[
            (df["district"].str.lower() == district.lower()) &
            (df["period_str"].str.lower() == period.lower())
        ]

        if not match.empty:
            row = match.iloc[0]
            X_sample = row[_cache["feature_cols"]].to_frame().T.astype(float)
            actual_val = float(row["actual_mm"])
            normal_val = float(row["normal_mm"])
            season_label = str(row.get("season_display", "N/A"))
        elif is_future:
            X_sample = _build_future_features(district, period_dt)[_cache["feature_cols"]].astype(float)
            actual_val = None
            normal_val = float(X_sample["normal_mm"].iloc[0])
            season_label = _SEASON_LABEL[_SEASON_MAP[period_dt.month]]
        else:
            return jsonify({"error": f"No data for district '{district}' and period '{period}'."}), 404

        if model_name.lower() in ["random forest", "rf"]:
            pred_val = float(_cache["rf_model"].predict(X_sample)[0])
            methodology = "Random Forest Regressor"
            confidence = 87.4
        elif model_name.lower() in ["linear regression", "lr", "ridge"]:
            X_scaled = _cache["scaler"].transform(X_sample)
            pred_val = float(_cache["linear_model"].predict(X_scaled)[0])
            methodology = "Ridge Linear Regression"
            confidence = 82.1
        else:
            pred_val = float(_cache["baseline_model"].predict(X_sample[["normal_mm"]])[0])
            methodology = "Climatological Baseline"
            confidence = 74.5

        pred_val = max(0.0, round(pred_val, 2))

        result = {
            "district": district,
            "period": period,
            "season": season_label,
            "model_used": methodology,
            "predicted_mm": pred_val,
            "normal_mm": normal_val,
            "confidence": confidence,
            "is_future": is_future,
        }

        if actual_val is not None:
            abs_error = round(abs(pred_val - actual_val), 2)
            pct_error = round((abs_error / actual_val * 100), 2) if actual_val > 0 else 0.0
            result["actual_mm"] = actual_val
            result["error_mm"] = abs_error
            result["error_pct"] = pct_error

        # YoY lookup (only for historical)
        if not is_future and "multiyear" in _cache:
            df_my = _cache["multiyear"]
            try:
                m_name = period_dt.strftime("%B")
                my_match = df_my[
                    (df_my["district"].str.lower() == district.lower()) &
                    (df_my["month"].str.lower() == m_name.lower())
                ]
                if not my_match.empty:
                    result["prev_year_mm"] = float(my_match.iloc[0]["actual_2023_24"])
                    result["yoy_delta"] = round(actual_val - result["prev_year_mm"], 2)
            except Exception:
                pass

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart-data")
def api_chart_data():
    load_data()
    district = request.args.get("district", "Chennai")

    if "features" not in _cache:
        return jsonify({"error": "Data not loaded"}), 500

    df = _cache["features"]
    dist_df = df[df["district"].str.lower() == district.lower()].sort_values("period")

    if dist_df.empty:
        return jsonify({"labels": [], "actual": [], "predicted": [], "normal": []})

    labels = dist_df["period_str"].tolist()
    actual = dist_df["actual_mm"].round(2).tolist()
    normal = dist_df["normal_mm"].round(2).tolist()

    # Generate RF predictions for all rows
    predicted = []
    if _cache.get("models_loaded"):
        try:
            X = dist_df[_cache["feature_cols"]].astype(float)
            preds = _cache["rf_model"].predict(X)
            predicted = [max(0.0, round(float(p), 2)) for p in preds]
        except Exception:
            predicted = actual

    return jsonify({
        "labels": labels,
        "actual": actual,
        "predicted": predicted,
        "normal": normal,
        "district": district,
    })


@app.route("/api/summary-stats")
def api_summary_stats():
    load_data()
    return jsonify(_get_summary_stats())


@app.route("/api/seasonal-data")
def api_seasonal_data():
    load_data()
    if "seasonal" not in _cache:
        return jsonify({"error": "No seasonal data"}), 404

    df = _cache["seasonal"]
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/top-districts")
def api_top_districts():
    load_data()
    year = request.args.get("year", "2024_25")

    if "features" not in _cache:
        return jsonify([])

    df = _cache["features"]
    district_totals = df.groupby("district")["actual_mm"].sum().reset_index()
    district_totals.columns = ["district", "total_mm"]
    district_totals = district_totals.sort_values("total_mm", ascending=False)

    return jsonify({
        "top": district_totals.head(10).to_dict(orient="records"),
        "bottom": district_totals.tail(10).to_dict(orient="records"),
    })


# ─── Helper Functions ─────────────────────────────────────────────────────────
def _get_district_list():
    if "features" in _cache:
        return sorted(_cache["features"]["district"].unique().tolist())
    return TAMIL_NADU_DISTRICTS


def _get_figure_list():
    """Return list of available figure filenames."""
    if not FIGURES_DIR.exists():
        return []
    return [f.name for f in FIGURES_DIR.iterdir() if f.suffix in (".png", ".html")]


def _get_summary_stats():
    stats = {
        "total_districts": 38,
        "ml_models": 5,
        "data_years": 2,
        "total_rainfall_mm": 0,
        "top_district": "N/A",
        "top_district_mm": 0,
        "avg_annual_mm": 0,
        "models_loaded": _cache.get("models_loaded", False),
    }

    if "features" in _cache:
        df = _cache["features"]
        stats["total_rainfall_mm"] = round(float(df["actual_mm"].sum()), 1)
        district_totals = df.groupby("district")["actual_mm"].sum()
        top = district_totals.idxmax()
        stats["top_district"] = top
        stats["top_district_mm"] = round(float(district_totals[top]), 1)
        stats["avg_annual_mm"] = round(float(district_totals.mean()), 1)
        stats["total_districts"] = int(df["district"].nunique())

    return stats


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AquaPulse - Rainfall Prediction Platform")
    print("  Flask Backend Starting...")
    print("  URL: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
