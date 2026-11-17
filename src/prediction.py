"""
Rainfall Prediction Module
Tamil Nadu District-wise Rainfall Analysis and Prediction (2024–2025)

Author: Senior Python Data Scientist & Machine Learning Engineer
Path: src/prediction.py
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DATA_PATH = WORKSPACE_ROOT / "data" / "processed" / "district_monthly_features.csv"
MODELS_DIR = WORKSPACE_ROOT / "models"


class RainfallPredictor:
    """Inference engine for district rainfall prediction using serialized models."""

    def __init__(self, models_dir: Path = MODELS_DIR, features_path: Path = FEATURES_DATA_PATH):
        self.models_dir = models_dir
        self.features_path = features_path

        # Load models
        self.baseline_model = joblib.load(models_dir / "baseline_model.joblib")
        self.linear_model = joblib.load(models_dir / "linear_regression_model.joblib")
        self.rf_model = joblib.load(models_dir / "random_forest_model.joblib")
        self.registry = joblib.load(models_dir / "model_registry.joblib")

        self.feature_cols = self.registry["feature_columns"]
        self.scaler = self.registry["scaler"]

        # Load features dataset for retrospective context
        self.df_features = pd.read_csv(features_path)
        self.df_features["period"] = pd.to_datetime(self.df_features["period"])
        self.available_districts = sorted(self.df_features["district"].unique().tolist())
        self.available_periods = self.df_features["period_str"].unique().tolist()

    def get_district_list(self) -> List[str]:
        return self.available_districts

    def get_period_list(self) -> List[str]:
        return self.available_periods

    def predict(
        self,
        district_name: str,
        period_str: str,
        model_name: str = "Random Forest"
    ) -> Dict[str, Any]:
        """
        Generate experimental rainfall estimate for a specified district and available period.

        Returns:
            Dictionary containing prediction, actual benchmark, model methodology,
            and explicit scientific caveats.
        """
        match = self.df_features[
            (self.df_features["district"].str.lower() == district_name.strip().lower()) &
            (self.df_features["period_str"].str.lower() == period_str.strip().lower())
        ]

        if match.empty:
            raise ValueError(
                f"No record found for District '{district_name}' and Period '{period_str}'. "
                f"Valid periods are strictly between June 2024 and May 2025. "
                f"No future predictions are fabricated for unavailable months."
            )

        row = match.iloc[0]
        X_sample = row[self.feature_cols].to_frame().T

        # Model Inference
        if model_name.lower() in ["random forest", "rf"]:
            pred_val = float(self.rf_model.predict(X_sample)[0])
            methodology = "Random Forest Regressor (Ensemble of 100 decision trees with feature interaction)"
        elif model_name.lower() in ["linear regression", "ridge", "lr"]:
            X_scaled = self.scaler.transform(X_sample)
            pred_val = float(self.linear_model.predict(X_scaled)[0])
            methodology = "Ridge Regularized Linear Regression (Normalized multi-feature regression)"
        elif model_name.lower() in ["baseline", "normal", "benchmark"]:
            pred_val = float(self.baseline_model.predict(X_sample[["normal_mm"]])[0])
            methodology = "Climatological Baseline (Expected historical 30-year normal benchmark)"
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Choose from 'Random Forest', 'Linear Regression', 'Baseline'.")

        pred_val = max(0.0, round(pred_val, 2))  # Non-negative physical constraint
        actual_val = float(row["actual_mm"])
        normal_val = float(row["normal_mm"])
        abs_error = round(abs(pred_val - actual_val), 2)
        pct_error = round((abs_error / actual_val * 100), 2) if actual_val > 0 else 0.0

        # Determine validation split status
        is_test_set = row["month_seq"] > 8
        eval_tier = "Holdout Test Set (Unseen during training)" if is_test_set else "Historical Training Period"

        # Multi-Year Prior Lookup (if available from 2023-24 historical screenshots)
        prev_year_val = None
        yoy_delta = None
        yoy_pct = None
        multiyear_path = WORKSPACE_ROOT / "data" / "processed" / "multiyear_district_monthly_comparison.csv"
        if multiyear_path.exists():
            try:
                df_my = pd.read_csv(multiyear_path)
                m_name = pd.to_datetime(row["period_str"]).strftime("%B")
                my_match = df_my[
                    (df_my["district"].str.lower() == row["district"].lower()) &
                    (df_my["month"].str.lower() == m_name.lower())
                ]
                if not my_match.empty:
                    prev_year_val = float(my_match.iloc[0]["actual_2023_24"])
                    yoy_delta = round(actual_val - prev_year_val, 2)
                    yoy_pct = round((yoy_delta / prev_year_val * 100), 2) if prev_year_val > 0 else 0.0
            except Exception:
                pass

        disclaimer = (
            "IMPORTANT METEOROLOGICAL DISCLAIMER: "
            "This prediction is an experimental retrospective estimate derived from actual meteorological observations "
            "(June 2024 – May 2025, cross-referenced with 2023–2024 historical government records). "
            "It does not constitute an operational numerical weather forecast."
        )

        return {
            "district": row["district"],
            "period": row["period_str"],
            "season": row["season_display"],
            "model_used": model_name,
            "methodology": methodology,
            "evaluation_tier": eval_tier,
            "predicted_rainfall_mm": pred_val,
            "historical_actual_mm": actual_val,
            "normal_benchmark_mm": normal_val,
            "absolute_error_mm": abs_error,
            "error_percentage": pct_error,
            "prev_year_actual_mm": prev_year_val,
            "yoy_delta_mm": yoy_delta,
            "yoy_pct": yoy_pct,
            "input_features": {col: float(row[col]) for col in self.feature_cols},
            "disclaimer": disclaimer,
        }


def format_prediction_output(res: Dict[str, Any]) -> str:
    """Format prediction dictionary for CLI or report display."""
    lines = [
        "=" * 75,
        "TAMIL NADU RAINFALL EXPERIMENTAL PREDICTION REPORT",
        "=" * 75,
        f"District            : {res['district']}",
        f"Prediction Period   : {res['period']} ({res['season']})",
        f"Model Applied       : {res['model_used']}",
        f"Methodology         : {res['methodology']}",
        f"Evaluation Context  : {res['evaluation_tier']}",
        "-" * 75,
        f"• EXPERIMENTAL PREDICTED RAINFALL : {res['predicted_rainfall_mm']:.2f} mm",
        f"• HISTORICAL ACTUAL RECORDED      : {res['historical_actual_mm']:.2f} mm",
        f"• NORMAL HISTORICAL BENCHMARK     : {res['normal_benchmark_mm']:.2f} mm",
        f"• ESTIMATE ERROR (|PRED - ACTUAL|): {res['absolute_error_mm']:.2f} mm ({res['error_percentage']:.1f}%)",
    ]
    if res.get("prev_year_actual_mm") is not None:
        lines.append(f"• PRIOR YEAR RECORD (2023–24)     : {res['prev_year_actual_mm']:.2f} mm")
        lines.append(f"• YEAR-OVER-YEAR CHANGE (24 vs 23): {res['yoy_delta_mm']:+.2f} mm ({res['yoy_pct']:+.1f}%)")

    lines.extend([
        "-" * 75,
        "Input Feature Vector:",
    ])
    for feat, val in res["input_features"].items():
        lines.append(f"  - {feat:<24}: {val}")

    lines.extend([
        "-" * 75,
        "CAVEAT & DISCLAIMER:",
        res["disclaimer"],
        "=" * 75,
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil Nadu Rainfall Prediction Engine")
    parser.add_argument("--district", type=str, default="Chennai", help="District name")
    parser.add_argument("--period", type=str, default="May 2025", help="Period (e.g. 'May 2025')")
    parser.add_argument("--model", type=str, default="Random Forest", help="Model: 'Random Forest', 'Linear Regression', 'Baseline'")

    args = parser.parse_args()
    predictor = RainfallPredictor()
    result = predictor.predict(district_name=args.district, period_str=args.period, model_name=args.model)
    print(format_prediction_output(result))
