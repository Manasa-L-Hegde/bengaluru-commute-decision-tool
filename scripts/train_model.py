"""
train_model.py - Train risk_score + TTI prediction models.
Saves to /models/risk_model.pkl and /models/tti_model.pkl.
Run:  python scripts/train_model.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SLOT_HOUR = {
    "Morning Peak (07-09)": 8, "Midday (11-13)": 12,
    "Evening Peak (17-19)": 18, "Night (21-23)": 22,
}

def main():
    print("=" * 60)
    print("  Model Training - Bengaluru Commute Decision Tool")
    print("=" * 60)

    # --- Load ---
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["hour"] = df["time_slot"].map(SLOT_HOUR)
    print(f"Loaded {len(df):,} rows\n")

    # --- Encode categoricals ---
    le_route = LabelEncoder()
    le_slot  = LabelEncoder()
    df["route_enc"]    = le_route.fit_transform(df["route"])
    df["time_slot_enc"] = le_slot.fit_transform(df["time_slot"])

    features = ["route_enc", "day_of_week", "time_slot_enc",
                "total_rain_mm", "congestion_level"]
    extra_features = ["road_capacity_utilization", "incident_reports",
                      "min_visibility_m", "weather_severity"]
    for ef in extra_features:
        if ef in df.columns:
            features.append(ef)

    X = df[features].fillna(0)

    # ============ Model 1: risk_score ============
    y_risk = df["risk_score"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_risk, test_size=0.2, random_state=42
    )
    rf_risk = RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf_risk.fit(Xr_train, yr_train)
    yr_pred = rf_risk.predict(Xr_test)

    mae_r = mean_absolute_error(yr_test, yr_pred)
    r2_r  = r2_score(yr_test, yr_pred)
    print("--- Risk Score Model ---")
    print(f"  Features : {features}")
    print(f"  Train    : {len(Xr_train):,}  |  Test: {len(Xr_test):,}")
    print(f"  MAE      : {mae_r:.3f}")
    print(f"  R^2      : {r2_r:.4f}")

    importances = pd.Series(rf_risk.feature_importances_, index=features).sort_values(ascending=False)
    print(f"  Feature importance:\n{importances.to_string()}\n")

    # ============ Model 2: travel_time_index ============
    y_tti = df["travel_time_index"]
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X, y_tti, test_size=0.2, random_state=42
    )
    rf_tti = RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf_tti.fit(Xt_train, yt_train)
    yt_pred = rf_tti.predict(Xt_test)

    mae_t = mean_absolute_error(yt_test, yt_pred)
    r2_t  = r2_score(yt_test, yt_pred)
    print("--- TTI Model ---")
    print(f"  MAE      : {mae_t:.4f}")
    print(f"  R^2      : {r2_t:.4f}\n")

    # ============ Save ============
    bundle = {
        "risk_model":   rf_risk,
        "tti_model":    rf_tti,
        "le_route":     le_route,
        "le_slot":      le_slot,
        "features":     features,
        "route_classes": list(le_route.classes_),
        "slot_classes":  list(le_slot.classes_),
    }
    out_path = MODEL_DIR / "risk_model.pkl"
    joblib.dump(bundle, out_path, compress=3)
    print(f"Saved to {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
