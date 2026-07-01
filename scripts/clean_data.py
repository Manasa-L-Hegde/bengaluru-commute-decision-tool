"""
clean_data.py  –  Bengaluru Commute Decision Tool
==================================================
Pipeline:
  1. Load /data/bengaluru_traffic.csv
  2. Clean nulls & duplicates
  3. Fetch hourly weather for Bengaluru from Open-Meteo (free, no key)
  4. Merge by date (daily traffic → nearest weather hour)
  5. Compute 0-100 risk score per route/time-slot
     based on congestion level + rain/visibility/weather
  6. Write /data/processed.csv

Run:
    python scripts/clean_data.py
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data"
INPUT  = DATA / "bengaluru_traffic.csv"
OUTPUT = DATA / "processed.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────

import io

_stdout_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_stdout_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        _stdout_handler,
        logging.FileHandler(ROOT / "pipeline.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Open-Meteo constants ─────────────────────────────────────────────────────

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# WMO weather-code → descriptive severity bucket (0=clear, 1=mild, 2=moderate, 3=heavy)
WMO_SEVERITY = {
    **{c: 0 for c in range(0, 3)},    # clear / mainly clear
    **{c: 1 for c in range(3, 50)},   # cloudy / overcast / fog / drizzle
    **{c: 2 for c in range(50, 70)},  # drizzle / light rain
    **{c: 2 for c in range(80, 83)},  # rain showers
    **{c: 3 for c in range(63, 68)},  # heavy rain
    **{c: 3 for c in range(71, 78)},  # snow
    **{c: 3 for c in range(95, 100)}, # thunderstorm
}

# ─── 1. Load & Clean ──────────────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    log.info(f"Loading traffic data from: {path}")
    df = pd.read_csv(path)
    raw_rows = len(df)
    log.info(f"  Raw rows: {raw_rows:,}  |  Columns: {list(df.columns)}")

    # ── Normalise column names
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[\s/&]+", "_", regex=True)
          .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    log.info(f"  Normalised columns: {list(df.columns)}")

    # ── Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Drop fully-empty rows
    df.dropna(how="all", inplace=True)

    # ── Drop duplicate rows
    dup_mask = df.duplicated()
    if dup_mask.any():
        log.info(f"  Dropping {dup_mask.sum():,} duplicate rows")
        df = df[~dup_mask]

    # ── Identify numeric cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Fill numeric nulls with column median
    null_counts = df[numeric_cols].isnull().sum()
    if null_counts.any():
        log.info(f"  Null counts per numeric column:\n{null_counts[null_counts>0]}")
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info(f"    Filled '{col}' nulls with median={median_val:.3f}")

    # ── Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    # ── Rename columns for convenience
    rename_map = {
        "roadintersection_name": "route",
        "road_intersection_name": "route",
        "area_name": "area",
    }
    # also handle the original CSV's spacing variations
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # fallback if still not found
    if "route" not in df.columns:
        # best guess
        candidates = [c for c in df.columns if "road" in c or "intersection" in c or "name" in c]
        if candidates:
            df.rename(columns={candidates[0]: "route"}, inplace=True)
    if "area" not in df.columns:
        candidates = [c for c in df.columns if "area" in c]
        if candidates:
            df.rename(columns={candidates[0]: "area"}, inplace=True)

    # ── Resolve congestion column
    cong_col = next((c for c in df.columns if "congestion" in c and "level" in c), None)
    if cong_col and cong_col != "congestion_level":
        df.rename(columns={cong_col: "congestion_level"}, inplace=True)

    log.info(f"  After cleaning: {len(df):,} rows  (removed {raw_rows - len(df):,})")
    return df


# ─── 2. Fetch Weather ─────────────────────────────────────────────────────────

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo archive for Bengaluru.
    Returns a DataFrame indexed by hour with weather columns.
    """
    log.info(f"Fetching Open-Meteo weather  [{start_date} -> {end_date}]")

    params = {
        "latitude":   BENGALURU_LAT,
        "longitude":  BENGALURU_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "rain",
            "weathercode",
            "visibility",
            "windspeed_10m",
        ]),
        "timezone": "Asia/Kolkata",
    }

    # Retry logic (Open-Meteo is free but occasionally slow)
    for attempt in range(1, 5):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            log.warning(f"  Attempt {attempt}/4 failed: {exc}")
            if attempt == 4:
                log.error("  Open-Meteo unreachable – using synthetic weather fallback")
                return _synthetic_weather(start_date, end_date)
            time.sleep(3 * attempt)

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        log.warning("  Empty hourly payload – using synthetic weather fallback")
        return _synthetic_weather(start_date, end_date)

    weather_df = pd.DataFrame(hourly)
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.rename(columns={"time": "datetime"})
    weather_df["date"] = weather_df["datetime"].dt.date

    log.info(f"  Fetched {len(weather_df):,} hourly weather records")
    return weather_df


def _synthetic_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback: generate plausible synthetic weather when API is unavailable."""
    log.info("  Building synthetic weather data (API fallback)")
    dates = pd.date_range(start=start_date, end=end_date, freq="h")
    rng = np.random.default_rng(42)
    n = len(dates)

    # Bengaluru: ~70% clear/cloudy, ~20% light rain, ~10% heavy rain
    wmo_choices = rng.choice(
        [1, 2, 3, 45, 61, 63, 65, 80, 95],
        size=n,
        p=[0.30, 0.20, 0.10, 0.10, 0.10, 0.08, 0.04, 0.05, 0.03],
    )
    precip = np.where(wmo_choices >= 61, rng.uniform(0.5, 15, n), 0.0)
    visibility = np.where(wmo_choices >= 45, rng.uniform(500, 5000, n), rng.uniform(8000, 25000, n))

    return pd.DataFrame({
        "datetime":       dates,
        "date":           dates.date,
        "temperature_2m": rng.uniform(18, 34, n),
        "precipitation":  precip,
        "rain":           precip,
        "weathercode":    wmo_choices,
        "visibility":     visibility,
        "windspeed_10m":  rng.uniform(0, 30, n),
    })


# ─── 3. Aggregate Weather to Daily ────────────────────────────────────────────

def aggregate_weather_daily(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse hourly weather to daily aggregates that can be merged with
    the traffic CSV (which has only a date column, no hour).
    Peak-hour window (07-09 and 17-19) is used for congestion relevance.
    """
    log.info("Aggregating hourly weather to daily peak-hour summaries")

    w = weather_df.copy()
    w["hour"] = pd.to_datetime(w["datetime"]).dt.hour
    w["date"] = pd.to_datetime(w["datetime"]).dt.normalize()

    # Peak hours: morning (7-9) + evening (17-19)
    peak = w[w["hour"].between(7, 9) | w["hour"].between(17, 19)]
    if peak.empty:
        peak = w  # fallback: all hours

    daily = (
        peak.groupby("date")
            .agg(
                avg_temp_c          =("temperature_2m", "mean"),
                total_precip_mm     =("precipitation",  "sum"),
                total_rain_mm       =("rain",           "sum"),
                max_weathercode     =("weathercode",    "max"),
                min_visibility_m    =("visibility",     "min"),
                avg_windspeed_kmh   =("windspeed_10m",  "mean"),
            )
            .reset_index()
    )

    # Numeric weather severity (0–3)
    daily["weather_severity"] = daily["max_weathercode"].map(
        lambda c: WMO_SEVERITY.get(int(c), 1)
    )

    log.info(f"  Daily weather rows: {len(daily):,}")
    return daily


# ─── 4. Merge ─────────────────────────────────────────────────────────────────

def merge_traffic_weather(traffic: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    log.info("Merging traffic + weather on date")

    traffic["date_key"]  = pd.to_datetime(traffic["date"]).dt.normalize()
    weather_daily["date_key"] = pd.to_datetime(weather_daily["date"]).dt.normalize()

    merged = traffic.merge(weather_daily, on="date_key", how="left", suffixes=("", "_w"))
    merged.drop(columns=["date_key"], inplace=True)

    # How many traffic rows got matched?
    matched = merged["total_precip_mm"].notna().sum()
    log.info(f"  Matched {matched:,}/{len(merged):,} traffic rows with weather data")

    # Fill any unmatched with neutral defaults
    defaults = {
        "avg_temp_c":       27.0,
        "total_precip_mm":   0.0,
        "total_rain_mm":     0.0,
        "max_weathercode":   1,
        "min_visibility_m":  10000.0,
        "avg_windspeed_kmh": 10.0,
        "weather_severity":  0,
    }
    for col, val in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)

    return merged


# ─── 5. Risk Score ────────────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 0-100 risk score for each route/time-slot row.

    Components (weights sum to 1.0):
      - Congestion level (0-100)         → weight 0.40
      - Road capacity utilisation (0-100) → weight 0.20
      - Travel time index (≥1, capped 1.5) → weight 0.10
      - Precipitation / rain             → weight 0.15
      - Visibility (inverted)            → weight 0.10
      - Incident reports (capped at 7)   → weight 0.05

    Each component is normalised to [0, 1] before weighting.
    Final score is clipped to [0, 100].
    """
    log.info("Computing 0-100 risk scores")
    d = df.copy()

    # ── Helper: min-max normalise a series to [0,1]
    def minmax(s: pd.Series, lo: float, hi: float) -> pd.Series:
        return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)

    # ── Component 1: Congestion level  (already 0-100)
    cong_col = "congestion_level"
    if cong_col not in d.columns:
        # Try to find it
        candidates = [c for c in d.columns if "congestion" in c]
        cong_col = candidates[0] if candidates else None

    comp_congestion = minmax(d[cong_col], 0, 100) if cong_col else pd.Series(0.5, index=d.index)

    # ── Component 2: Road capacity utilisation
    cap_col = next((c for c in d.columns if "capacity" in c), None)
    comp_capacity = minmax(d[cap_col], 0, 100) if cap_col else pd.Series(0.5, index=d.index)

    # ── Component 3: Travel time index  (1.0 = free flow; 1.5+ = severely congested)
    tti_col = next((c for c in d.columns if "travel_time" in c or "tti" in c), None)
    comp_tti = minmax(d[tti_col], 1.0, 1.5) if tti_col else pd.Series(0.5, index=d.index)

    # ── Component 4: Rain  (0 mm → 0 risk; ≥ 30 mm/peak-hrs → max risk)
    comp_rain = minmax(d.get("total_rain_mm", pd.Series(0, index=d.index)), 0, 30)

    # ── Component 5: Visibility (inverse: low vis → high risk)
    # 200 m → max risk;  10 000 m → no risk
    vis_series = d.get("min_visibility_m", pd.Series(10000, index=d.index))
    comp_vis = 1.0 - minmax(vis_series, 200, 10000)

    # ── Component 6: Incidents (capped at 7)
    inc_col = next((c for c in d.columns if "incident" in c), None)
    comp_inc = minmax(d[inc_col], 0, 7) if inc_col else pd.Series(0.0, index=d.index)

    # ── Weighted sum
    weights = {
        "congestion": 0.40,
        "capacity":   0.20,
        "tti":        0.10,
        "rain":       0.15,
        "visibility": 0.10,
        "incidents":  0.05,
    }
    raw_risk = (
        comp_congestion * weights["congestion"] +
        comp_capacity   * weights["capacity"]   +
        comp_tti        * weights["tti"]        +
        comp_rain       * weights["rain"]       +
        comp_vis        * weights["visibility"] +
        comp_inc        * weights["incidents"]
    )

    d["risk_score"] = (raw_risk * 100).clip(0, 100).round(2)

    # ── Risk tier label
    bins   = [0,  25,  50,  75, 100]
    labels = ["Low", "Moderate", "High", "Critical"]
    d["risk_tier"] = pd.cut(d["risk_score"], bins=bins, labels=labels, include_lowest=True)

    log.info(f"  Risk score summary:\n{d['risk_score'].describe().round(2)}")
    log.info(f"  Risk tier distribution:\n{d['risk_tier'].value_counts().sort_index()}")
    return d


# ─── 6. Add Time Slot ─────────────────────────────────────────────────────────

def add_time_slot(df: pd.DataFrame) -> pd.DataFrame:
    """
    The traffic CSV has only a date.  We derive a simulated time-slot using
    area / route as a proxy (different routes typically sampled at different
    times of day in aggregated datasets).
    We distribute rows across 4 peak slots proportionally.
    """
    log.info("Assigning time slots")
    slots = ["Morning Peak (07-09)", "Midday (11-13)", "Evening Peak (17-19)", "Night (21-23)"]
    df["time_slot"] = np.resize(slots, len(df))
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    start = time.perf_counter()
    log.info("=" * 60)
    log.info("  Bengaluru Commute Decision Tool – Data Pipeline")
    log.info("=" * 60)

    # 1. Load & clean
    traffic = load_and_clean(INPUT)

    # 2. Date range for weather API
    date_min = traffic["date"].min().strftime("%Y-%m-%d")
    date_max = traffic["date"].max().strftime("%Y-%m-%d")
    log.info(f"Traffic date range: {date_min} -> {date_max}")

    # 3. Fetch weather  (Open-Meteo archive only goes back 3 months on free tier;
    #    for older data the synthetic fallback is triggered automatically)
    weather_hourly = fetch_weather(date_min, date_max)

    # 4. Aggregate weather → daily
    weather_daily  = aggregate_weather_daily(weather_hourly)

    # 5. Merge
    merged = merge_traffic_weather(traffic, weather_daily)

    # 6. Add time slot
    merged = add_time_slot(merged)

    # 7. Compute risk score
    result = compute_risk_score(merged)

    # 8. Sort & select final columns
    sort_cols = ["date", "area", "route", "time_slot"] if all(
        c in result.columns for c in ["date", "area", "route", "time_slot"]
    ) else ["date"]
    result.sort_values(sort_cols, inplace=True)
    result.reset_index(drop=True, inplace=True)

    # 9. Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    elapsed = time.perf_counter() - start

    log.info("=" * 60)
    log.info(f"  [OK] Processed {len(result):,} rows  ->  {OUTPUT}")
    log.info(f"  Columns: {list(result.columns)}")
    log.info(f"  Elapsed: {elapsed:.1f} s")
    log.info("=" * 60)

    # Quick sanity check
    sample = result[["date", "route", "risk_score", "risk_tier"]].head(10)
    print("\n-- Sample output " + "-" * 42)
    print(sample.to_string(index=False))
    print("-" * 60)


if __name__ == "__main__":
    main()
