"""
bigquery_dashboard.py  -  Bengaluru Commute Decision Tool
==========================================================
Queries average risk score and travel-time index grouped by
HOUR and ROUTE (and optionally: area, weather severity, risk tier)
from BigQuery, producing dashboard-ready output CSVs.

Pipeline
--------
  1. Authenticate via Application Default Credentials (ADC)  OR
     a service-account JSON key (set env var GCP_KEY_PATH).
  2. Create BigQuery dataset + table if they do not exist.
  3. Upload data/processed.csv  (skip if table already populated).
  4. Run the three dashboard queries:
       Q1  avg_risk + avg_travel_time  by  hour x route
       Q2  hourly trend               by  hour (all routes)
       Q3  route heatmap              by  route x risk_tier
  5. Save results to  data/dashboard/  as CSV files.
  6. Print a formatted summary table.

Authentication quick-start
--------------------------
  Option A  (recommended for local dev)
    gcloud auth application-default login

  Option B  (service account)
    set GCP_KEY_PATH=C:/path/to/your-sa-key.json
    set GCP_PROJECT=your-project-id

  Option C  (offline / no BigQuery)
    set OFFLINE_MODE=1
    The script falls back to pandas and still produces the same outputs.

Requirements
------------
    pip install google-cloud-bigquery google-cloud-bigquery-storage
    pip install pandas pyarrow db-dtypes tqdm
"""

import os
import sys
import logging
import time
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─── Logging ──────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data"
INPUT_CSV = DATA / "processed.csv"
OUT_DIR   = DATA / "dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
) if hasattr(sys.stdout, "buffer") else sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(_stdout),
        logging.FileHandler(ROOT / "bq_dashboard.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GCP_PROJECT   = os.getenv("GCP_PROJECT",  "bengaluru-commute-tool")
BQ_DATASET    = os.getenv("BQ_DATASET",   "commute_data")
BQ_TABLE      = os.getenv("BQ_TABLE",     "traffic_processed")
GCP_KEY_PATH  = os.getenv("GCP_KEY_PATH", "")           # optional SA key
OFFLINE_MODE  = os.getenv("OFFLINE_MODE", "0") == "1"   # local pandas fallback
LOCATION      = "US"                                     # BQ dataset location

FULL_TABLE_ID = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

# ─── BigQuery Schema ──────────────────────────────────────────────────────────

BQ_SCHEMA = [
    # --- traffic cols ---
    {"name": "date",                           "type": "DATE"},
    {"name": "area",                           "type": "STRING"},
    {"name": "route",                          "type": "STRING"},
    {"name": "traffic_volume",                 "type": "INTEGER"},
    {"name": "average_speed",                  "type": "FLOAT"},
    {"name": "travel_time_index",              "type": "FLOAT"},
    {"name": "congestion_level",               "type": "FLOAT"},
    {"name": "road_capacity_utilization",      "type": "FLOAT"},
    {"name": "incident_reports",               "type": "INTEGER"},
    {"name": "environmental_impact",           "type": "FLOAT"},
    {"name": "public_transport_usage",         "type": "FLOAT"},
    {"name": "traffic_signal_compliance",      "type": "FLOAT"},
    {"name": "parking_usage",                  "type": "FLOAT"},
    {"name": "pedestrian_and_cyclist_count",   "type": "INTEGER"},
    {"name": "weather_conditions",             "type": "STRING"},
    {"name": "roadwork_and_construction_activity", "type": "STRING"},
    # --- weather cols ---
    {"name": "date_w",                         "type": "STRING"},
    {"name": "avg_temp_c",                     "type": "FLOAT"},
    {"name": "total_precip_mm",                "type": "FLOAT"},
    {"name": "total_rain_mm",                  "type": "FLOAT"},
    {"name": "max_weathercode",                "type": "INTEGER"},
    {"name": "min_visibility_m",               "type": "FLOAT"},
    {"name": "avg_windspeed_kmh",              "type": "FLOAT"},
    {"name": "weather_severity",               "type": "INTEGER"},
    # --- derived cols ---
    {"name": "time_slot",                      "type": "STRING"},
    {"name": "hour",                           "type": "INTEGER"},
    {"name": "risk_score",                     "type": "FLOAT"},
    {"name": "risk_tier",                      "type": "STRING"},
]

# ─── Time-slot helpers ────────────────────────────────────────────────────────

# Maps the 4 time-slot labels (assigned in clean_data.py) to a representative hour
SLOT_TO_HOUR = {
    "Morning Peak (07-09)": 8,
    "Midday (11-13)":       12,
    "Evening Peak (17-19)": 18,
    "Night (21-23)":        22,
}

SLOT_PATTERN = r"\((\d{2})"   # extract the first hour from "(HH-HH)"


def _enrich_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer `hour` column derived from `time_slot`."""
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = df["time_slot"].map(SLOT_TO_HOUR)
        # fallback: parse first HH from pattern "(07-09)"
        mask = df["hour"].isna()
        if mask.any():
            df.loc[mask, "hour"] = (
                df.loc[mask, "time_slot"]
                  .str.extract(SLOT_PATTERN)[0]
                  .astype(float)
                  .astype("Int64")
            )
    return df


# ─── 1. Load CSV ──────────────────────────────────────────────────────────────

def load_processed() -> pd.DataFrame:
    log.info(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = _enrich_hour(df)
    log.info(f"  Loaded {len(df):,} rows")
    return df


# ─── 2. BigQuery helpers ──────────────────────────────────────────────────────

def _get_bq_client():
    """Return an authenticated BigQuery client."""
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        if GCP_KEY_PATH and Path(GCP_KEY_PATH).exists():
            log.info(f"  Authenticating via service-account key: {GCP_KEY_PATH}")
            creds = service_account.Credentials.from_service_account_file(
                GCP_KEY_PATH,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = bigquery.Client(project=GCP_PROJECT, credentials=creds)
        else:
            log.info("  Authenticating via Application Default Credentials (ADC)")
            client = bigquery.Client(project=GCP_PROJECT)

        # quick connection test
        _ = client.project
        log.info(f"  Connected to GCP project: {client.project}")
        return client

    except Exception as exc:
        log.warning(f"  BigQuery connection failed: {exc}")
        log.warning("  Falling back to offline (pandas) mode.")
        return None


def _ensure_dataset(client) -> None:
    from google.cloud import bigquery

    dataset_ref = f"{client.project}.{BQ_DATASET}"
    try:
        client.get_dataset(dataset_ref)
        log.info(f"  Dataset exists: {dataset_ref}")
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = LOCATION
        ds.description = "Bengaluru Commute Decision Tool – processed traffic + weather data"
        client.create_dataset(ds, exists_ok=True)
        log.info(f"  Created dataset: {dataset_ref}")


def _table_exists_and_populated(client) -> bool:
    """Return True only if the table has at least one row."""
    try:
        q = f"SELECT COUNT(*) AS n FROM `{FULL_TABLE_ID}` LIMIT 1"
        row = next(iter(client.query(q).result()))
        populated = row["n"] > 0
        log.info(f"  Table {FULL_TABLE_ID} exists with {row['n']:,} rows")
        return populated
    except Exception:
        return False


def _upload_to_bq(client, df: pd.DataFrame) -> None:
    from google.cloud import bigquery

    log.info(f"  Uploading {len(df):,} rows to {FULL_TABLE_ID} ...")

    # Build schema objects
    field_types = {f["name"]: f["type"] for f in BQ_SCHEMA}
    type_map = {
        "DATE":    bigquery.enums.SqlTypeNames.DATE,
        "STRING":  bigquery.enums.SqlTypeNames.STRING,
        "INTEGER": bigquery.enums.SqlTypeNames.INTEGER,
        "FLOAT":   bigquery.enums.SqlTypeNames.FLOAT,
    }
    schema = [
        bigquery.SchemaField(
            name=col,
            field_type=type_map.get(field_types.get(col, "STRING"), "STRING"),
        )
        for col in df.columns
        if col in field_types
    ]

    # Coerce types
    upload_df = df[[f["name"] for f in BQ_SCHEMA if f["name"] in df.columns]].copy()
    upload_df["date"] = pd.to_datetime(upload_df["date"]).dt.date

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.PARQUET,
    )

    try:
        job = client.load_table_from_dataframe(upload_df, FULL_TABLE_ID, job_config=job_config)
        job.result()                      # wait for completion
        log.info(f"  Upload complete. Rows loaded: {job.output_rows:,}")
    except Exception as exc:
        log.error(f"  Upload failed: {exc}")
        raise


# ─── 3. Dashboard Queries ─────────────────────────────────────────────────────

# Q1 – Primary: avg risk score + travel time, grouped by hour x route
Q1_BQ = f"""
SELECT
    hour,
    route,
    area,
    COUNT(*)                              AS record_count,
    ROUND(AVG(risk_score),       2)       AS avg_risk_score,
    ROUND(STDDEV(risk_score),    2)       AS stddev_risk_score,
    ROUND(AVG(travel_time_index), 4)      AS avg_travel_time_index,
    ROUND(STDDEV(travel_time_index), 4)   AS stddev_travel_time_index,
    ROUND(AVG(congestion_level), 2)       AS avg_congestion_level,
    ROUND(AVG(average_speed),    2)       AS avg_speed_kmh,
    ROUND(AVG(total_rain_mm),    3)       AS avg_rain_mm,
    ROUND(AVG(min_visibility_m), 0)       AS avg_visibility_m,
    ROUND(AVG(incident_reports), 3)       AS avg_incidents,
    ANY_VALUE(risk_tier)                  AS dominant_risk_tier
FROM `{FULL_TABLE_ID}`
GROUP BY hour, route, area
ORDER BY hour, avg_risk_score DESC
"""

# Q2 – Hourly trend across all routes (for a line chart)
Q2_BQ = f"""
SELECT
    hour,
    time_slot,
    COUNT(DISTINCT route)                 AS num_routes,
    ROUND(AVG(risk_score),       2)       AS avg_risk_score,
    ROUND(MAX(risk_score),       2)       AS max_risk_score,
    ROUND(MIN(risk_score),       2)       AS min_risk_score,
    ROUND(AVG(travel_time_index), 4)      AS avg_travel_time_index,
    ROUND(AVG(congestion_level), 2)       AS avg_congestion,
    ROUND(AVG(total_rain_mm),    3)       AS avg_rain_mm,
    ROUND(AVG(average_speed),    2)       AS avg_speed_kmh
FROM `{FULL_TABLE_ID}`
GROUP BY hour, time_slot
ORDER BY hour
"""

# Q3 – Route heatmap: percentage of records in each risk tier per route
Q3_BQ = f"""
SELECT
    route,
    area,
    risk_tier,
    COUNT(*)                              AS record_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY route),
        2
    )                                     AS pct_in_tier,
    ROUND(AVG(risk_score),       2)       AS avg_risk_score,
    ROUND(AVG(travel_time_index), 4)      AS avg_travel_time_index,
    ROUND(AVG(total_rain_mm),    3)       AS avg_rain_mm
FROM `{FULL_TABLE_ID}`
GROUP BY route, area, risk_tier
ORDER BY route, risk_tier
"""

# ─── Equivalent pandas queries (offline mode) ─────────────────────────────────

def _q1_pandas(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["hour", "route", "area"])
    result = grp.agg(
        record_count           =("risk_score",        "count"),
        avg_risk_score         =("risk_score",        "mean"),
        stddev_risk_score      =("risk_score",        "std"),
        avg_travel_time_index  =("travel_time_index", "mean"),
        stddev_travel_time_index=("travel_time_index","std"),
        avg_congestion_level   =("congestion_level",  "mean"),
        avg_speed_kmh          =("average_speed",     "mean"),
        avg_rain_mm            =("total_rain_mm",     "mean"),
        avg_visibility_m       =("min_visibility_m",  "mean"),
        avg_incidents          =("incident_reports",  "mean"),
    ).reset_index()
    # dominant risk tier = mode per group
    tier_mode = (
        df.groupby(["hour", "route", "area"])["risk_tier"]
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
          .reset_index()
          .rename(columns={"risk_tier": "dominant_risk_tier"})
    )
    result = result.merge(tier_mode, on=["hour", "route", "area"], how="left")
    result = result.round({"avg_risk_score": 2, "stddev_risk_score": 2,
                           "avg_travel_time_index": 4, "stddev_travel_time_index": 4,
                           "avg_congestion_level": 2, "avg_speed_kmh": 2,
                           "avg_rain_mm": 3, "avg_visibility_m": 0, "avg_incidents": 3})
    return result.sort_values(["hour", "avg_risk_score"], ascending=[True, False])


def _q2_pandas(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["hour", "time_slot"])
    result = grp.agg(
        num_routes             =("route",             "nunique"),
        avg_risk_score         =("risk_score",        "mean"),
        max_risk_score         =("risk_score",        "max"),
        min_risk_score         =("risk_score",        "min"),
        avg_travel_time_index  =("travel_time_index", "mean"),
        avg_congestion         =("congestion_level",  "mean"),
        avg_rain_mm            =("total_rain_mm",     "mean"),
        avg_speed_kmh          =("average_speed",     "mean"),
    ).reset_index()
    return result.round({"avg_risk_score": 2, "max_risk_score": 2, "min_risk_score": 2,
                         "avg_travel_time_index": 4, "avg_congestion": 2,
                         "avg_rain_mm": 3, "avg_speed_kmh": 2}).sort_values("hour")


def _q3_pandas(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["route", "area", "risk_tier"])
    counts = grp.agg(
        record_count           =("risk_score",        "count"),
        avg_risk_score         =("risk_score",        "mean"),
        avg_travel_time_index  =("travel_time_index", "mean"),
        avg_rain_mm            =("total_rain_mm",     "mean"),
    ).reset_index()
    route_totals = counts.groupby(["route"])["record_count"].transform("sum")
    counts["pct_in_tier"] = (100.0 * counts["record_count"] / route_totals).round(2)
    return counts.round({"avg_risk_score": 2, "avg_travel_time_index": 4,
                         "avg_rain_mm": 3}).sort_values(["route", "risk_tier"])


# ─── 4. Run Query (BQ or pandas) ──────────────────────────────────────────────

def run_query(label: str, bq_sql: str, pandas_fn, client, df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    if client is not None:
        log.info(f"  Running BQ query: {label}")
        try:
            result = client.query(bq_sql).to_dataframe()
            elapsed = time.perf_counter() - t0
            log.info(f"  [{label}] {len(result):,} rows  ({elapsed:.2f}s)  [BigQuery]")
            return result
        except Exception as exc:
            log.warning(f"  BQ query failed ({exc}), falling back to pandas")
    # Offline / fallback
    result = pandas_fn(df)
    elapsed = time.perf_counter() - t0
    log.info(f"  [{label}] {len(result):,} rows  ({elapsed:.2f}s)  [pandas]")
    return result


# ─── 5. Print & Save ──────────────────────────────────────────────────────────

def save(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    df.to_csv(path, index=False)
    log.info(f"  Saved: {path}  ({len(df):,} rows)")
    return path


def print_banner(title: str) -> None:
    line = "=" * 62
    log.info(line)
    log.info(f"  {title}")
    log.info(line)


def print_table(df: pd.DataFrame, n: int = 12, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    print(df.head(n).to_string(index=False))
    if len(df) > n:
        print(f"  ... ({len(df) - n} more rows — see CSV)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    print_banner("Bengaluru Commute Dashboard - BigQuery Analytics")

    # ── Load local processed data
    df = load_processed()

    # ── Connect to BigQuery (or go offline)
    client = None
    if not OFFLINE_MODE:
        log.info("Connecting to BigQuery ...")
        client = _get_bq_client()

    if client is not None:
        log.info("Preparing BigQuery dataset + table ...")
        try:
            _ensure_dataset(client)
            if not _table_exists_and_populated(client):
                _upload_to_bq(client, df)
            else:
                log.info("  Table already populated, skipping upload.")
        except Exception as exc:
            log.warning(f"  BQ setup error: {exc}. Switching to offline mode.")
            client = None
    else:
        log.info("Running in OFFLINE mode (pandas only).")

    # ── Q1: avg risk + travel time by hour x route
    log.info("Running Q1: risk & travel-time by hour x route ...")
    q1 = run_query("Q1_hour_route", Q1_BQ, _q1_pandas, client, df)
    save(q1, "q1_hour_x_route.csv")

    # ── Q2: hourly trend across all routes
    log.info("Running Q2: hourly system-wide trend ...")
    q2 = run_query("Q2_hourly_trend", Q2_BQ, _q2_pandas, client, df)
    save(q2, "q2_hourly_trend.csv")

    # ── Q3: route x risk-tier heatmap
    log.info("Running Q3: route x risk-tier heatmap ...")
    q3 = run_query("Q3_route_heatmap", Q3_BQ, _q3_pandas, client, df)
    save(q3, "q3_route_heatmap.csv")

    # ── Print summaries
    print()
    print_banner("Q1 - Avg Risk Score & Travel Time  (hour x route)")
    print_table(
        q1[["hour", "route", "avg_risk_score", "avg_travel_time_index",
            "avg_speed_kmh", "dominant_risk_tier"]],
        title="Top rows by risk score per hour:"
    )

    print()
    print_banner("Q2 - System-wide Hourly Trend")
    print_table(
        q2[["hour", "time_slot", "avg_risk_score", "avg_travel_time_index",
            "avg_congestion", "avg_speed_kmh", "avg_rain_mm"]],
        n=8,
        title="All hours:"
    )

    print()
    print_banner("Q3 - Route x Risk-Tier Heatmap  (top risky routes)")
    critical_routes = (
        q3[q3["risk_tier"].isin(["High", "Critical"])]
          .sort_values("avg_risk_score", ascending=False)
          .head(15)
    )
    print_table(
        critical_routes[["route", "area", "risk_tier", "pct_in_tier",
                          "avg_risk_score", "avg_travel_time_index"]],
        n=15
    )

    # ── Final summary
    elapsed = time.perf_counter() - t_start
    print()
    print_banner("Pipeline Complete")
    log.info(f"  Mode     : {'BigQuery' if client else 'Offline (pandas)'}")
    log.info(f"  Q1 rows  : {len(q1):,}")
    log.info(f"  Q2 rows  : {len(q2):,}")
    log.info(f"  Q3 rows  : {len(q3):,}")
    log.info(f"  Outputs  : {OUT_DIR}/")
    log.info(f"  Elapsed  : {elapsed:.1f}s")
    print_banner("Done")


if __name__ == "__main__":
    main()
