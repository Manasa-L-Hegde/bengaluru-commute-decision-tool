"""
weather_api.py - Unified weather helper for the commute app.
  get_weather(target_date, time_slot, df_hist) ->
      {"rain_mm": float, "source": "historical"|"forecast"}
"""
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

SLOT_HOURS = {
    "Morning Peak (07-09)": (7, 9),
    "Midday (11-13)":       (11, 13),
    "Evening Peak (17-19)": (17, 19),
    "Night (21-23)":        (21, 23),
}

HIST_DATE_MIN = date(2022, 1, 1)
HIST_DATE_MAX = date(2024, 8, 9)


def get_weather(target_date, time_slot, df_hist=None):
    """
    Returns dict with rainfall and source label.
    - historical range: look up from df_hist
    - today / future: call Open-Meteo forecast API
    """
    td = target_date if isinstance(target_date, date) else target_date.date()

    if HIST_DATE_MIN <= td <= HIST_DATE_MAX and df_hist is not None:
        return _historical_lookup(td, time_slot, df_hist)
    else:
        return _forecast_lookup(td, time_slot)


def _historical_lookup(td, time_slot, df_hist):
    mask = (pd.to_datetime(df_hist["date"]).dt.date == td)
    rows = df_hist[mask]
    if rows.empty:
        return {"rain_mm": 0.0, "source": "historical", "note": "no data for date"}
    rain = rows["total_rain_mm"].mean()
    return {"rain_mm": float(rain), "source": "historical"}


def _forecast_lookup(td, time_slot):
    """Call Open-Meteo forecast endpoint (free, no key)."""
    today = date.today()
    delta = (td - today).days

    # Open-Meteo forecast covers up to 16 days ahead
    if delta < 0:
        # Past date outside historical range - use archive
        return _archive_lookup(td, time_slot)
    if delta > 16:
        return {"rain_mm": 0.0, "source": "forecast",
                "note": "beyond 16-day forecast; using 0mm default"}

    h_start, h_end = SLOT_HOURS.get(time_slot, (7, 9))
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  BENGALURU_LAT,
            "longitude": BENGALURU_LON,
            "hourly":    "precipitation,rain",
            "timezone":  "Asia/Kolkata",
            "start_date": td.isoformat(),
            "end_date":   td.isoformat(),
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        times = data.get("time", [])
        rain  = data.get("rain", data.get("precipitation", []))

        # Sum rain for time-slot hours
        total = 0.0
        for t, r in zip(times, rain):
            h = int(t.split("T")[1].split(":")[0])
            if h_start <= h <= h_end:
                total += r if r else 0.0

        return {"rain_mm": round(total, 3), "source": "forecast"}

    except Exception as e:
        return {"rain_mm": 0.0, "source": "forecast",
                "note": f"API error: {e}"}


def _archive_lookup(td, time_slot):
    """Fallback: Open-Meteo archive for dates before historical CSV."""
    h_start, h_end = SLOT_HOURS.get(time_slot, (7, 9))
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":  BENGALURU_LAT,
            "longitude": BENGALURU_LON,
            "hourly":    "precipitation,rain",
            "timezone":  "Asia/Kolkata",
            "start_date": td.isoformat(),
            "end_date":   td.isoformat(),
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("hourly", {})
        times = data.get("time", [])
        rain  = data.get("rain", data.get("precipitation", []))

        total = 0.0
        for t, r in zip(times, rain):
            h = int(t.split("T")[1].split(":")[0])
            if h_start <= h <= h_end:
                total += r if r else 0.0

        return {"rain_mm": round(total, 3), "source": "archive"}
    except Exception:
        return {"rain_mm": 0.0, "source": "archive", "note": "API fallback failed"}
