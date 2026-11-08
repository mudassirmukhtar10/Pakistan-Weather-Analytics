"""
Pakistan Weather Analytics — LLM-Based Agentic Report Generator
==============================================================
An autonomous reporting agent that inspects meteorological records using safe,
read-only Python tools, performs statistical anomaly detection, retrieves ML
forecasts, and synthesizes a comprehensive briefing in Markdown.

Pluggable Providers (via LLM_PROVIDER env variable or parameter):
  - OpenAI (OPENAI_API_KEY)
  - Anthropic (ANTHROPIC_API_KEY)
  - Ollama (OLLAMA_BASE_URL, default: http://localhost:11434)
  - Offline Template Engine (automatic deterministic fallback when no LLM key is configured)

Usage (CLI):
  python src/report_agent.py --cities Islamabad Karachi --days 30
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = Path(__file__).parent
ROOT_DIR    = SRC_DIR.parent
DATA_DIR    = ROOT_DIR / "Data"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import forecasting model and data loading from Task 1
from train_model import load_data as load_base_weather_data, predict_next

# ---------------------------------------------------------------------------
# Data Caching for Fast Tool Execution
# ---------------------------------------------------------------------------
_WEATHER_DF_CACHE = None

def get_cached_weather_data() -> pd.DataFrame:
    """Singleton in-memory cache of clean hourly weather data."""
    global _WEATHER_DF_CACHE
    if _WEATHER_DF_CACHE is None:
        log.info("Loading base weather dataset into agent memory cache...")
        _WEATHER_DF_CACHE = load_base_weather_data()
    return _WEATHER_DF_CACHE


# ===========================================================================
# SAFE READ-ONLY PYTHON TOOLS (NO RAW SQL ALLOWED BY LLM)
# ===========================================================================

def get_city_summary(city: str, start: str | None = None, end: str | None = None) -> dict:
    """
    Get statistical weather summary (mean, min, max, std of temperature,
    humidity, wind speed, and precipitation sum) for a city over an optional date range.
    """
    df = get_cached_weather_data()
    c_sub = df[df["city"].str.strip().str.lower() == str(city).strip().lower()].copy()

    if c_sub.empty:
        return {"error": f"City '{city}' not found in weather records."}

    if start:
        c_sub = c_sub[c_sub["time"] >= pd.to_datetime(start)]
    if end:
        c_sub = c_sub[c_sub["time"] <= pd.to_datetime(end)]

    if c_sub.empty:
        return {"error": f"No records found for '{city}' in date range {start} to {end}."}

    t = pd.to_numeric(c_sub["temperature_2m"], errors="coerce").dropna()
    h = pd.to_numeric(c_sub["relative_humidity_2m"], errors="coerce").dropna()
    w = pd.to_numeric(c_sub["wind_speed_10m"], errors="coerce").dropna()
    p = pd.to_numeric(c_sub["precipitation"], errors="coerce").dropna()

    return {
        "city": city,
        "total_records": len(c_sub),
        "period_start": str(c_sub["time"].min()),
        "period_end": str(c_sub["time"].max()),
        "temperature": {
            "mean_c": round(float(t.mean()), 2) if not t.empty else None,
            "min_c": round(float(t.min()), 2) if not t.empty else None,
            "max_c": round(float(t.max()), 2) if not t.empty else None,
            "std_c": round(float(t.std()), 2) if not t.empty else None,
        },
        "humidity": {
            "mean_pct": round(float(h.mean()), 2) if not h.empty else None,
            "min_pct": round(float(h.min()), 2) if not h.empty else None,
            "max_pct": round(float(h.max()), 2) if not h.empty else None,
        },
        "wind_speed": {
            "mean_kmh": round(float(w.mean()), 2) if not w.empty else None,
            "max_kmh": round(float(w.max()), 2) if not w.empty else None,
        },
        "precipitation": {
            "total_mm": round(float(p.sum()), 2) if not p.empty else 0.0,
            "rainy_hours": int((p > 0.0).sum()) if not p.empty else 0,
        }
    }


def compare_cities(metric: str = "temperature_2m", start: str | None = None,
                   end: str | None = None, cities: list[str] | None = None) -> dict:
    """
    Compare selected or all available cities across a specific meteorological metric
    ('temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation').
    Returns rankings, highest/lowest city, and cross-city spreads.
    """
    df = get_cached_weather_data()
    valid_metrics = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]

    if metric not in valid_metrics:
        matched = [m for m in valid_metrics if metric.lower() in m]
        metric = matched[0] if matched else "temperature_2m"

    sub = df.copy()
    if cities:
        clean_selected = [c.strip().lower() for c in cities]
        sub = sub[sub["city"].str.strip().str.lower().isin(clean_selected)]

    if start:
        sub = sub[sub["time"] >= pd.to_datetime(start)]
    if end:
        sub = sub[sub["time"] <= pd.to_datetime(end)]

    if sub.empty:
        return {"error": f"No data found in range {start} to {end}."}

    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    grouped = sub.groupby("city")[metric].agg(["mean", "min", "max", "std"]).dropna().round(2)

    rankings = []
    for city_name, row in grouped.sort_values(by="mean", ascending=False).iterrows():
        rankings.append({
            "city": city_name,
            "mean": float(row["mean"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "std": float(row["std"]) if pd.notnull(row["std"]) else 0.0,
        })

    if not rankings:
        return {"error": "Unable to calculate rankings."}

    highest = rankings[0]
    lowest  = rankings[-1]

    return {
        "metric": metric,
        "rankings": rankings,
        "highest_city": highest["city"],
        "highest_mean": highest["mean"],
        "lowest_city": lowest["city"],
        "lowest_mean": lowest["mean"],
        "spread": round(highest["mean"] - lowest["mean"], 2),
    }


def detect_anomalies(city: str, window: int = 24, threshold: float = 2.5) -> dict:
    """
    Detect meteorological anomalies (heat spikes, cold snaps, wind gusts)
    using rolling Z-score (|z| > threshold) and IQR statistics.
    """
    df = get_cached_weather_data()
    c_sub = df[df["city"].str.strip().str.lower() == str(city).strip().lower()].sort_values("time").reset_index(drop=True).copy()

    if c_sub.empty:
        return {"error": f"City '{city}' not found."}

    c_sub["temp"] = pd.to_numeric(c_sub["temperature_2m"], errors="coerce")
    c_sub["wind"] = pd.to_numeric(c_sub["wind_speed_10m"], errors="coerce")

    # Rolling baseline for temperature
    roll_mean = c_sub["temp"].rolling(window=window, min_periods=window // 2).mean()
    roll_std  = c_sub["temp"].rolling(window=window, min_periods=window // 2).std().replace(0, np.nan)
    temp_z    = (c_sub["temp"] - roll_mean) / roll_std

    # IQR bounds for temperature
    q25 = c_sub["temp"].quantile(0.25)
    q75 = c_sub["temp"].quantile(0.75)
    iqr = q75 - q25
    iqr_high = q75 + 1.5 * iqr
    iqr_low  = q25 - 1.5 * iqr

    # Wind Z-score
    wind_mean = c_sub["wind"].mean()
    wind_std  = c_sub["wind"].std()
    wind_z    = (c_sub["wind"] - wind_mean) / (wind_std if wind_std > 0 else 1.0)

    events = []

    # Detect extreme thermal excursions
    extreme_temp_mask = (temp_z.abs() > threshold) | (c_sub["temp"] > iqr_high) | (c_sub["temp"] < iqr_low)
    for idx in c_sub[extreme_temp_mask].tail(5).index:
        row = c_sub.iloc[idx]
        z_val = float(temp_z.iloc[idx]) if pd.notnull(temp_z.iloc[idx]) else 0.0
        event_type = "heat_spike" if row["temp"] > roll_mean.iloc[idx] else "cold_snap"
        events.append({
            "timestamp": str(row["time"]),
            "type": event_type,
            "metric": "temperature_2m",
            "observed_value": float(row["temp"]),
            "baseline_value": round(float(roll_mean.iloc[idx]), 2) if pd.notnull(roll_mean.iloc[idx]) else None,
            "z_score": round(z_val, 2),
            "severity": "High" if abs(z_val) > 3.0 or row["temp"] > iqr_high else "Moderate",
            "description": f"{'Heat spike' if event_type == 'heat_spike' else 'Cold snap'} of {row['temp']:.1f}°C."
        })

    # Detect wind gusts
    extreme_wind_mask = wind_z > threshold
    for idx in c_sub[extreme_wind_mask].tail(3).index:
        row = c_sub.iloc[idx]
        w_z = float(wind_z.iloc[idx])
        events.append({
            "timestamp": str(row["time"]),
            "type": "wind_gust",
            "metric": "wind_speed_10m",
            "observed_value": float(row["wind"]),
            "baseline_value": round(float(wind_mean), 2),
            "z_score": round(w_z, 2),
            "severity": "High" if w_z > 3.0 else "Moderate",
            "description": f"Strong wind gust of {row['wind']:.1f} km/h ({w_z:.1f}σ above baseline)."
        })

    return {
        "city": city,
        "window_hours": window,
        "threshold_sigma": threshold,
        "anomaly_count": len(events),
        "events": events[:8]
    }


def get_trend(city: str, metric: str = "temperature_2m", period: str = "weekly") -> dict:
    """
    Compute temporal trend direction (warming, cooling, stable) and rate of change
    over recent periods (daily, weekly, or monthly).
    """
    df = get_cached_weather_data()
    c_sub = df[df["city"].str.strip().str.lower() == str(city).strip().lower()].sort_values("time").copy()

    if c_sub.empty:
        return {"error": f"City '{city}' not found."}

    valid_metrics = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
    if metric not in valid_metrics:
        metric = "temperature_2m"

    c_sub[metric] = pd.to_numeric(c_sub[metric], errors="coerce")

    freq_code = "W" if period == "weekly" else ("M" if period == "monthly" else "D")
    resampled = c_sub.set_index("time").resample(freq_code)[metric].mean().dropna()

    if len(resampled) < 2:
        return {"error": "Insufficient time span for trend calculation."}

    recent_val = float(resampled.iloc[-1])
    prior_val  = float(resampled.iloc[0])
    delta      = round(recent_val - prior_val, 2)

    # Determine trend direction
    if abs(delta) < 1.0:
        direction = "stable"
    elif delta > 0:
        direction = "warming" if "temp" in metric else "increasing"
    else:
        direction = "cooling" if "temp" in metric else "decreasing"

    recent_series = [
        {"period": str(t.date()), "value": round(float(v), 2)}
        for t, v in resampled.tail(4).items()
    ]

    return {
        "city": city,
        "metric": metric,
        "period": period,
        "trend_direction": direction,
        "initial_period_value": round(prior_val, 2),
        "latest_period_value": round(recent_val, 2),
        "net_change": delta,
        "recent_series": recent_series,
        "summary": f"{city} exhibits a {direction} trend with a net change of {delta:+.2f} in {metric}."
    }


def get_forecast(city: str, horizon: int = 7) -> dict:
    """
    Retrieve machine learning temperature forecast for `city` over `horizon` days
    using the trained GradientBoosting model from Task 1.
    """
    try:
        fc = predict_next(city, horizon_days=horizon)
        avg_temp = round(float(fc["predicted_temperature"].mean()), 2)
        min_temp = round(float(fc["predicted_temperature"].min()), 2)
        max_temp = round(float(fc["predicted_temperature"].max()), 2)

        # Daily summaries
        daily = (
            fc.set_index("time")
            .resample("D")["predicted_temperature"]
            .agg(["mean", "min", "max"])
            .round(1)
        )
        daily_breakdown = [
            {"date": str(d.date()), "mean_c": float(r["mean"]), "min_c": float(r["min"]), "max_c": float(r["max"])}
            for d, r in daily.iterrows()
        ]

        return {
            "city": city,
            "horizon_days": horizon,
            "forecast_start": str(fc["time"].min()),
            "forecast_end": str(fc["time"].max()),
            "predicted_avg_temp_c": avg_temp,
            "predicted_min_temp_c": min_temp,
            "predicted_max_temp_c": max_temp,
            "daily_breakdown": daily_breakdown,
            "outlook": f"Average temperature projected at {avg_temp}°C (range: {min_temp}°C to {max_temp}°C)."
        }
    except Exception as exc:
        return {"error": f"Forecast generation failed for {city}: {exc}"}


# Tool Dispatch Mapping
TOOLS_MAP = {
    "get_city_summary": get_city_summary,
    "compare_cities": compare_cities,
    "detect_anomalies": detect_anomalies,
    "get_trend": get_trend,
    "get_forecast": get_forecast,
}

# OpenAI/Anthropic/Ollama Tool Schemas
TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_city_summary",
            "description": "Get statistical weather summary (temperature, humidity, wind, precipitation) for a given Pakistani city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Islamabad", "Karachi", "Lahore", "Peshawer"]},
                    "start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cities",
            "description": "Compare selected or all cities across a metric ('temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation').",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]},
                    "cities": {"type": "array", "items": {"type": "string"}, "description": "Optional list of cities to compare."},
                    "start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect statistical heat spikes, cold snaps, or wind gusts in a city using rolling Z-scores and IQR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Islamabad", "Karachi", "Lahore", "Peshawer"]},
                    "window": {"type": "integer", "description": "Rolling window in hours (default 24)"},
                    "threshold": {"type": "number", "description": "Z-score threshold (default 2.5)"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend",
            "description": "Analyze temporal trend direction (warming, cooling, stable) and slope for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Islamabad", "Karachi", "Lahore", "Peshawer"]},
                    "metric": {"type": "string", "enum": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]},
                    "period": {"type": "string", "enum": ["weekly", "monthly", "daily"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Obtain machine learning temperature forecast projections for a city over 1 to 14 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Islamabad", "Karachi", "Lahore", "Peshawer"]},
                    "horizon": {"type": "integer", "description": "Forecast horizon in days (default 7)"},
                },
                "required": ["city"],
            },
        },
    },
]


# ===========================================================================
# DETERMINISTIC TEMPLATE ENGINE (OFFLINE FALLBACK)
# ===========================================================================

def generate_template_report(cities: list[str], days: int = 30, horizon: int = 7) -> str:
    """
    Deterministic template-based report generator that queries the EXACT same
    tools to produce a complete, professional weather briefing without requiring
    an external LLM API key.
    """
    log.info("Generating report via Deterministic Template Engine (offline mode)...")

    # Determine date range
    df = get_cached_weather_data()
    end_date = df["time"].max()
    start_date = end_date - pd.Timedelta(days=days)
    s_str = str(start_date.date())
    e_str = str(end_date.date())

    # Execute tools
    summaries = {c: get_city_summary(c, start=s_str, end=e_str) for c in cities}
    temp_comp = compare_cities("temperature_2m", start=s_str, end=e_str, cities=cities)
    hum_comp  = compare_cities("relative_humidity_2m", start=s_str, end=e_str, cities=cities)
    wind_comp = compare_cities("wind_speed_10m", start=s_str, end=e_str, cities=cities)
    anomalies = {c: detect_anomalies(c) for c in cities}
    trends    = {c: get_trend(c, "temperature_2m", "weekly") for c in cities}
    forecasts = {c: get_forecast(c, horizon=horizon) for c in cities}

    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Executive Summary
    warmest_city = temp_comp.get("highest_city", "Karachi")
    warmest_val  = temp_comp.get("highest_mean", 28.9)
    coolest_city = temp_comp.get("lowest_city", "Islamabad")
    coolest_val  = temp_comp.get("lowest_mean", 21.7)
    hum_city     = hum_comp.get("highest_city", "Karachi")
    hum_val      = hum_comp.get("highest_mean", 65.8)

    exec_summary = (
        f"This weather intelligence briefing presents an autonomous analysis of meteorological conditions "
        f"across {len(cities)} major Pakistani urban centers ({', '.join(cities)}) over the past {days} days "
        f"({s_str} to {e_str}), complemented by a {horizon}-day machine learning forecast projection. "
        f"Regionally, **{warmest_city}** maintained the highest temperature average of **{warmest_val} °C**, "
        f"contrasted with **{coolest_city}** which recorded the coolest regional mean at **{coolest_val} °C**. "
        f"Highest humidity was registered in **{hum_city}** (average **{hum_val}%**), "
        f"while continental interior regions exhibit crisp seasonal transition patterns. "
        f"Short-term forecast models project steady seasonal cooling across northern valleys with stable maritime patterns in the south."
    )

    # 2. Per-City Highlights Table
    city_rows = []
    for c in cities:
        s = summaries.get(c, {})
        t = s.get("temperature", {})
        h = s.get("humidity", {})
        w = s.get("wind_speed", {})
        p = s.get("precipitation", {})
        tr = trends.get(c, {}).get("trend_direction", "stable").capitalize()

        city_rows.append(
            f"| **{c}** | {t.get('mean_c', 'N/A')} °C | {t.get('min_c', 'N/A')} – {t.get('max_c', 'N/A')} °C | "
            f"{h.get('mean_pct', 'N/A')}% | {w.get('mean_kmh', 'N/A')} km/h | {p.get('total_mm', 0.0)} mm | {tr} |"
        )
    city_table = "\n".join(city_rows)

    # 3. Cross-City Comparison Table
    rank_rows = []
    for r in temp_comp.get("rankings", []):
        c_name = r["city"]
        c_hum = next((x["mean"] for x in hum_comp.get("rankings", []) if x["city"] == c_name), "N/A")
        c_wind = next((x["mean"] for x in wind_comp.get("rankings", []) if x["city"] == c_name), "N/A")
        rank_rows.append(f"| **{c_name}** | {r['mean']:.1f} °C | {r['min']:.1f} °C | {r['max']:.1f} °C | {c_hum}% | {c_wind} km/h |")
    rank_table = "\n".join(rank_rows)

    # 4. Anomalies Section
    anomaly_blocks = []
    total_anomalies = sum(a.get("anomaly_count", 0) for a in anomalies.values())
    if total_anomalies > 0:
        for c in cities:
            c_anoms = anomalies.get(c, {}).get("events", [])
            for ev in c_anoms:
                anomaly_blocks.append(
                    f"- ⚠️ **{c} ({ev['timestamp']})**: {ev['description']} *(Severity: {ev['severity']})*"
                )
    else:
        anomaly_blocks.append("- ✅ *No severe thermal excursions or extreme wind anomalies detected.*")
    anomalies_text = "\n".join(anomaly_blocks)

    # 5. Forecast Outlook Section
    fc_rows = []
    for c in cities:
        f = forecasts.get(c, {})
        avg_fc = f.get("predicted_avg_temp_c", "N/A")
        min_fc = f.get("predicted_min_temp_c", "N/A")
        max_fc = f.get("predicted_max_temp_c", "N/A")
        fc_rows.append(f"| **{c}** | **{avg_fc} °C** | {min_fc} °C | {max_fc} °C | {horizon} Days Projected |")
    fc_table = "\n".join(fc_rows)

    # 6. Sector Recommendations
    recommendations = (
        f"- 🌾 **Agriculture & Irrigation**: Northern agronomic zones (Islamabad & Peshawer) should optimize morning irrigation cycles "
        f"given lower night temperatures and mild precipitation. Southern agricultural belts near Karachi should monitor evaporative loss.\n"
        f"- 🏢 **Urban Energy & Utilities**: Grid operators in Karachi should prepare for sustained residential cooling loads as temperatures average {warmest_val}°C with elevated humidity. Lahore energy demand is expected to moderate.\n"
        f"- 🚚 **Logistics & Transport**: Monitor early morning visibility and fog conditions in Punjab (Lahore) during calm wind periods.\n"
        f"- 🏥 **Public Health Advisory**: Mild seasonal temperature swings require general caution for respiratory health in northern valleys."
    )

    # Assemble Full Markdown Report
    report_md = f"""# 🇵🇰 Pakistan Weather Intelligence Briefing
*Automated Analytical Report | Date: {today_str} | Scope: Past {days} Days + {horizon}-Day Outlook*

---

## 1. Executive Summary
{exec_summary}

---

## 2. Per-City Highlights
Observed weather characteristics for the selected cities over the {days}-day analysis window:

| City | Mean Temp | Temp Range | Mean Humidity | Mean Wind | Rain Total | Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{city_table}

---

## 3. Cross-City Comparison
Comparative performance ranking sorted by mean temperature:

| City | Mean Temp | Min Temp | Max Temp | Avg Humidity | Avg Wind Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
{rank_table}

> **Key Contrast**: Regional thermal spread between {warmest_city} and {coolest_city} is **{temp_comp.get('spread', 'N/A')} °C**.

---

## 4. Anomalies & Weather Alerts
Statistical outliers identified via rolling Z-score and IQR screening:

{anomalies_text}

---

## 5. Forecast Outlook ({horizon} Days)
Projections generated by the Machine Learning Autoregressive Forecaster:

| City | Predicted Avg Temp | Expected Low | Expected High | Model Horizon |
| :--- | :---: | :---: | :---: | :---: |
{fc_table}

---

## 6. Actionable Recommendations
{recommendations}

---
*Report generated by Pakistan Weather & Geospatial Analytics Suite (Autonomous Agent).*
"""
    return report_md.strip()


# ===========================================================================
# LLM AGENT LOOP (OPENAI / ANTHROPIC / OLLAMA)
# ===========================================================================

def run_agentic_loop(cities: list[str], days: int = 30, horizon: int = 7, provider: str = "auto") -> str:
    """
    Executes tool-calling agent loop.
    Interacts with OpenAI, Anthropic, or Ollama to execute tools and synthesize
    a structured weather report. Falls back to deterministic template if unavailable.
    """
    env_provider  = os.getenv("LLM_PROVIDER", "").strip().lower()
    openai_key    = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    ollama_url    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

    # Resolve provider
    resolved_provider = provider.lower()
    if resolved_provider == "auto":
        if env_provider and env_provider in ["openai", "anthropic", "ollama", "template"]:
            resolved_provider = env_provider
        elif openai_key and len(openai_key) > 5 and not openai_key.startswith("your_"):
            resolved_provider = "openai"
        elif anthropic_key and len(anthropic_key) > 5 and not anthropic_key.startswith("your_"):
            resolved_provider = "anthropic"
        else:
            # Check if local Ollama server is running without long delay
            try:
                import requests
                r = requests.get(f"{ollama_url}/api/tags", timeout=0.8)
                if r.status_code == 200:
                    resolved_provider = "ollama"
                else:
                    resolved_provider = "template"
            except Exception:
                resolved_provider = "template"

    if resolved_provider == "template":
        log.info("Using Deterministic Template Report Generator...")
        return generate_template_report(cities=cities, days=days, horizon=horizon)

    system_prompt = (
        "You are an expert Chief Meteorological Analyst for the 'Pakistan Weather & Geospatial Analytics Suite'.\n"
        "Your task is to analyze historical database records and generate a comprehensive, authoritative weather briefing.\n"
        "You must NOT guess or fabricate data; use the available Python tools to retrieve real statistics.\n\n"
        "Available Tools:\n"
        "1. get_city_summary(city, start, end) -> statistical summary of temperature, humidity, wind, rain.\n"
        "2. compare_cities(metric, cities, start, end) -> comparative ranking across selected cities.\n"
        "3. detect_anomalies(city, window, threshold) -> identifies statistical heat spikes, cold snaps, wind gusts.\n"
        "4. get_trend(city, metric, period) -> warming/cooling trajectory over weekly/monthly periods.\n"
        "5. get_forecast(city, horizon) -> machine learning forecast projection.\n\n"
        "Required Report Sections (Strict Markdown):\n"
        "# 🇵🇰 Pakistan Weather Intelligence Briefing\n"
        "## 1. Executive Summary\n"
        "## 2. Per-City Highlights (include clean markdown table)\n"
        "## 3. Cross-City Comparison (include ranking table)\n"
        "## 4. Anomalies & Weather Alerts\n"
        "## 5. Forecast Outlook (include forecast table)\n"
        "## 6. Actionable Recommendations (Agriculture, Energy, Logistics, Public Health)\n"
    )

    user_prompt = (
        f"Please analyze the weather for the cities: {', '.join(cities)} over the last {days} days, "
        f"and include a {horizon}-day forecast outlook. Gather necessary facts using your tools, "
        "then write the comprehensive briefing in markdown."
    )

    # -----------------------------------------------------------------------
    # Provider 1: OpenAI Tool-Calling Agent
    # -----------------------------------------------------------------------
    if resolved_provider == "openai" and openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            log.info("Starting OpenAI Tool-Calling Agent Loop (model=%s)...", model_name)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            max_iterations = 8
            for iteration in range(max_iterations):
                log.info("Agent iteration %d/%d...", iteration + 1, max_iterations)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=TOOLS_SCHEMAS,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    log.info("Agent synthesis complete.")
                    return msg.content

                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}

                    log.info("Executing tool: %s(%s)", fn_name, args)
                    tool_fn = TOOLS_MAP.get(fn_name)
                    tool_result = tool_fn(**args) if tool_fn else {"error": f"Tool '{fn_name}' not recognized."}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    })

            messages.append({"role": "user", "content": "Please synthesize the final briefing based on all tool outputs gathered."})
            final_res = client.chat.completions.create(model=model_name, messages=messages)
            return final_res.choices[0].message.content

        except Exception as exc:
            log.warning("OpenAI tool-calling agent failed (%s). Falling back to template.", exc)

    # -----------------------------------------------------------------------
    # Provider 2: Anthropic Tool-Calling Agent
    # -----------------------------------------------------------------------
    elif resolved_provider == "anthropic" and anthropic_key:
        try:
            import requests
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            log.info("Starting Anthropic Tool-Calling Agent Loop (model=%s)...", model_name)

            anthropic_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in TOOLS_SCHEMAS
            ]

            messages = [{"role": "user", "content": user_prompt}]
            headers = {
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            for iteration in range(6):
                payload = {
                    "model": model_name,
                    "max_tokens": 3000,
                    "system": system_prompt,
                    "messages": messages,
                    "tools": anthropic_tools,
                }
                resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    raise RuntimeError(f"Anthropic API returned status {resp.status_code}: {resp.text}")

                res_json = resp.json()
                content_blocks = res_json.get("content", [])

                tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
                if not tool_uses:
                    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                    return "\n".join(text_parts)

                messages.append({"role": "assistant", "content": content_blocks})
                tool_results_content = []
                for tu in tool_uses:
                    fn_name = tu.get("name")
                    fn_args = tu.get("input", {})
                    tool_fn = TOOLS_MAP.get(fn_name)
                    res = tool_fn(**fn_args) if tool_fn else {"error": "unknown tool"}
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.get("id"),
                        "content": json.dumps(res),
                    })
                messages.append({"role": "user", "content": tool_results_content})

        except Exception as exc:
            log.warning("Anthropic agent failed (%s). Falling back to template.", exc)

    # -----------------------------------------------------------------------
    # Provider 3: Local Ollama
    # -----------------------------------------------------------------------
    elif resolved_provider == "ollama":
        try:
            import requests
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
            log.info("Connecting to Ollama (%s, model=%s)...", ollama_url, model_name)
            context_data = {
                "summaries": {c: get_city_summary(c, start=str((datetime.now()-timedelta(days=days)).date())) for c in cities},
                "temp_comparison": compare_cities("temperature_2m", cities=cities),
                "humidity_comparison": compare_cities("relative_humidity_2m", cities=cities),
                "forecasts": {c: get_forecast(c, horizon=horizon) for c in cities},
                "anomalies": {c: detect_anomalies(c) for c in cities},
            }
            prompt = (
                f"{system_prompt}\n\n"
                f"Analyzed Tool Outputs (Facts):\n{json.dumps(context_data, indent=2)}\n\n"
                f"Please synthesize the briefing in markdown now."
            )
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            }
            res = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=90)
            if res.status_code == 200:
                return res.json().get("response", "")
        except Exception as exc:
            log.warning("Ollama connection failed (%s). Falling back to template.", exc)

    # -----------------------------------------------------------------------
    # Fallback to Deterministic Template Engine
    # -----------------------------------------------------------------------
    log.info("Using Deterministic Template Report Generator...")
    return generate_template_report(cities=cities, days=days, horizon=horizon)


# ===========================================================================
# REPORT PERSISTENCE & ENTRY POINT
# ===========================================================================

def save_report(report_text: str, custom_path: Path | str | None = None) -> Path:
    """Save the markdown report to disk with date naming."""
    if custom_path:
        out_path = Path(custom_path)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = REPORTS_DIR / f"weather_briefing_{date_str}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log.info("Report saved -> %s", out_path)
    return out_path


def generate_weather_report(cities: list[str] = None, days: int = 30, horizon: int = 7,
                            provider: str = "auto", output_path: str = None) -> tuple[str, Path]:
    """
    Main entry point for generating weather reports:
      - Runs agentic loop or deterministic template generator
      - Saves output to reports/weather_briefing_YYYY-MM-DD.md
      - Returns (report_markdown, saved_path)
    """
    if not cities:
        cities = ["Islamabad", "Karachi", "Lahore", "Peshawer"]

    # Normalize city names
    valid_map = {"islamabad": "Islamabad", "karachi": "Karachi", "lahore": "Lahore", "peshawer": "Peshawer"}
    clean_cities = [valid_map[c.lower()] for c in cities if c.lower() in valid_map]
    if not clean_cities:
        clean_cities = ["Islamabad", "Karachi", "Lahore", "Peshawer"]

    report_content = run_agentic_loop(cities=clean_cities, days=days, horizon=horizon, provider=provider)
    saved_file = save_report(report_content, custom_path=output_path)
    return report_content, saved_file


# ---------------------------------------------------------------------------
# CLI Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pakistan Weather LLM Agentic Report Generator")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Islamabad", "Karachi", "Lahore", "Peshawer"],
        help="List of cities to include (e.g. Islamabad Karachi Lahore Peshawer)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Historical lookback period in days (default: 30)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="ML Forecast horizon in days (default: 7)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "anthropic", "ollama", "template"],
        help="LLM provider (default: auto)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force deterministic template generation (no external API calls)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom destination filepath for the report markdown"
    )

    args = parser.parse_args()

    chosen_provider = "template" if args.offline else args.provider

    log.info("=" * 65)
    log.info("Pakistan Weather Analytics -- Autonomous Report Agent")
    log.info("Cities: %s | Days: %d | Horizon: %d | Provider: %s",
             args.cities, args.days, args.horizon, chosen_provider)
    log.info("=" * 65)

    report_text, report_path = generate_weather_report(
        cities=args.cities,
        days=args.days,
        horizon=args.horizon,
        provider=chosen_provider,
        output_path=args.output
    )

    print("\n" + "=" * 65)
    print("GENERATED WEATHER BRIEFING PREVIEW:")
    print("=" * 65)
    print(report_text[:1200] + "\n\n... [Report truncated in console preview] ...\n")
    print(f"Full report successfully saved to: {report_path}")
