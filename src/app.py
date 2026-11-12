import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine

# Import report agent for autonomous weather briefings
try:
    from report_agent import generate_weather_report, REPORTS_DIR
except ImportError:
    try:
        from src.report_agent import generate_weather_report, REPORTS_DIR
    except ImportError:
        generate_weather_report = None
        REPORTS_DIR = Path(__file__).parent.parent / "reports"

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistan Weather & Geospatial Analytics",
    page_icon="⛅",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR    = Path(__file__).parent
ROOT_DIR   = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH   = MODELS_DIR / "temp_forecaster.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
DATA_CSV_PATH = ROOT_DIR / "weather_data.csv"

# Add src to path so we can import train_model helpers
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Standard geographical coordinates for Pakistan cities (used for mapping & realignment)
CITY_COORDINATES = {
    "islamabad": (33.70826, 73.00621, 583.0),
    "karachi":   (24.850615, 66.992485, 8.0),
    "lahore":    (31.599297, 74.371254, 219.0),
    "peshawer":  (33.989452, 71.57304, 340.0),
}

# Consistent visual color palette for cities
CITY_COLORS = {
    "Islamabad": "#1f77b4",  # Steel Blue
    "Karachi":   "#ff7f0e",  # Safety Orange
    "Lahore":    "#2ca02c",  # Emerald Green
    "Peshawer":  "#9467bd",  # Royal Purple
}

# ---------------------------------------------------------------------------
# Robust Data Cleaning on Load
# ---------------------------------------------------------------------------

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw weather data loaded from MySQL:
      1. Drops metadata rows from the 'city' column ('Cleaned_weather', 'none', 'city', empty strings).
      2. Detects and resolves misaligned column layouts (Open-Meteo multi-header format).
      3. Drops any residual header strings or string artifacts from data rows.
      4. Coerces all meteorological & coordinate columns to numeric via pd.to_numeric(..., errors='coerce').
      5. Parses datetime values safely.
    """
    if df.empty:
        return df

    # 1. Remove metadata / invalid cities
    invalid_city_names = {"cleaned_weather", "none", "city", "", "nan", "null"}
    if "city" in df.columns:
        df = df[
            df["city"].notna() &
            ~df["city"].astype(str).str.strip().str.lower().isin(invalid_city_names)
        ].copy()

    if df.empty:
        return df

    coords_lookup = dict(CITY_COORDINATES)

    # Check if table columns are in the misaligned layout:
    # Columns: ['latitude', 'longitude', 'elevation', 'utc_offset_seconds', 'timezone', ...]
    # where row values contain strings like 'time', 'temperature_2m', or timestamps '2025-...'
    has_temp_col = any("temp" in str(c).lower() for c in df.columns)

    # Dynamically extract coordinates from metadata rows if present
    if "latitude" in df.columns and "timezone" in df.columns and not has_temp_col:
        meta_mask = (
            df["latitude"].astype(str).str.match(r"^\d+\.\d+$") &
            (df["timezone"].astype(str).str.strip() == "GMT")
        )
        for _, row in df[meta_mask].iterrows():
            c_key = str(row["city"]).strip().lower()
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                elev = float(row.get("elevation", 0))
                coords_lookup[c_key] = (lat, lon, elev)
            except (ValueError, TypeError):
                pass

    if not has_temp_col and "latitude" in df.columns and "longitude" in df.columns:
        # Detect header rows (where latitude == 'time' or longitude contains 'temp' or 'weather_code')
        header_mask = (
            df["latitude"].astype(str).str.strip().str.lower().isin(["time", "latitude"]) |
            df["longitude"].astype(str).str.lower().str.contains("temp|weather_code|longitude", regex=True)
        )

        # Detect rows where latitude contains a timestamp or date: '2025-...'
        is_data_row = (
            df["latitude"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}") &
            ~header_mask
        )

        # Keep pure hourly rows (containing 'T' in timestamp) if available for full hourly alignment
        hourly_mask = is_data_row & df["latitude"].astype(str).str.contains("T", na=False)
        if hourly_mask.any():
            data_df = df[hourly_mask].copy()
        else:
            data_df = df[is_data_row].copy()

        # Realign misaligned columns
        rename_map = {
            "latitude": "time",
            "longitude": "temperature_2m",
            "elevation": "wind_speed_10m",
            "utc_offset_seconds": "relative_humidity_2m",
            "timezone": "precipitation"
        }
        data_df.rename(columns=rename_map, inplace=True)

        # Populate true geographical coordinates by city
        data_df["latitude"] = data_df["city"].apply(
            lambda c: coords_lookup.get(str(c).strip().lower(), (np.nan, np.nan, np.nan))[0]
        )
        data_df["longitude"] = data_df["city"].apply(
            lambda c: coords_lookup.get(str(c).strip().lower(), (np.nan, np.nan, np.nan))[1]
        )
        data_df["elevation"] = data_df["city"].apply(
            lambda c: coords_lookup.get(str(c).strip().lower(), (np.nan, np.nan, np.nan))[2]
        )
        df = data_df
    else:
        # General cleaning: drop any row where any column contains header string values
        header_vals = {
            "time", "latitude", "longitude", "elevation", "temperature",
            "wind_speed", "humidity", "none", "null", "nan"
        }
        mask_drop = pd.Series(False, index=df.index)
        for col in df.columns:
            str_col = df[col].astype(str).str.strip().str.lower()
            mask_drop |= str_col.isin(header_vals) | str_col.str.contains(r"\(°c\)|\(km/h\)|\(%\)", regex=True)
        df = df[~mask_drop].copy()

    # 3. Numeric conversion on all meteorological and geospatial columns
    numeric_keywords = ["temp", "humid", "wind", "precip", "elevation", "latitude", "longitude"]
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in numeric_keywords):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Clean and parse datetime column
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])

    # Reorder columns logically for clean display
    preferred_order = [
        "time", "city", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "precipitation", "latitude", "longitude", "elevation"
    ]
    existing_cols = [c for c in preferred_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in preferred_order and not c.startswith("_")]
    df = df[existing_cols + extra_cols]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load weather data directly from weather_data.csv and apply robust data cleaning."""
    try:
        if DATA_CSV_PATH.exists():
            raw_df = pd.read_csv(DATA_CSV_PATH, low_memory=False)
            return clean_weather_data(raw_df)
        
        # Fallback to cleaned_weather.csv in Data/ directory if root csv is missing
        backup_csv = ROOT_DIR / "Data" / "cleaned_weather.csv"
        if backup_csv.exists():
            raw_df = pd.read_csv(backup_csv, low_memory=False)
            return clean_weather_data(raw_df)

        st.error(f"Weather dataset not found at {DATA_CSV_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading weather data from CSV: {e}")
        return pd.DataFrame()


@st.cache_data
def load_history_from_csv() -> pd.DataFrame:
    """
    Load clean hourly history from city CSVs for forecasting charts.
    """
    def _parse(path, city):
        df = pd.read_csv(
            path, skiprows=3, header=None, on_bad_lines="skip",
            names=["time", "temperature_2m", "wind_speed_10m",
                   "relative_humidity_2m", "precipitation", "_drop"],
        )
        df.drop(columns=["_drop"], inplace=True, errors="ignore")
        df["city"] = city
        df = df[df["time"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy()
        df["time"] = pd.to_datetime(df["time"], format="mixed")
        df["temperature_2m"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
        return df

    city_files = {
        "Islamabad": ROOT_DIR / "Data" / "islamabad.csv",
        "Karachi":   ROOT_DIR / "Data" / "karachi.csv",
        "Lahore":    ROOT_DIR / "Data" / "lahore.csv",
        "Peshawer":  ROOT_DIR / "Data" / "peshawer.csv",
    }
    frames = []
    for city, path in city_files.items():
        if path.exists():
            frames.append(_parse(path, city))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_resource
def load_model():
    """Load the saved model artifact with st.cache_resource (singleton)."""
    import joblib
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics() -> list:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Helper Functions: Column Matching, Metric Calculations, and Trends
# ---------------------------------------------------------------------------

def find_column(columns, keywords: list) -> str | None:
    """Safely match a column by case-insensitive keyword search."""
    for kw in keywords:
        matches = [c for c in columns if kw in str(c).lower()]
        if matches:
            return matches[0]
    return None


def safe_metric_mean(df: pd.DataFrame, col_name: str | None) -> float | None:
    """Compute mean safely after numeric coercion, ignoring NaNs."""
    if col_name and col_name in df.columns:
        series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if not series.empty:
            val = series.mean()
            if pd.notnull(val):
                return float(val)
    return None


@st.cache_data
def compute_monthly_temperature_trends(df_data: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    """Compute monthly average temperature per city (cached)."""
    if "time" not in df_data.columns or temp_col not in df_data.columns:
        return pd.DataFrame()
    subset = df_data[["time", "city", temp_col]].dropna().copy()
    subset["month_year"] = subset["time"].dt.to_period("M").astype(str)
    monthly = (
        subset.groupby(["month_year", "city"])[temp_col]
        .mean()
        .reset_index()
        .round(2)
    )
    return monthly


def generate_comparative_insights(df_comp: pd.DataFrame, temp_col: str, hum_col: str, wind_col: str) -> list[str]:
    """Generate dynamic textual insights comparing cities within the filtered dataset."""
    insights = []
    if df_comp.empty or "city" not in df_comp.columns:
        return insights

    agg_dict = {}
    if temp_col and temp_col in df_comp.columns:
        agg_dict[temp_col] = ["mean", "min", "max", "std"]
    if hum_col and hum_col in df_comp.columns:
        agg_dict[hum_col] = ["mean", "min", "max"]
    if wind_col and wind_col in df_comp.columns:
        agg_dict[wind_col] = ["mean", "max"]

    if not agg_dict:
        return insights

    grouped = df_comp.groupby("city").agg(agg_dict)
    if grouped.empty:
        return insights

    # Flatten multiindex columns for clean indexing
    flat_stats = pd.DataFrame(index=grouped.index)
    if temp_col and temp_col in df_comp.columns:
        flat_stats["avg_temp"] = grouped[temp_col]["mean"]
        flat_stats["min_temp"] = grouped[temp_col]["min"]
        flat_stats["max_temp"] = grouped[temp_col]["max"]
        flat_stats["std_temp"] = grouped[temp_col]["std"]
    if hum_col and hum_col in df_comp.columns:
        flat_stats["avg_hum"]  = grouped[hum_col]["mean"]
        flat_stats["min_hum"]  = grouped[hum_col]["min"]
        flat_stats["max_hum"]  = grouped[hum_col]["max"]
    if wind_col and wind_col in df_comp.columns:
        flat_stats["avg_wind"] = grouped[wind_col]["mean"]
        flat_stats["max_wind"] = grouped[wind_col]["max"]

    flat_stats = flat_stats.dropna(how="all")

    if len(flat_stats) > 1:
        # Multi-city comparisons
        if "avg_temp" in flat_stats:
            hot_c = flat_stats["avg_temp"].idxmax()
            hot_v = flat_stats.loc[hot_c, "avg_temp"]
            cool_c = flat_stats["avg_temp"].idxmin()
            cool_v = flat_stats.loc[cool_c, "avg_temp"]
            insights.append(
                f"🌡️ **Temperature**: **{hot_c}** recorded the warmest overall average at **{hot_v:.1f} °C**, "
                f"while **{cool_c}** had the coolest average at **{cool_v:.1f} °C**."
            )

        if "avg_hum" in flat_stats:
            hum_c = flat_stats["avg_hum"].idxmax()
            hum_v = flat_stats.loc[hum_c, "avg_hum"]
            dry_c = flat_stats["avg_hum"].idxmin()
            dry_v = flat_stats.loc[dry_c, "avg_hum"]
            insights.append(
                f"💧 **Humidity**: **{hum_c}** experienced the highest average humidity at **{hum_v:.1f}%**, "
                f"whereas **{dry_c}** was the driest on average at **{dry_v:.1f}%**."
            )

        if "std_temp" in flat_stats:
            vol_c = flat_stats["std_temp"].idxmax()
            vol_v = flat_stats.loc[vol_c, "std_temp"]
            stab_c = flat_stats["std_temp"].idxmin()
            stab_v = flat_stats.loc[stab_c, "std_temp"]
            insights.append(
                f"📊 **Climate Stability**: **{vol_c}** showed the widest seasonal temperature swings (std dev: **{vol_v:.1f} °C**), "
                f"compared to **{stab_c}** which maintained the most stable climate (std dev: **{stab_v:.1f} °C**)."
            )

        if "avg_wind" in flat_stats:
            wind_c = flat_stats["avg_wind"].idxmax()
            wind_v = flat_stats.loc[wind_c, "avg_wind"]
            insights.append(
                f"🌬️ **Wind Speeds**: **{wind_c}** observed the highest average wind speeds (**{wind_v:.1f} km/h**)."
            )
    elif len(flat_stats) == 1:
        # Single city summary
        c_name = flat_stats.index[0]
        avg_t = flat_stats.loc[c_name, "avg_temp"] if "avg_temp" in flat_stats else None
        avg_h = flat_stats.loc[c_name, "avg_hum"] if "avg_hum" in flat_stats else None
        avg_w = flat_stats.loc[c_name, "avg_wind"] if "avg_wind" in flat_stats else None
        summary_parts = []
        if avg_t is not None:
            summary_parts.append(f"average temperature: **{avg_t:.1f} °C**")
        if avg_h is not None:
            summary_parts.append(f"average humidity: **{avg_h:.1f}%**")
        if avg_w is not None:
            summary_parts.append(f"average wind speed: **{avg_w:.1f} km/h**")
        insights.append(f"📍 **{c_name} Profile**: " + ", ".join(summary_parts) + ".")

    return insights


# ---------------------------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------------------------
st.title("🇵🇰 Pakistan Weather & Geospatial Analytics Dashboard")
st.markdown("Explore weather patterns, key metrics, cross-city comparisons, and ML temperature forecasts.")

# ---------------------------------------------------------------------------
# Load and Validate Data
# ---------------------------------------------------------------------------
df = load_data()

# ---------------------------------------------------------------------------
# Global Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Dashboard Filters")

if not df.empty and "city" in df.columns:
    valid_cities = sorted([
        str(c) for c in df["city"].dropna().unique()
        if str(c).strip().lower() not in ["cleaned_weather", "none", "city", "", "nan"]
    ])
    cities = ["All"] + valid_cities

    selected_city = st.sidebar.selectbox(
        "Select City",
        options=cities,
        index=0,
        help="Type to search or click the arrow to view all cities"
    )

    if selected_city != "All":
        filtered_df = df[df["city"] == selected_city].copy()
    else:
        filtered_df = df[df["city"].isin(valid_cities)].copy()
else:
    valid_cities = []
    filtered_df = df.copy()

# Date range filter (applies across Overview and Comparative Analysis tabs)
if not df.empty and "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]):
    min_date = df["time"].dropna().dt.date.min()
    max_date = df["time"].dropna().dt.date.max()

    if pd.notnull(min_date) and pd.notnull(max_date):
        selected_date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter weather records within a specific time period"
        )

        if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = filtered_df[
                (filtered_df["time"].dt.date >= start_date) &
                (filtered_df["time"].dt.date <= end_date)
            ]
        elif isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 1:
            start_date = selected_date_range[0]
            filtered_df = filtered_df[filtered_df["time"].dt.date >= start_date]

# ---------------------------------------------------------------------------
# Tabs: Overview | Comparative Analysis | Forecast | AI Report
# ---------------------------------------------------------------------------
tab_overview, tab_comp, tab_forecast, tab_report = st.tabs([
    "📊 Overview & Map",
    "📈 Comparative Analysis",
    "🔮 Temperature Forecast",
    "🤖 AI Weather Briefing"
])

# ===========================================================================
# TAB 1 — Overview & Geospatial Analytics
# ===========================================================================
with tab_overview:
    if not filtered_df.empty:
        # Dynamic Column Identification
        temp_col = find_column(filtered_df.columns, ["temp", "temperature"])
        humidity_col = find_column(filtered_df.columns, ["humid", "relative_humidity", "humidity"])
        wind_col = find_column(filtered_df.columns, ["wind", "wind_speed"])

        # Display Top Metrics Cards with Safe Calculations
        st.subheader("Key Weather Metrics")
        col1, col2, col3 = st.columns(3)

        avg_temp = safe_metric_mean(filtered_df, temp_col)
        avg_hum  = safe_metric_mean(filtered_df, humidity_col)
        avg_wind = safe_metric_mean(filtered_df, wind_col)

        with col1:
            st.metric(
                label="Average Temperature",
                value=f"{avg_temp:.2f} °C" if avg_temp is not None else "N/A"
            )

        with col2:
            st.metric(
                label="Average Humidity",
                value=f"{avg_hum:.2f} %" if avg_hum is not None else "N/A"
            )

        with col3:
            st.metric(
                label="Average Wind Speed",
                value=f"{avg_wind:.2f} km/h" if avg_wind is not None else "N/A"
            )

        # Data Table Preview
        st.subheader("Dataset Preview")
        st.dataframe(filtered_df, use_container_width=True)

        # Geospatial Map Visualization
        lat_col = find_column(filtered_df.columns, ["latitude", "lat"])
        lon_col = find_column(filtered_df.columns, ["longitude", "lon"])

        if lat_col and lon_col:
            st.subheader("Geospatial Weather Map")
            map_df = filtered_df[[lat_col, lon_col]].dropna().copy()
            map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors="coerce")
            map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors="coerce")
            map_df = map_df.dropna().rename(columns={lat_col: "latitude", lon_col: "longitude"})

            # Clean coordinates within Pakistan bounding box
            map_df = map_df[
                (map_df["latitude"] >= 23.0) & (map_df["latitude"] <= 37.0) &
                (map_df["longitude"] >= 60.0) & (map_df["longitude"] <= 78.0)
            ].drop_duplicates()

            if not map_df.empty:
                st.map(map_df)
            else:
                st.info("No valid latitude/longitude coordinates available for map rendering.")
    else:
        st.warning("No weather records available for the selected city and date range filter.")


# ===========================================================================
# TAB 2 — Comparative Analysis
# ===========================================================================
with tab_comp:
    st.header("📈 Cross-City Comparative Analysis")
    st.markdown("Compare weather distributions, temperature and humidity variations, and historical monthly trends across cities.")

    if not filtered_df.empty:
        temp_col = find_column(filtered_df.columns, ["temp", "temperature"])
        humidity_col = find_column(filtered_df.columns, ["humid", "relative_humidity", "humidity"])
        wind_col = find_column(filtered_df.columns, ["wind", "wind_speed"])

        available_cities = sorted(filtered_df["city"].dropna().unique().tolist())

        # -------------------------------------------------------------------
        # 1. Distribution Boxplots (Temperature & Humidity side-by-side)
        # -------------------------------------------------------------------
        st.subheader("Distribution Analysis")
        box_col1, box_col2 = st.columns(2)

        with box_col1:
            if temp_col and temp_col in filtered_df.columns:
                fig_tbox = go.Figure()
                for c in available_cities:
                    c_vals = filtered_df[filtered_df["city"] == c][temp_col].dropna()
                    fig_tbox.add_trace(go.Box(
                        y=c_vals,
                        name=c,
                        marker_color=CITY_COLORS.get(c, "#1f77b4"),
                        boxpoints="outliers",
                        line_width=1.5,
                    ))
                fig_tbox.update_layout(
                    title="Temperature Distribution (°C)",
                    yaxis_title="Temperature (°C)",
                    template="plotly_white",
                    height=420,
                    showlegend=False,
                )
                st.plotly_chart(fig_tbox, use_container_width=True)
            else:
                st.info("Temperature column not found for distribution boxplot.")

        with box_col2:
            if humidity_col and humidity_col in filtered_df.columns:
                fig_hbox = go.Figure()
                for c in available_cities:
                    c_vals = filtered_df[filtered_df["city"] == c][humidity_col].dropna()
                    fig_hbox.add_trace(go.Box(
                        y=c_vals,
                        name=c,
                        marker_color=CITY_COLORS.get(c, "#ff7f0e"),
                        boxpoints="outliers",
                        line_width=1.5,
                    ))
                fig_hbox.update_layout(
                    title="Relative Humidity Distribution (%)",
                    yaxis_title="Relative Humidity (%)",
                    template="plotly_white",
                    height=420,
                    showlegend=False,
                )
                st.plotly_chart(fig_hbox, use_container_width=True)
            else:
                st.info("Humidity column not found for distribution boxplot.")

        # -------------------------------------------------------------------
        # 2. Grouped Bar Chart: Average Temperature & Humidity per City
        # -------------------------------------------------------------------
        st.subheader("Average Metrics Comparison")
        if temp_col and humidity_col and temp_col in filtered_df.columns and humidity_col in filtered_df.columns:
            avg_summary = (
                filtered_df.groupby("city")
                .agg(
                    avg_temp=(temp_col, "mean"),
                    avg_hum=(humidity_col, "mean")
                )
                .reindex(available_cities)
                .dropna(how="all")
                .reset_index()
            )

            fig_grouped = go.Figure()
            fig_grouped.add_trace(go.Bar(
                x=avg_summary["city"],
                y=avg_summary["avg_temp"].round(2),
                name="Avg Temperature (°C)",
                marker_color="#2b5c8f",
                text=avg_summary["avg_temp"].round(1).astype(str) + " °C",
                textposition="outside",
            ))
            fig_grouped.add_trace(go.Bar(
                x=avg_summary["city"],
                y=avg_summary["avg_hum"].round(2),
                name="Avg Humidity (%)",
                marker_color="#20c997",
                text=avg_summary["avg_hum"].round(1).astype(str) + " %",
                textposition="outside",
            ))
            fig_grouped.update_layout(
                title="Average Temperature and Humidity per City",
                barmode="group",
                yaxis_title="Metric Value (°C / %)",
                template="plotly_white",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_grouped, use_container_width=True)
        else:
            st.info("Insufficient metrics data to construct grouped bar comparison.")

        # -------------------------------------------------------------------
        # 3. Monthly Average Temperature Trends (Line Chart)
        # -------------------------------------------------------------------
        st.subheader("Monthly Average Temperature Trends")
        if temp_col and "time" in filtered_df.columns and temp_col in filtered_df.columns:
            monthly_data = compute_monthly_temperature_trends(filtered_df, temp_col)
            if not monthly_data.empty:
                fig_monthly = go.Figure()
                for c in available_cities:
                    c_month = monthly_data[monthly_data["city"] == c].sort_values("month_year")
                    if not c_month.empty:
                        fig_monthly.add_trace(go.Scatter(
                            x=c_month["month_year"],
                            y=c_month[temp_col],
                            mode="lines+markers",
                            name=c,
                            line=dict(color=CITY_COLORS.get(c, "#1f77b4"), width=2.5),
                            marker=dict(size=6),
                        ))
                fig_monthly.update_layout(
                    title="Monthly Average Temperature Across Cities",
                    xaxis_title="Month",
                    yaxis_title="Average Temperature (°C)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("No monthly temperature data available for current date range.")

        # -------------------------------------------------------------------
        # 4. Auto-Generated Meteorological Insights
        # -------------------------------------------------------------------
        st.subheader("💡 Key Meteorological Insights")
        insights = generate_comparative_insights(filtered_df, temp_col, humidity_col, wind_col)
        if insights:
            for item in insights:
                st.markdown(f"- {item}")
        else:
            st.info("Not enough data to generate comparative insights.")

    else:
        st.warning("No weather records available for the selected city and date range filter.")


# ===========================================================================
# TAB 3 — Temperature Forecast
# ===========================================================================
with tab_forecast:
    st.header("🌡️ ML Temperature Forecast")

    model_artifact = load_model()

    if model_artifact is None:
        st.warning(
            "No trained model found. Run the training script first:\n\n"
            "```bash\npython src/train_model.py\n```"
        )
    else:
        # --- Controls ---
        CITY_ORDER = model_artifact.get("city_order", ["Islamabad", "Karachi", "Lahore", "Peshawer"])

        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        with ctrl_col1:
            forecast_city = st.selectbox("Select City", CITY_ORDER, key="fc_city")
            horizon_days  = st.slider(
                "Forecast Horizon (days)", min_value=1, max_value=14, value=7, key="fc_horizon"
            )
            history_days  = st.slider(
                "Show Historical Days", min_value=7, max_value=90, value=30, key="fc_history"
            )

        # --- Generate forecast ---
        forecast_df = None
        try:
            from train_model import predict_next
            forecast_df = predict_next(city=forecast_city, horizon_days=horizon_days)
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")

        # --- Load historical data ---
        hist_df = load_history_from_csv()
        if hist_df.empty:
            st.warning("Could not load historical CSV data for chart.")

        if forecast_df is not None and not hist_df.empty:
            city_hist = (
                hist_df[hist_df["city"] == forecast_city]
                .sort_values("time")
                .tail(history_days * 24)
            )

            # --- Plotly chart: history + forecast ---
            fig = go.Figure()

            # Historical line
            fig.add_trace(go.Scatter(
                x=city_hist["time"],
                y=city_hist["temperature_2m"],
                mode="lines",
                name="Historical Temp (°C)",
                line=dict(color="#1f77b4", width=1.5),
                opacity=0.85,
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df["time"],
                y=forecast_df["predicted_temperature"],
                mode="lines",
                name="Forecast Temp (°C)",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            ))

            # Vertical divider at forecast start
            split_time = forecast_df["time"].iloc[0]
            fig.add_shape(
                type="line",
                x0=split_time, x1=split_time,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="gray", width=1.5, dash="dot"),
            )
            fig.add_annotation(
                x=split_time,
                y=1,
                yref="paper",
                text="Forecast Start",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color="gray", size=11),
            )

            fig.update_layout(
                title=f"{forecast_city} — Temperature: Last {history_days}d History + {horizon_days}d Forecast",
                xaxis_title="Date / Time",
                yaxis_title="Temperature (°C)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500,
                template="plotly_white",
            )

            with ctrl_col2:
                # Forecast summary stats
                avg_fc = forecast_df["predicted_temperature"].mean()
                max_fc = forecast_df["predicted_temperature"].max()
                min_fc = forecast_df["predicted_temperature"].min()

                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Forecast Temp", f"{avg_fc:.1f} °C")
                m2.metric("Max Forecast Temp", f"{max_fc:.1f} °C")
                m3.metric("Min Forecast Temp", f"{min_fc:.1f} °C")

            st.plotly_chart(fig, use_container_width=True)

            # Forecast data table (daily aggregation)
            st.subheader("📋 Daily Forecast Summary")
            daily = (
                forecast_df
                .set_index("time")
                .resample("D")["predicted_temperature"]
                .agg(["mean", "min", "max"])
                .rename(columns={"mean": "Avg (°C)", "min": "Min (°C)", "max": "Max (°C)"})
                .round(2)
            )
            daily.index = daily.index.strftime("%Y-%m-%d")
            st.dataframe(daily, use_container_width=True)

            # Forecast raw data expander
            with st.expander("Show Raw Hourly Forecast Data"):
                st.dataframe(forecast_df.rename(columns={
                    "time": "Datetime",
                    "predicted_temperature": "Predicted Temp (°C)"
                }), use_container_width=True)

        # --- Model Metrics ---
        st.subheader("📈 Model Performance Metrics (Test Set)")
        metrics = load_metrics()
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(
                metrics_df.style
                .highlight_min(subset=["MAE"],  color="#c6efce", axis=0)
                .highlight_min(subset=["RMSE"], color="#c6efce", axis=0)
                .highlight_max(subset=["R2"],   color="#c6efce", axis=0),
                use_container_width=True,
            )

            # Bar chart comparison
            fig_m = go.Figure()
            for metric in ["MAE", "RMSE"]:
                fig_m.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df["model"],
                    y=metrics_df[metric],
                    text=metrics_df[metric].round(3),
                    textposition="outside",
                ))
            fig_m.update_layout(
                title="MAE & RMSE Comparison (lower is better)",
                barmode="group",
                height=350,
                template="plotly_white",
                yaxis_title="Error (°C)",
            )
            st.plotly_chart(fig_m, use_container_width=True)

            st.caption(
                "**Chronological 80/20 split** — no random shuffle to prevent data leakage. "
                "Green cells = best score per column."
            )
        else:
            st.info("Metrics not found. Run `python src/train_model.py` to generate them.")

# ===========================================================================
# TAB 4 — LLM-Based Agentic Report Generator
# ===========================================================================
with tab_report:
    st.header("🤖 Autonomous AI Weather Intelligence Briefing")
    st.markdown(
        "An autonomous agent analyzes database records via safe, read-only Python tools (no raw SQL), "
        "detects statistical anomalies, calculates ML forecasts, and synthesizes a comprehensive briefing in Markdown."
    )

    # Configuration controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 2, 2, 2])

    available_briefing_cities = ["Islamabad", "Karachi", "Lahore", "Peshawer"] if not valid_cities else valid_cities
    with ctrl_col1:
        selected_report_cities = st.multiselect(
            "Select Cities for Briefing",
            options=available_briefing_cities,
            default=available_briefing_cities,
            help="Select one or more Pakistani cities to include in the briefing."
        )

    with ctrl_col2:
        report_lookback_days = st.slider(
            "Lookback Period (Days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of historical days to inspect for trends and statistics."
        )

    with ctrl_col3:
        report_forecast_horizon = st.slider(
            "Forecast Horizon (Days)",
            min_value=1,
            max_value=14,
            value=7,
            step=1,
            help="ML autoregressive temperature projection window."
        )

    with ctrl_col4:
        selected_provider = st.selectbox(
            "LLM Provider",
            options=["auto", "openai", "anthropic", "ollama", "template"],
            index=0,
            help="Pluggable LLM provider. 'auto' selects configured API key or falls back to deterministic template."
        )

    # Generation trigger button
    b_col1, b_col2 = st.columns([2, 5])
    with b_col1:
        generate_btn = st.button("⚡ Generate Briefing", type="primary", use_container_width=True)
    with b_col2:
        st.caption(
            "🔒 **Tool-Calling Architecture**: Read-only Python tools retrieve verified statistics. "
            "Never executes raw SQL. Offline deterministic fallback is 100% functional without API keys."
        )

    # Report Generation Process
    if generate_btn:
        if not selected_report_cities:
            st.warning("Please select at least one city to generate a briefing.")
        elif generate_weather_report is None:
            st.error("Report generator module not found. Check `src/report_agent.py`.")
        else:
            with st.spinner(f"🤖 Agent executing tools for {', '.join(selected_report_cities)} (lookback: {report_lookback_days}d, forecast: {report_forecast_horizon}d)..."):
                try:
                    rep_content, rep_path = generate_weather_report(
                        cities=selected_report_cities,
                        days=report_lookback_days,
                        horizon=report_forecast_horizon,
                        provider=selected_provider,
                    )
                    st.session_state["latest_weather_report"] = rep_content
                    st.session_state["latest_weather_report_path"] = str(rep_path)
                    st.session_state["latest_weather_report_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("Weather briefing successfully generated and saved!")
                except Exception as exc:
                    st.error(f"Error during report generation: {exc}")

    # Check session state or disk for the most recent report to display
    report_display_text = st.session_state.get("latest_weather_report")
    report_display_path = st.session_state.get("latest_weather_report_path")
    report_display_time = st.session_state.get("latest_weather_report_timestamp", "")

    # If not in session state, attempt loading today's or the latest report from reports/ directory
    if not report_display_text and REPORTS_DIR.exists():
        existing_reports = sorted(REPORTS_DIR.glob("weather_briefing_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing_reports:
            latest_file = existing_reports[0]
            try:
                report_display_text = latest_file.read_text(encoding="utf-8")
                report_display_path = str(latest_file)
                report_display_time = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

    st.divider()

    if report_display_text:
        # Action bar: Info & Download
        action_col1, action_col2 = st.columns([5, 2])
        with action_col1:
            st.markdown(
                f"**Status:** Briefing ready *(Generated: {report_display_time} | File: `{Path(report_display_path).name}`)*"
            )
        with action_col2:
            report_filename = Path(report_display_path).name if report_display_path else "weather_briefing.md"
            st.download_button(
                label="📥 Download Markdown Report",
                data=report_display_text,
                file_name=report_filename,
                mime="text/markdown",
                use_container_width=True,
            )

        # Rendered Markdown Report Container
        with st.container():
            st.markdown(report_display_text)
    else:
        st.info("ℹ️ No weather briefing generated yet. Configure options above and click **'⚡ Generate Briefing'** to start.")
