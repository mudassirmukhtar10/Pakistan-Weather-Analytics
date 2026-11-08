"""
Tests for Autonomous Reporting Agent Tools, Template Fallback, and Security Isolation.
"""

import os
import pytest
from pathlib import Path

from src.report_agent import (
    get_city_summary,
    compare_cities,
    detect_anomalies,
    get_trend,
    get_forecast,
    generate_template_report,
)


def test_get_city_summary():
    """Verify statistical summary tool returns valid ranges and expected fields."""
    res = get_city_summary("Islamabad")
    assert "error" not in res, f"Tool returned unexpected error: {res}"
    assert res["city"] == "Islamabad"
    assert res["total_records"] > 0

    t = res["temperature"]
    assert t["mean_c"] is not None
    assert t["min_c"] <= t["mean_c"] <= t["max_c"]

    h = res["humidity"]
    assert 0 <= h["mean_pct"] <= 100


def test_compare_cities():
    """Verify cross-city comparison tool computes rankings and thermal spread."""
    res = compare_cities(metric="temperature_2m", cities=["Islamabad", "Karachi"])
    assert "error" not in res
    assert "rankings" in res
    assert len(res["rankings"]) == 2
    assert "spread" in res
    assert res["spread"] >= 0.0


def test_detect_anomalies():
    """Verify statistical anomaly detection tool flags outliers using Z-score and IQR."""
    res = detect_anomalies("Lahore", window=24, threshold=2.5)
    assert "error" not in res
    assert res["city"] == "Lahore"
    assert "anomaly_count" in res
    assert "events" in res
    assert isinstance(res["events"], list)
    if res["events"]:
        first_event = res["events"][0]
        assert "type" in first_event
        assert "severity" in first_event


def test_get_trend():
    """Verify trend direction tool calculates warming/cooling trajectory."""
    res = get_trend("Karachi", metric="temperature_2m", period="weekly")
    assert "error" not in res
    assert res["city"] == "Karachi"
    assert res["trend_direction"] in ["warming", "cooling", "stable"]
    assert "net_change" in res


def test_get_forecast():
    """Verify ML forecast retrieval tool returns projected temperatures."""
    res = get_forecast("Islamabad", horizon=3)
    assert "error" not in res
    assert res["city"] == "Islamabad"
    assert res["horizon_days"] == 3
    assert len(res["daily_breakdown"]) == 3
    assert res["predicted_avg_temp_c"] is not None


def test_generate_template_report():
    """Verify deterministic offline report contains all required sections."""
    cities = ["Islamabad", "Karachi"]
    report_md = generate_template_report(cities=cities, days=14, horizon=5)

    assert "## 1. Executive Summary" in report_md
    assert "## 2. Per-City Highlights" in report_md
    assert "## 3. Cross-City Comparison" in report_md
    assert "## 4. Anomalies & Weather Alerts" in report_md
    assert "## 5. Forecast Outlook" in report_md
    assert "## 6. Actionable Recommendations" in report_md
    for c in cities:
        assert c in report_md


def test_database_port_isolation():
    """
    SECURITY REQUIREMENT:
    Ensure port 3306 (legacy POS database) is NEVER referenced in active code or connection strings.
    Only port 3307 is permitted.
    """
    root_dir = Path(__file__).parent.parent
    src_dir = root_dir / "src"

    for py_file in src_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "3306" not in content, (
            f"SECURITY VIOLATION: Port 3306 found in {py_file}! Port 3306 is legacy POS and must NEVER be referenced."
        )

