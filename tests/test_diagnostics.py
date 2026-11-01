"""
Automated Unit Tests for Diagnostic & Performance Profiling Engines
Verifies:
1. FaultLocalizationEngine correctly parses simulated failure tracebacks, extracts line numbers,
   extracts code snippets, classifies fault categories, and writes markdown reports.
2. PerformanceProfiler measures all 4 stages and flags regressions when delta exceeds threshold.
"""

import os
import pytest

from diagnostics import FaultLocalizationEngine, FaultDetail
from performance_profiler import PerformanceProfiler, ProfileSummary, StageMetrics


def test_fault_localization_traceback_parsing(tmp_path):
    """Verify FaultLocalizationEngine isolates exact source file and line from pytest tracebacks."""
    simulated_pytest_output = (
        '    def test_triangle():\n'
        '        a = Point3D(0, 0, 0)\n'
        '        b = Point3D(10, 0, 0)\n'
        '        c = Point3D(5, 5, 0)\n'
        '        validate_triangle(a, b, c)\n'
        'tests/test_geometry.py:55: in test_triangle\n'
        '    validate_triangle(a, b, c)\n'
        'geometry.py:113: in validate_triangle\n'
        '    raise GeometryValidationError(f"Triangle is not isosceles: {diff}")\n'
        'E   geometry.GeometryValidationError: Triangle is not isosceles: 2.5 >= 0.001\n'
    )

    fault = FaultLocalizationEngine.analyze_failure(
        suite_id="Suite 1",
        test_file="tests/test_geometry.py",
        raw_output=simulated_pytest_output
    )

    assert fault.suite_id == "Suite 1"
    assert fault.source_file == "geometry.py"
    assert fault.source_line == 113
    assert "GEOMETRIC_INVARIANT_VIOLATION" in fault.fault_category
    assert fault.code_snippet is not None
    assert "113" in fault.code_snippet

    # Test markdown report generation
    report_file = str(tmp_path / "test_report.md")
    report_text = FaultLocalizationEngine.generate_markdown_report([fault], report_file)
    assert os.path.isfile(report_file)
    assert "geometry.py" in report_text
    assert "Line **113**" in report_text


def test_performance_regression_alert():
    """Verify PerformanceProfiler flags stage as culprit when latency jumps > 30%."""
    baseline = ProfileSummary(
        commit_hash="abc1234",
        commit_message="Initial baseline",
        timestamp="2026-11-01 10:00:00",
        stage_a_geometry_ms=0.020,
        stage_b_projection_ms=1.000,
        stage_c_normalization_ms=0.040,
        stage_d_encoding_ms=0.300,
        overall_latency_ms=1.360,
        overall_fps=735.0,
    )

    # Current has an artificial 5x slowdown in Stage B (Projection math)
    current = ProfileSummary(
        commit_hash="def5678",
        commit_message="Unvectorized loop commit",
        timestamp="2026-11-01 11:00:00",
        stage_a_geometry_ms=0.021,        # ~same
        stage_b_projection_ms=5.000,        # +400% slowdown!
        stage_c_normalization_ms=0.041,    # ~same
        stage_d_encoding_ms=0.310,        # ~same
        overall_latency_ms=5.372,
        overall_fps=186.0,
    )

    stages = PerformanceProfiler.analyze_regression(current, baseline)
    assert len(stages) == 4

    # Stage A: Healthy
    assert stages[0].culprit is False
    assert stages[0].status == "HEALTHY"

    # Stage B: Culprit flagged!
    assert stages[1].culprit is True
    assert stages[1].status == "REGRESSION"
    assert stages[1].delta_pct == pytest.approx(400.0, abs=1.0)
    assert "projection_engine.py" in stages[1].component

    # Report formatting includes alert
    report_str = PerformanceProfiler.format_profiling_report(current, baseline, stages)
    assert "PERFORMANCE REGRESSION PINPOINTED" in report_str
    assert "projection_engine.py" in report_str


def test_performance_history_logging(tmp_path):
    """Verify PerformanceProfiler appends records cleanly to CSV and Markdown history logs."""
    test_csv = str(tmp_path / "test_history.csv")
    test_md = str(tmp_path / "test_history.md")

    sample_summary = ProfileSummary(
        commit_hash="abc1234",
        commit_message="Test run commit",
        timestamp="2026-11-01 12:00:00",
        stage_a_geometry_ms=0.025,
        stage_b_projection_ms=1.100,
        stage_c_normalization_ms=0.050,
        stage_d_encoding_ms=0.350,
        overall_latency_ms=1.525,
        overall_fps=655.7,
    )

    # 1. Record first entry
    PerformanceProfiler.record_history(
        sample_summary, status="HEALTHY", csv_path=test_csv, md_path=test_md
    )
    assert os.path.isfile(test_csv)
    assert os.path.isfile(test_md)

    records = PerformanceProfiler.get_history(csv_path=test_csv)
    assert len(records) == 1
    assert records[0]["commit_hash"] == "abc1234"
    assert records[0]["overall_fps"] == "655.7"

    # 2. Record second entry
    sample_summary2 = ProfileSummary(
        commit_hash="def5678",
        commit_message="Second test run",
        timestamp="2026-11-01 13:00:00",
        stage_a_geometry_ms=0.024,
        stage_b_projection_ms=1.050,
        stage_c_normalization_ms=0.048,
        stage_d_encoding_ms=0.340,
        overall_latency_ms=1.462,
        overall_fps=684.0,
    )
    PerformanceProfiler.record_history(
        sample_summary2, status="BASELINE_SAVED", csv_path=test_csv, md_path=test_md
    )
    records2 = PerformanceProfiler.get_history(csv_path=test_csv)
    assert len(records2) == 2
    assert records2[1]["commit_hash"] == "def5678"
    assert records2[1]["status"] == "BASELINE_SAVED"

    # Check terminal formatting
    table_str = PerformanceProfiler.format_history_table(csv_path=test_csv)
    assert "abc1234" in table_str
    assert "def5678" in table_str
