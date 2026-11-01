"""
Unified Test Suite Runner & Regression Diagnostic System (Intro2SE / Ex0)

Executes all 9 Software Test Plan (STP) Suites and provides:
1. Executive STP Verification Summary Table.
2. Injected Bug & Fault Localization Engine: Pinpoints exact source file, line number,
   and code snippet on failure (generates reports/fault_localization_report.md).
3. Pipeline Stage Performance Profiler & Regression Pinpointer: Compares Stages A, B, C, D
   against a Git-tagged baseline (.perf_baseline.json) and isolates performance culprits.

Usage:
    python run_all_tests.py                         # Standard run (Suites + Profiler comparison)
    python run_all_tests.py --save-baseline         # Records current commit as new baseline
    python run_all_tests.py --skip-profile          # Fast run: executes test suites only
"""

from __future__ import annotations
import sys
import os
import time
import subprocess
import argparse
from typing import List, Tuple, Dict, Any

from diagnostics import FaultLocalizationEngine, FaultDetail
from performance_profiler import PerformanceProfiler, ProfileSummary, BASELINE_FILE


STP_SUITES = [
    {
        "id": "Suite 1",
        "name": "Geometric Validation (Triangles & Trapezoids)",
        "file": "tests/test_geometry.py",
        "keywords": "triangle or trapezoid",
        "description": "Validates isosceles constraints, base ordering, coplanarity, and degeneracy",
    },
    {
        "id": "Suite 2",
        "name": "Mathematical Sanity & Reference Comparison",
        "file": "tests/test_projection.py",
        "keywords": "analytical or equivalence",
        "description": "Compares projections against ground-truth gradient volume and baseline math",
    },
    {
        "id": "Suite 3",
        "name": "NRRD File Integrity & Bounds Validation",
        "file": "tests/test_nrrd_loader.py",
        "keywords": "",
        "description": "Verifies MRHead_1.nrrd, multi-dtypes (uint8/16/32), 1024^3 bounds, and corruption handling",
    },
    {
        "id": "Suite 4",
        "name": "Performance & High-FPS Benchmarking",
        "file": "tests/test_benchmark_500.py tests/test_projection.py",
        "keywords": "benchmark",
        "description": "Verifies >= 30 FPS minimum and >= 60 FPS target on 500^3 volume and MRI head scan",
    },
    {
        "id": "Suite 5",
        "name": "Boundary & Out-of-Bounds Sampling Safety",
        "file": "tests/test_projection.py",
        "keywords": "bounds or exterior",
        "description": "Verifies safe zero-padding for partial and fully exterior planes",
    },
    {
        "id": "Suite 6",
        "name": "Output Video & MP4 Encoding Integrity",
        "file": "tests/test_video_writer.py",
        "keywords": "mp4 or variable or auto",
        "description": "Verifies MP4 container creation, headers, frame count, canvas centering, and playability",
    },
    {
        "id": "Suite 7",
        "name": "Visual Continuity & Non-Blocking Warning",
        "file": "tests/test_geometry.py",
        "keywords": "continuity",
        "description": "Verifies adaptive Euclidean delta check and non-blocking [WARN] diagnostic logging",
    },
    {
        "id": "Suite 8",
        "name": "Geometric & Sampling Edge Cases",
        "file": "tests/test_geometric_edge_cases.py",
        "keywords": "",
        "description": "Verifies triangle apex singularity (d12=0), boundary skimming, and 45-deg oblique planes",
    },
    {
        "id": "Suite 9",
        "name": "Video Quality & Container Edge Cases",
        "file": "tests/test_video_quality_edge_cases.py",
        "keywords": "",
        "description": "Verifies anti-flicker global normalization, jitter-free centering, and 1-frame MP4 container",
    },
]


def run_suite(suite: Dict[str, str]) -> Tuple[bool, float, str]:
    """Runs a single test suite using pytest with local variable introspection."""
    files = suite["file"].split()
    # -l / --showlocals captures local variables in tracebacks for deep diagnostics
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "-l"] + files
    if suite["keywords"]:
        cmd += ["-k", suite["keywords"]]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_elapsed = time.perf_counter() - t0

    passed = (proc.returncode == 0)
    output = proc.stdout if passed else proc.stdout + "\n" + proc.stderr
    return passed, t_elapsed, output


def execute_self_test() -> None:
    """
    Self-Verification Mode ('Testing the Tester'):
    Verifies that the Fault Localizer and Performance Pinpointer work correctly
    under simulated injected bugs and performance regressions without modifying
    any project files on disk.
    """
    sep = "=" * 80
    print(sep)
    print("         DIAGNOSTIC HARNESS SELF-TEST (VERIFYING THE TEST ENGINE)")
    print(sep)
    print("[*] Running non-destructive self-verification simulations...")

    # -------------------------------------------------------------------------
    # Simulation 1: Injected Bug in geometry.py (Line 113)
    # -------------------------------------------------------------------------
    print("\n[SIMULATION 1] Injecting simulated geometric bug into geometry.py:113...")
    simulated_tb = (
        '    def test_triangle():\n'
        '        a = Point3D(0, 0, 0)\n'
        '        b = Point3D(10, 0, 0)\n'
        '        c = Point3D(5, 5, 0)\n'
        'tests/test_geometry.py:55: in test_triangle\n'
        '    validate_triangle(a, b, c)\n'
        'geometry.py:113: in validate_triangle\n'
        '    raise GeometryValidationError(f"Triangle is not isosceles: {diff}")\n'
        'E   geometry.GeometryValidationError: Triangle is not isosceles: 2.500000 >= 0.001\n'
    )
    fault = FaultLocalizationEngine.analyze_failure("Simulated_Suite", "tests/test_geometry.py", simulated_tb)
    print("\n" + FaultLocalizationEngine.format_terminal_box(fault))

    assert fault.source_file == "geometry.py", f"Expected geometry.py, got {fault.source_file}"
    assert fault.source_line == 113, f"Expected line 113, got {fault.source_line}"
    assert "GEOMETRIC_INVARIANT_VIOLATION" in fault.fault_category
    print("\n[+] SIMULATION 1 RESULT: SUCCESS -> Fault Localizer pinpointed geometry.py:113 with 100% precision.")

    # -------------------------------------------------------------------------
    # Simulation 2: Injected 500% Slowdown in projection_engine.py
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[SIMULATION 2] Injecting simulated latency surge in Stage B (projection_engine.py)...")
    baseline = ProfileSummary(
        commit_hash="8931d8c",
        commit_message="Healthy baseline",
        timestamp="2026-11-01 12:00:00",
        stage_a_geometry_ms=0.022,
        stage_b_projection_ms=0.840,
        stage_c_normalization_ms=0.042,
        stage_d_encoding_ms=0.292,
        overall_latency_ms=1.196,
        overall_fps=836.3,
    )
    # Simulated unvectorized loop: Stage B jumps from 0.84ms to 5.20ms (+519%)
    simulated_current = ProfileSummary(
        commit_hash="uncommitted",
        commit_message="Unvectorized loop experiment",
        timestamp="2026-11-01 13:00:00",
        stage_a_geometry_ms=0.023,
        stage_b_projection_ms=5.200,  # Slowdown!
        stage_c_normalization_ms=0.043,
        stage_d_encoding_ms=0.295,
        overall_latency_ms=5.561,
        overall_fps=179.8,
    )
    stages = PerformanceProfiler.analyze_regression(simulated_current, baseline)
    print("\n" + PerformanceProfiler.format_profiling_report(simulated_current, baseline, stages))

    assert stages[0].culprit is False, "Stage A should be healthy"
    assert stages[1].culprit is True, "Stage B must be flagged as culprit"
    assert "projection_engine.py" in stages[1].component
    print("[+] SIMULATION 2 RESULT: SUCCESS -> Performance Pinpointer isolated Stage B (projection_engine.py) as culprit.")

    print("\n" + sep)
    print(">>> DIAGNOSTIC HARNESS SELF-TEST PASSED: ALL DETECTORS FULLY OPERATIONAL <<<")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive STP Verification & Diagnostic Runner for Ex0"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Record the current commit performance as the new reference baseline (.perf_baseline.json)",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip stage-by-stage performance profiling and run test suites only",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-diagnostic verification ('testing the tester') under simulated bugs and slowdowns",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Display the continuous performance telemetry history table",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append this run's metrics to the performance history log",
    )
    args = parser.parse_args()

    if args.show_history:
        print("\n" + PerformanceProfiler.format_history_table())
        sys.exit(0)

    if args.self_test:
        execute_self_test()
        sys.exit(0)

    print("=" * 80)
    print("       3D NRRD PLANE PROJECTION - COMPREHENSIVE STP VERIFICATION RUNNER")
    print("=" * 80)
    print(f"[*] Python Runtime: {sys.version.split()[0]} ({sys.platform})")
    print(f"[*] Working Directory: {os.getcwd()}")
    print(f"[*] Total STP Suites to execute: {len(STP_SUITES)}")
    print("=" * 80)

    overall_start = time.perf_counter()
    results = []
    faults: List[FaultDetail] = []

    # 1. Run All Functional & Verification Test Suites
    for suite in STP_SUITES:
        print(f"\n[RUNNING] {suite['id']}: {suite['name']}...")
        passed, duration, output = run_suite(suite)
        status_str = "PASSED" if passed else "FAILED"
        print(f"[{status_str}] in {duration:.3f}s")
        results.append((suite, passed, duration, output))

        if not passed:
            # Pinpoint exact failure in project source code
            fault = FaultLocalizationEngine.analyze_failure(suite["id"], suite["file"], output)
            faults.append(fault)

    total_time = time.perf_counter() - overall_start

    # 2. Print STP Verification Summary Table
    print("\n" + "=" * 80)
    print("                           STP VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Suite ID':<10} | {'Suite Name':<42} | {'Duration':<10} | {'Status'}")
    print("-" * 80)

    all_passed = True
    for suite, passed, duration, _ in results:
        status_text = "PASS [OK]" if passed else "FAIL [XX]"
        if not passed:
            all_passed = False
        print(f"{suite['id']:<10} | {suite['name']:<42} | {duration:>7.3f}s   | {status_text}")

    print("-" * 80)
    print(f"Total Test Suite Execution Time: {total_time:.3f} seconds")
    print("=" * 80)

    # 3. If any test failed: Display Pinpointed Fault Boxes and Generate Markdown Report
    if not all_passed:
        print("\n" + "!" * 80)
        print("          ONE OR MORE TEST SUITES FAILED -> TRIGGERING FAULT LOCALIZER")
        print("!" * 80)

        for fault in faults:
            print("\n" + FaultLocalizationEngine.format_terminal_box(fault))

        report_path = "reports/fault_localization_report.md"
        FaultLocalizationEngine.generate_markdown_report(faults, report_path)
        print(f"\n[+] Detailed Markdown Diagnostic Report generated at: '{report_path}'")

    # 4. Stage-by-Stage Performance Profiling & Regression Pinpointer
    if not args.skip_profile:
        print("\n[PROFILING] Initiating multi-stage performance micro-benchmarking...")
        current_profile = PerformanceProfiler.profile_all_stages()

        if args.save_baseline:
            PerformanceProfiler.save_baseline(current_profile, BASELINE_FILE)
            baseline_profile = current_profile
            status_tag = "BASELINE_SAVED"
        else:
            baseline_profile = PerformanceProfiler.load_baseline(BASELINE_FILE)
            status_tag = "HEALTHY"

        stages = PerformanceProfiler.analyze_regression(current_profile, baseline_profile)
        if any(s.culprit for s in stages):
            status_tag = "REGRESSION"

        print("\n" + PerformanceProfiler.format_profiling_report(current_profile, baseline_profile, stages))

        # Record run to continuous telemetry history log (CSV + Markdown)
        if not args.no_history:
            PerformanceProfiler.record_history(current_profile, status=status_tag)
            print("[+] Performance telemetry appended to: 'reports/performance_history.csv' & '.md'")

    # 5. Final Status
    if all_passed:
        print(f"\n>>> ALL {len(STP_SUITES)} STP TEST SUITES PASSED STRICT VERIFICATION PROTOCOLS (100% SUCCESS) <<<\n")
        sys.exit(0)
    else:
        print(f"\n>>> TEST RUN FAILED ({len(faults)} failure(s) detected). SEE PINPOINTED DIAGNOSTIC REPORT ABOVE. <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
