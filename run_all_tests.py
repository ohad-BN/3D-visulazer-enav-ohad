"""
Unified Test Suite Runner for 3D NRRD Plane Projection (Ex0)
Executes all 7 Software Test Plan (STP) Suites and outputs an executive verification report.

Usage:
    python run_all_tests.py
"""

from __future__ import annotations
import sys
import os
import time
import subprocess
from typing import List, Tuple, Dict


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
]


def run_suite(suite: Dict[str, str]) -> Tuple[bool, float, str]:
    """Runs a single test suite using pytest."""
    files = suite["file"].split()
    cmd = [sys.executable, "-m", "pytest", "-v"] + files
    if suite["keywords"]:
        cmd += ["-k", suite["keywords"]]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_elapsed = time.perf_counter() - t0

    passed = (proc.returncode == 0)
    output = proc.stdout if passed else proc.stdout + "\n" + proc.stderr
    return passed, t_elapsed, output


def main() -> None:
    print("=" * 80)
    print("       3D NRRD PLANE PROJECTION - COMPREHENSIVE STP VERIFICATION RUNNER")
    print("=" * 80)
    print(f"[*] Python Runtime: {sys.version.split()[0]} ({sys.platform})")
    print(f"[*] Working Directory: {os.getcwd()}")
    print(f"[*] Total STP Suites to execute: {len(STP_SUITES)}")
    print("=" * 80)

    overall_start = time.perf_counter()
    results = []

    for suite in STP_SUITES:
        print(f"\n[RUNNING] {suite['id']}: {suite['name']}...")
        passed, duration, output = run_suite(suite)
        status_str = "PASSED" if passed else "FAILED"
        print(f"[{status_str}] in {duration:.3f}s")
        results.append((suite, passed, duration, output))

    total_time = time.perf_counter() - overall_start

    # Print Executive Summary Table
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
    print(f"Total Verification Execution Time: {total_time:.3f} seconds")
    print("=" * 80)

    if all_passed:
        print("\n>>> ALL 7 STP TEST SUITES PASSED STRICT VERIFICATION PROTOCOLS (100% SUCCESS) <<<\n")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE TEST SUITES FAILED. SEE DETAILS ABOVE. <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
