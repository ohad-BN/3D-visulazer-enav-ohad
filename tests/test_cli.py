"""
Automated Test Suite for CLI Runner & Pipeline Integration (Phase 6)
Covers:
- CLI argument validation and error messaging
- End-to-end pipeline execution (NRRD -> CSV -> Vectorized Projection -> MP4)
- Subprocess execution tests verifying CLI exit codes
- High-resolution FPS profiler output verification
"""

import os
import sys
import subprocess
import tempfile
import pytest
import numpy as np
import nrrd

from Ex0 import validate_cli_arguments, run_pipeline
from video_writer import inspect_video_file


def create_test_dataset(tmp_path):
    """Creates a temporary valid NRRD volume and trajectory CSV."""
    # 1. Create small 30x30x30 uint16 NRRD volume
    data = np.arange(30 * 30 * 30, dtype=np.uint16).reshape((30, 30, 30))
    nrrd_file = tmp_path / "test_vol.nrrd"
    nrrd.write(str(nrrd_file), data)

    # 2. Create trajectory CSV with 5 valid planes
    csv_file = tmp_path / "test_traj.csv"
    csv_content = """# Test Trajectory
# Triangle
0.0, 15.0, 5.0, -10.0, 0.0, 5.0, 10.0, 0.0, 5.0
# Trapezoid
2.0, 12.0, 10.0, 8.0, 12.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 10.0
# Rectangle
0.0, 10.0, 15.0, 10.0, 10.0, 15.0, 0.0, 0.0, 15.0, 10.0, 0.0, 15.0
# Triangle
0.0, 15.0, 20.0, -10.0, 0.0, 20.0, 10.0, 0.0, 20.0
# Rectangle
0.0, 10.0, 25.0, 10.0, 10.0, 25.0, 0.0, 0.0, 25.0, 10.0, 0.0, 25.0
"""
    csv_file.write_text(csv_content, encoding="utf-8")

    return str(nrrd_file), str(csv_file)


# ============================================================================
# 1. CLI Argument Parsing Tests
# ============================================================================

def test_validate_cli_arguments_missing():
    """Verify ValueError when wrong number of arguments are passed."""
    with pytest.raises(ValueError, match="Expected exactly 3 arguments"):
        validate_cli_arguments(["Ex0.py"])

    with pytest.raises(ValueError, match="Expected exactly 3 arguments"):
        validate_cli_arguments(["Ex0.py", "one.nrrd", "two.csv"])


def test_validate_cli_arguments_nonexistent_files(tmp_path):
    """Verify FileNotFoundError when input files do not exist."""
    fake_nrrd = str(tmp_path / "does_not_exist.nrrd")
    fake_csv = str(tmp_path / "does_not_exist.csv")
    out_mp4 = str(tmp_path / "out.mp4")

    # Missing NRRD
    with pytest.raises(FileNotFoundError, match="NRRD input file does not exist"):
        validate_cli_arguments(["Ex0.py", fake_nrrd, fake_csv, out_mp4])

    # Existing NRRD, missing CSV
    real_nrrd = tmp_path / "real.nrrd"
    real_nrrd.write_bytes(b"dummy")
    with pytest.raises(FileNotFoundError, match="CSV trajectory input file does not exist"):
        validate_cli_arguments(["Ex0.py", str(real_nrrd), fake_csv, out_mp4])


def test_validate_cli_arguments_valid(tmp_path):
    """Verify valid arguments pass cleanly and create output directory."""
    nrrd_file, csv_file = create_test_dataset(tmp_path)
    out_dir = tmp_path / "sub_directory"
    mp4_file = out_dir / "out.mp4"

    res = validate_cli_arguments(["Ex0.py", nrrd_file, csv_file, str(mp4_file)])
    assert res == (nrrd_file, csv_file, str(mp4_file))
    assert out_dir.exists()


# ============================================================================
# 2. End-to-End Pipeline Execution Tests
# ============================================================================

def test_run_pipeline_end_to_end(tmp_path, capsys):
    """
    Verify complete end-to-end execution:
    Loads volume, parses trajectory, projects planes, and encodes valid MP4.
    """
    nrrd_file, csv_file = create_test_dataset(tmp_path)
    mp4_file = str(tmp_path / "pipeline_output.mp4")

    fps_avg = run_pipeline(nrrd_file, csv_file, mp4_file, fps=30.0, verbose=True)

    assert fps_avg >= 30.0, f"Expected FPS >= 30, got {fps_avg}"
    assert os.path.exists(mp4_file)
    assert os.path.getsize(mp4_file) > 0

    # Verify video container
    info = inspect_video_file(mp4_file)
    assert info["is_valid"] is True
    assert info["frame_count"] == 5
    assert info["width"] > 0
    assert info["height"] > 0

    # Check printed summary banner
    captured = capsys.readouterr()
    assert "Output Video Successfully Generated" in captured.out
    assert "Average Throughput" in captured.out
    assert "PASSED" in captured.out


# ============================================================================
# 3. Subprocess CLI Invocation Tests
# ============================================================================

def test_cli_subprocess_missing_args():
    """Verify invoking python Ex0.py without arguments exits with code 1."""
    cmd = [sys.executable, "Ex0.py"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "Usage: python Ex0.py" in proc.stderr


def test_cli_subprocess_help():
    """Verify invoking python Ex0.py --help exits with code 0."""
    cmd = [sys.executable, "Ex0.py", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Usage: python Ex0.py" in proc.stdout


def test_cli_subprocess_full_execution(tmp_path):
    """Verify full CLI execution via subprocess runs end-to-end with code 0."""
    nrrd_file, csv_file = create_test_dataset(tmp_path)
    mp4_file = str(tmp_path / "cli_subprocess.mp4")

    cmd = [sys.executable, "Ex0.py", nrrd_file, csv_file, mp4_file]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"CLI failed with error: {proc.stderr}"
    assert "Output Video Successfully Generated" in proc.stdout
    assert os.path.exists(mp4_file)

    info = inspect_video_file(mp4_file)
    assert info["is_valid"] is True
    assert info["frame_count"] == 5
