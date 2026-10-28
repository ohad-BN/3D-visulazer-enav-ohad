"""
Ex0.py - Main CLI Entry Point for 3D NRRD Plane Projection and Video Generation

Usage:
    python Ex0.py <file.nrrd> <file.csv> <file.mp4>

Specification & Traceability:
    - REQ-01: Ingests 3D raster data matrix M (up to 1024^3).
    - REQ-02: Parses trajectory plane definitions from CSV.
    - REQ-03: Validates geometry with epsilon = 0.001.
    - REQ-04: Projects each valid plane into 2D matrix E.
    - REQ-05: Safely clamps out-of-bounds samples to 0.
    - REQ-06: Encodes frames sequentially into MP4 video.
    - NFR-01/02: High-resolution profiler measuring execution time and FPS (target >= 60 FPS).
"""

from __future__ import annotations
import sys
import os
import time
import math
from typing import List, Tuple

from geometry import (
    Point3D,
    PlaneDefinition,
    load_trajectory_csv,
    check_trajectory_continuity,
    GeometryValidationError,
    CSVParseError,
)
from nrrd_loader import load_nrrd, NRRDLoadError, NRRDFormatError, NRRDVolume
from projection_engine import project_plane
from video_writer import MP4VideoWriter, VideoWriterError, inspect_video_file


def resolve_dataset_file(given_path: str, default_folder: str, default_ext: str) -> str:
    """
    Resolves input file path, checking:
    1. Direct path as provided (relative or absolute).
    2. Path inside default_folder (e.g. 'NRRD' or 'CSV').
    3. Path with extension appended (e.g. '.nrrd' or '.csv').
    4. Path inside default_folder with extension appended.
    5. Case-insensitive match inside default_folder.
    """
    candidates = [
        given_path,
        os.path.join(default_folder, given_path),
    ]
    if not given_path.lower().endswith(default_ext.lower()):
        candidates.append(given_path + default_ext)
        candidates.append(os.path.join(default_folder, given_path + default_ext))

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    # Case-insensitive search within default_folder if it exists
    if os.path.isdir(default_folder):
        target_name = os.path.basename(given_path).lower()
        target_with_ext = (
            target_name if target_name.endswith(default_ext.lower()) else target_name + default_ext.lower()
        )
        for fname in os.listdir(default_folder):
            if fname.lower() in (target_name, target_with_ext):
                full_path = os.path.join(default_folder, fname)
                if os.path.isfile(full_path):
                    return full_path

    return given_path


def validate_cli_arguments(args: List[str]) -> Tuple[str, str, str]:
    """
    Validates CLI argument count and file syntax.
    Automatically resolves files within 'NRRD/' and 'CSV/' folders.
    Returns:
        Tuple of (nrrd_path, csv_path, mp4_path).
    Raises:
        ValueError: If arguments are missing or invalid.
    """
    if len(args) == 2 and args[1] in ("-h", "--help"):
        print("Usage: python Ex0.py <file.nrrd> <file.csv> <file.mp4>")
        print("\nArguments:")
        print("  <file.nrrd>  Path to input 3D NRRD volumetric dataset (or name in NRRD/ folder)")
        print("  <file.csv>   Path to input trajectory CSV (or name in CSV/ folder)")
        print("  <file.mp4>   Path to output MP4 video file")
        sys.exit(0)

    if len(args) != 4:
        raise ValueError(
            f"Expected exactly 3 arguments (<file.nrrd> <file.csv> <file.mp4>), got {len(args) - 1}."
        )

    nrrd_path = resolve_dataset_file(args[1], "NRRD", ".nrrd")
    csv_path = resolve_dataset_file(args[2], "CSV", ".csv")
    mp4_path = args[3]

    if not os.path.exists(nrrd_path):
        raise FileNotFoundError(f"NRRD input file does not exist: '{args[1]}' (also searched in 'NRRD/' directory)")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV trajectory input file does not exist: '{args[2]}' (also searched in 'CSV/' directory)")

    # Ensure parent output directory exists or can be created
    out_dir = os.path.dirname(os.path.abspath(mp4_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    return nrrd_path, csv_path, mp4_path


def run_pipeline(
    nrrd_path: str,
    csv_path: str,
    mp4_path: str,
    fps: float = 30.0,
    verbose: bool = True,
) -> float:
    """
    Runs the complete end-to-end projection and video encoding pipeline.

    Args:
        nrrd_path: Path to input .nrrd volume.
        csv_path: Path to input .csv trajectory.
        mp4_path: Path to destination .mp4 video.
        fps: Playback frame rate for output video (default: 30.0).
        verbose: If True, prints progress and benchmarking stats.

    Returns:
        Average FPS achieved during projection and video encoding.
    """
    if verbose:
        print("=" * 70)
        print("3D NRRD Volumetric Plane Projection & Video Generator (Ex0)")
        print("=" * 70)
        print(f"[*] Ingesting NRRD volume from: {nrrd_path}")

    volume: NRRDVolume = load_nrrd(nrrd_path)
    if verbose:
        print(f"[*] Volume loaded: shape {volume.shape}, dtype {volume.dtype}")
        print(f"[*] Ingesting trajectory CSV from: {csv_path}")

    planes: List[PlaneDefinition] = load_trajectory_csv(csv_path, skip_invalid=True)
    if verbose:
        print(f"[*] Parsed {len(planes)} valid plane definitions.")

    # Visual continuity & discontinuity warning check (STP Suite 7)
    warnings = check_trajectory_continuity(planes, float(volume.max_dimension))
    for w in warnings:
        print(w, file=sys.stderr)

    # Standardize canvas dimension from maximal dimensions in the trajectory
    max_h = max(int(math.ceil(p.leg_length)) for p in planes)
    max_w = max(int(math.ceil(p.base_width)) for p in planes)
    canvas_size = (max_w, max_h)

    if verbose:
        print(f"[*] Standardized canvas resolution: {canvas_size[0]}x{canvas_size[1]}")
        print(f"[*] Encoding destination: {mp4_path} at {fps:.1f} FPS")
        print("[*] Starting projection and video encoding pipeline...")

    # High-resolution performance profiler (STP Suite 4, NFR-01/02)
    start_time = time.perf_counter()

    vol_max = float(volume.data.max()) if volume.data.size > 0 else None
    with MP4VideoWriter(
        mp4_path, fps=fps, target_size=canvas_size, global_max=vol_max
    ) as writer:
        for plane in planes:
            frame = project_plane(volume, plane, vectorized=True)
            writer.add_frame(frame)

    elapsed_time = time.perf_counter() - start_time
    num_frames = len(planes)
    fps_avg = num_frames / elapsed_time if elapsed_time > 0 else 0.0

    if verbose:
        print("=" * 70)
        print(f"[+] Output Video Successfully Generated: {mp4_path}")
        print(f"[+] Total Frames Rendered: {num_frames}")
        print(f"[+] Processing Time:        {elapsed_time:.3f} seconds")
        print(f"[+] Average Throughput:     {fps_avg:.1f} FPS")
        status = "PASSED (>= 60 FPS Target)" if fps_avg >= 60.0 else (
            "PASSED (>= 30 FPS Minimum)" if fps_avg >= 30.0 else "FAILED (< 30 FPS)"
        )
        print(f"[+] Performance Status:     {status}")
        print("=" * 70)

    return fps_avg


def main() -> None:
    try:
        nrrd_path, csv_path, mp4_path = validate_cli_arguments(sys.argv)
    except ValueError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        print("Usage: python Ex0.py <file.nrrd> <file.csv> <file.mp4>", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(nrrd_path, csv_path, mp4_path, verbose=True)
        sys.exit(0)
    except Exception as err:
        print(f"[ERROR] Pipeline execution failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
