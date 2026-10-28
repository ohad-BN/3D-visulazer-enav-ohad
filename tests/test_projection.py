"""
Automated Test Suite for 2D Plane Projection Engine (Phase 4)
Covers:
- STP Suite 2: Mathematical Sanity & Reference Comparison
- STP Suite 5: Boundary & Out-of-Bounds Sampling Safety
- Equivalence between baseline and vectorized projection engines
- High-throughput performance benchmark (STP Suite 4, NFR-01/02 >= 60 FPS)
"""

import math
import time
import pytest
import numpy as np

from geometry import Point3D, PlaneDefinition
from nrrd_loader import NRRDVolume, load_nrrd
from projection_engine import (
    project_plane_baseline,
    project_plane_vectorized,
    project_plane,
)


def create_synthetic_gradient_volume(shape=(50, 50, 50)) -> NRRDVolume:
    """
    Creates a synthetic 3D volume where M[x, y, z] = x.
    This allows exact analytical prediction of projected values.
    """
    sx, sy, sz = shape
    # data[x, y, z] = x
    x_coords = np.arange(sx, dtype=np.uint16)
    data = np.broadcast_to(x_coords[:, None, None], (sx, sy, sz)).copy()
    return NRRDVolume(data=data, header={}, filepath="synthetic_gradient.nrrd")


# ============================================================================
# STP Suite 2: Mathematical Sanity & Reference Comparison
# ============================================================================

def test_axial_plane_analytical_ground_truth():
    """
    Verify projection on an axial plane z=k against analytical ground truth.
    Volume: M[x, y, z] = x.
    Rectangle plane:
      a1 = (10, 30, 25), a2 = (40, 30, 25)
      b  = (10, 10, 25), c  = (40, 10, 25)
      Leg length = dist(a1, b) = 20.
      Base width = dist(b, c) = 30.
    At row y and col x:
      x advances along horizontal base: x_sample = 10 + x1
      voxel M[px, py, pz] has value px = 10 + x1
    """
    vol = create_synthetic_gradient_volume((60, 60, 60))

    a1 = Point3D(10.0, 30.0, 25.0)
    a2 = Point3D(40.0, 30.0, 25.0)
    b = Point3D(10.0, 10.0, 25.0)
    c = Point3D(40.0, 10.0, 25.0)

    plane = PlaneDefinition(a1=a1, a2=a2, b=b, c=c, is_triangle=False)

    frame_base = project_plane_baseline(vol, plane)
    frame_vec = project_plane_vectorized(vol, plane)

    # 1. Verify equivalence between baseline and vectorized
    np.testing.assert_array_equal(frame_base, frame_vec)

    # 2. Verify dimensions H = 20, W = 30
    assert frame_vec.shape == (20, 30)

    # 3. Verify analytical values: each column x should equal 10 + x
    for y in range(20):
        for x in range(30):
            expected_val = 10 + x
            assert frame_vec[y, x] == expected_val, (
                f"Mismatch at (y={y}, x={x}): got {frame_vec[y, x]}, expected {expected_val}"
            )


def test_triangle_plane_equivalence():
    """Verify isosceles triangle plane produces identical results on both engines."""
    vol = create_synthetic_gradient_volume((60, 60, 60))

    # Triangle with apex at (25, 40, 20), base from (10, 10, 20) to (40, 10, 20)
    # dist(a, b) = sqrt(15^2 + 30^2) = sqrt(1125) ~= 33.541
    # dist(a, c) = sqrt(15^2 + 30^2) = sqrt(1125) ~= 33.541
    a = Point3D(25.0, 40.0, 20.0)
    b = Point3D(10.0, 10.0, 20.0)
    c = Point3D(40.0, 10.0, 20.0)

    plane = PlaneDefinition(a1=a, a2=a, b=b, c=c, is_triangle=True)

    frame_base = project_plane_baseline(vol, plane)
    frame_vec = project_plane_vectorized(vol, plane)

    assert frame_base.shape == frame_vec.shape
    np.testing.assert_array_equal(frame_base, frame_vec)


def test_oblique_trapezoid_equivalence():
    """Verify oblique 3D trapezoid produces identical results on baseline and vectorized."""
    vol = create_synthetic_gradient_volume((60, 60, 60))

    # Plane tilted in 3D: z varies with y
    a1 = Point3D(15.0, 35.0, 35.0)
    a2 = Point3D(35.0, 35.0, 35.0)
    b = Point3D(5.0, 10.0, 10.0)
    c = Point3D(45.0, 10.0, 10.0)

    plane = PlaneDefinition(a1=a1, a2=a2, b=b, c=c, is_triangle=False)

    frame_base = project_plane_baseline(vol, plane)
    frame_vec = project_plane_vectorized(vol, plane)

    np.testing.assert_array_equal(frame_base, frame_vec)


# ============================================================================
# STP Suite 5: Boundary & Out-of-Bounds Sampling Safety
# ============================================================================

def test_partial_out_of_bounds_clamping():
    """
    Verify pixels outside volume bounds are set strictly to 0 (REQ-05),
    while internal pixels retain correct values.
    """
    vol = create_synthetic_gradient_volume((50, 50, 50))

    # Plane starting at negative coordinates and extending into volume
    # b = (-10, 0, 25), c = (30, 0, 25) -> span 40 in x
    # a1 = (-10, 20, 25), a2 = (30, 20, 25)
    a1 = Point3D(-10.0, 20.0, 25.0)
    a2 = Point3D(30.0, 20.0, 25.0)
    b = Point3D(-10.0, 0.0, 25.0)
    c = Point3D(30.0, 0.0, 25.0)

    plane = PlaneDefinition(a1=a1, a2=a2, b=b, c=c, is_triangle=False)

    frame = project_plane(vol, plane, vectorized=True)

    # For x < 10 in frame, target px = -10 + x < 0 -> must be 0
    # For x >= 10 in frame, target px = -10 + x >= 0 -> must be px
    for y in range(frame.shape[0]):
        for x in range(10):
            assert frame[y, x] == 0, f"Expected 0 for out-of-bounds pixel at ({y}, {x})"
        for x in range(10, 30):
            expected = -10 + x
            assert frame[y, x] == expected, f"Expected {expected} for in-bounds pixel at ({y}, {x})"


def test_fully_exterior_plane_black_frame():
    """Verify plane completely exterior to volume produces an entirely black (zero) frame."""
    vol = create_synthetic_gradient_volume((50, 50, 50))

    # Plane located far beyond volume boundaries (z = 500)
    a1 = Point3D(10.0, 30.0, 500.0)
    a2 = Point3D(40.0, 30.0, 500.0)
    b = Point3D(10.0, 10.0, 500.0)
    c = Point3D(40.0, 10.0, 500.0)

    plane = PlaneDefinition(a1=a1, a2=a2, b=b, c=c, is_triangle=False)

    frame = project_plane(vol, plane, vectorized=True)

    # All pixels must be 0
    assert np.all(frame == 0)


# ============================================================================
# STP Suite 4: Performance & High-FPS Benchmarking (NFR-01/02)
# ============================================================================

def test_high_fps_throughput_benchmark():
    """
    Benchmark throughput over 100 consecutive frames on a 256x256x130 volume (MRHead_1).
    Assert average FPS >= 60 FPS (NFR-02) and strict minimum >= 30 FPS (NFR-01).
    """
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nrrd_path = os.path.join(repo_root, "MRHead_1.nrrd")
    vol = load_nrrd(nrrd_path)

    # Define a representative diagonal slicing plane (~150x150 pixels)
    num_frames = 100
    planes = []
    for i in range(num_frames):
        z_offset = 20.0 + (i * 0.8)
        p = PlaneDefinition(
            a1=Point3D(40.0, 180.0, z_offset),
            a2=Point3D(180.0, 180.0, z_offset),
            b=Point3D(40.0, 40.0, z_offset),
            c=Point3D(180.0, 40.0, z_offset),
            is_triangle=False,
        )
        planes.append(p)

    # Warm-up single frame
    _ = project_plane(vol, planes[0], vectorized=True)

    # Benchmark 100 frames
    t_start = time.perf_counter()
    for p in planes:
        _ = project_plane(vol, p, vectorized=True)
    t_total = time.perf_counter() - t_start

    fps_avg = num_frames / t_total
    print(f"\n[BENCHMARK] Processed {num_frames} frames in {t_total:.4f}s: {fps_avg:.1f} FPS")

    # Assert strict compliance with NFR-01 (>= 30 FPS) and NFR-02 (>= 60 FPS)
    assert fps_avg >= 30.0, f"Failed NFR-01: FPS was {fps_avg:.1f} < 30.0"
    assert fps_avg >= 60.0, f"Failed NFR-02: Target was >= 60 FPS, achieved {fps_avg:.1f} FPS"
