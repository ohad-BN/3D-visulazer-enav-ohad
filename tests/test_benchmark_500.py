"""
STP Suite 4: Performance & High-FPS Benchmarking on 500^3 Matrix
Verifies:
- Memory allocation and safe handling of 500x500x500 volumetric matrix (125 MB)
- Slicing and projection throughput over 100 consecutive frames
- NFR-01: Strict minimum frame rate >= 30 FPS
- NFR-02: Target average throughput >= 60 FPS (target >= 100 FPS on byte matrix)
"""

import time
import pytest
import numpy as np

from geometry import Point3D, PlaneDefinition
from nrrd_loader import NRRDVolume
from projection_engine import project_plane_vectorized


def test_stp_suite4_500cube_matrix_benchmark():
    """
    STP Suite 4 Benchmark:
    1. Allocate 500x500x500 uint8 volume M (125 MB).
    2. Define trajectory of 100 consecutive slicing frames (250x250 pixels each = 62,500 pixels/frame).
    3. Measure pure compute time and verify >= 60 FPS target (and >= 100 FPS on byte matrix).
    """
    sx, sy, sz = 500, 500, 500
    # Allocate synthetic 500^3 uint8 array
    data = np.zeros((sx, sy, sz), dtype=np.uint8)
    # Add some non-zero content
    data[100:400, 100:400, 100:400] = 128

    vol = NRRDVolume(data=data, header={}, filepath="benchmark_500cube.nrrd")

    # Define 100 frames sweeping through the volume
    num_frames = 100
    planes = []
    for i in range(num_frames):
        z_pos = 100.0 + (i * 2.5)  # Sweeps from z=100 to z=347.5
        # Large plane: 250 x 250 pixels
        p = PlaneDefinition(
            a1=Point3D(125.0, 375.0, z_pos),
            a2=Point3D(375.0, 375.0, z_pos),
            b=Point3D(125.0, 125.0, z_pos),
            c=Point3D(375.0, 125.0, z_pos),
            is_triangle=False,
        )
        planes.append(p)

    # Warm-up pass (ensures CPU cache and memory allocations are initialized)
    _ = project_plane_vectorized(vol, planes[0])

    # Benchmark loop
    t_start = time.perf_counter()
    for p in planes:
        frame = project_plane_vectorized(vol, p)
    t_compute = time.perf_counter() - t_start

    fps_avg = num_frames / t_compute if t_compute > 0 else 0.0
    print(f"\n[STP SUITE 4 - 500^3 BENCHMARK] Processed {num_frames} frames (250x250) in {t_compute:.4f}s: {fps_avg:.1f} FPS")

    # Verification of criteria
    assert fps_avg >= 30.0, f"Failed NFR-01: FPS {fps_avg:.1f} < 30 FPS"
    assert fps_avg >= 60.0, f"Failed NFR-02: FPS {fps_avg:.1f} < 60 FPS (target)"
    assert fps_avg >= 100.0, f"Failed NFR-02 500^3 byte matrix target: FPS {fps_avg:.1f} < 100 FPS"
