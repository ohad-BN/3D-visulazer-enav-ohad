"""
STP Edge Case Test Suite 1: Geometric & Sampling Edge Cases
Covers:
- Triangle Apex Singularity: verifies sampling at y=0 where p1=p2=a and d12=0 without NaN or ZeroDivisionError.
- Planes Skimming Matrix Boundaries: verifies safe handling of boundary planes at x=0, x=sx-1, and fractional edge grazes.
- True 3D 45-Degree Oblique Planes: verifies simultaneous XYZ diagonal projection equivalence between baseline and vectorized.
"""

import math
import pytest
import numpy as np

from geometry import Point3D, PlaneDefinition, validate_trapezoid, validate_triangle
from nrrd_loader import NRRDVolume
from projection_engine import (
    project_plane_baseline,
    project_plane_vectorized,
    project_plane,
)


def test_triangle_apex_singularity_sampling():
    """
    Test sampling at the apex of an isosceles triangle (d12 = 0).
    At y = 0:
      p1 = a, p2 = a -> horizontal distance d12 = ||p2 - p1|| = 0.
      The unit step vector dp_hat would be (0, 0, 0) / 0 if not guarded.
      The apex line must sample exactly 1 pixel at column x_offset.
    """
    # 50x50x50 volume of zeros with a specific sentinel value at the apex voxel
    vol_data = np.zeros((50, 50, 50), dtype=np.uint16)
    apex_coord = (25, 30, 15)
    vol_data[apex_coord] = 777
    vol = NRRDVolume(data=vol_data, header={}, filepath="singularity_test.nrrd")

    # Isosceles triangle:
    # apex a = (25, 30, 15)
    # base b = (10, 10, 15), c = (40, 10, 15)
    # dist(a, b) = sqrt(15^2 + 20^2) = 25
    # dist(a, c) = sqrt(15^2 + 20^2) = 25
    # dist(b, c) = 30
    a = Point3D(25.0, 30.0, 15.0)
    b = Point3D(10.0, 10.0, 15.0)
    c = Point3D(40.0, 10.0, 15.0)

    validate_triangle(a, b, c)
    plane = PlaneDefinition(a1=a, a2=a, b=b, c=c, is_triangle=True)

    # 1. Execute both engines without ZeroDivisionError or RuntimeWarning
    frame_base = project_plane_baseline(vol, plane)
    frame_vec = project_plane_vectorized(vol, plane)

    # 2. Both engines must produce identical results
    np.testing.assert_array_equal(frame_base, frame_vec)

    # Dimensions: H = ceil(25) = 25, W = ceil(30) = 30
    assert frame_vec.shape == (25, 30)

    # At row y = 0:
    # d12 = 0, x_offset = floor((30 - 0) / 2) = 15.
    # Apex voxel value 777 must be present at frame[0, 15]
    assert frame_vec[0, 15] == 777, (
        f"Expected apex voxel 777 at frame[0, 15], got {frame_vec[0, 15]}"
    )

    # All other columns in row 0 must be 0 (no spurious pixels outside the apex)
    for x in range(30):
        if x != 15:
            assert frame_vec[0, x] == 0, f"Expected 0 at frame[0, {x}], got {frame_vec[0, x]}"


def test_plane_skimming_matrix_boundary():
    """
    Test planes positioned right on the outer boundary faces of the volume.
    Verifies that float rounding doesn't drop valid edge voxels or cause out-of-bounds indexing.
    """
    sx, sy, sz = 40, 40, 40
    vol_data = np.zeros((sx, sy, sz), dtype=np.uint8)

    # Distinguish boundary faces with known sentinel values
    vol_data[0, :, :] = 101       # Face x = 0
    vol_data[sx - 1, :, :] = 202  # Face x = 39
    vol = NRRDVolume(data=vol_data, header={}, filepath="skimming_test.nrrd")

    # Case A: Plane lying exactly on face x = 0
    p_x0 = PlaneDefinition(
        a1=Point3D(0.0, 30.0, 30.0),
        a2=Point3D(0.0, 30.0, 10.0),
        b=Point3D(0.0, 10.0, 30.0),
        c=Point3D(0.0, 10.0, 10.0),
        is_triangle=False,
    )
    frame_x0_base = project_plane_baseline(vol, p_x0)
    frame_x0_vec = project_plane_vectorized(vol, p_x0)

    np.testing.assert_array_equal(frame_x0_base, frame_x0_vec)
    # Every sampled pixel on face x=0 must equal 101, zero boundary pixels dropped
    assert np.all(frame_x0_vec == 101)

    # Case B: Plane slightly grazing face x = 39 at x = 39.99
    # floor(39.99) = 39 -> within bounds [0, 40) -> samples face x = 39
    p_x39 = PlaneDefinition(
        a1=Point3D(39.99, 30.0, 30.0),
        a2=Point3D(39.99, 30.0, 10.0),
        b=Point3D(39.99, 10.0, 30.0),
        c=Point3D(39.99, 10.0, 10.0),
        is_triangle=False,
    )
    frame_x39 = project_plane_vectorized(vol, p_x39)
    assert np.all(frame_x39 == 202)

    # Case C: Plane slightly grazing outside at x = -0.001
    # floor(-0.001) = -1 -> safely detected as out of bounds -> clamped to 0
    p_neg = PlaneDefinition(
        a1=Point3D(-0.001, 30.0, 30.0),
        a2=Point3D(-0.001, 30.0, 10.0),
        b=Point3D(-0.001, 10.0, 30.0),
        c=Point3D(-0.001, 10.0, 10.0),
        is_triangle=False,
    )
    frame_neg = project_plane_vectorized(vol, p_neg)
    assert np.all(frame_neg == 0)


def test_arbitrary_45deg_oblique_diagonal_plane():
    """
    Test true 3D diagonal plane inclined at 45 degrees across all axes simultaneously.
    Verifies that multi-axis step vectors compute accurately and baseline/vectorized match 100%.
    """
    # 60x60x60 volume with linear coordinate sum: M[x, y, z] = x + y + z
    x_idx, y_idx, z_idx = np.indices((60, 60, 60), dtype=np.int32)
    vol_data = (x_idx + y_idx + z_idx).astype(np.uint16)
    vol = NRRDVolume(data=vol_data, header={}, filepath="oblique_45deg_test.nrrd")

    # Oblique plane:
    # a1 = (10, 10, 40), a2 = (30, 30, 40)
    # b  = (10, 10, 10), c  = (30, 30, 10)
    # Leg dist(a1, b) = 30.0 (along z)
    # Base dist(b, c) = sqrt(20^2 + 20^2) = sqrt(800) ~= 28.284 (along xy diagonal 45 deg)
    a1 = Point3D(10.0, 10.0, 40.0)
    a2 = Point3D(30.0, 30.0, 40.0)
    b = Point3D(10.0, 10.0, 10.0)
    c = Point3D(30.0, 30.0, 10.0)

    validate_trapezoid(a1, a2, b, c)
    plane = PlaneDefinition(a1=a1, a2=a2, b=b, c=c, is_triangle=False)

    frame_base = project_plane_baseline(vol, plane)
    frame_vec = project_plane_vectorized(vol, plane)

    # 1. Baseline and vectorized equivalence
    np.testing.assert_array_equal(frame_base, frame_vec)

    # 2. Dimensions check
    assert frame_vec.shape == (30, 29)

    # 3. Value check: verify values increase along both rows (z decreasing) and cols (x,y increasing)
    # Top-left corner: y=0, x=0 -> near (10, 10, 40) -> value ~ 60
    assert frame_vec[0, 0] == pytest.approx(60, abs=1)
    # Bottom-right corner: y=29, x=28 -> near (30, 30, 11) -> value ~ 71
    assert frame_vec[29, 28] == pytest.approx(70, abs=2)
