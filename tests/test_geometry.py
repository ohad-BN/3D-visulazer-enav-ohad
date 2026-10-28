"""
Automated Test Suite for Geometric Modeling & CSV Trajectory Parsing (Phase 2)
Covers:
- STP Suite 1: Geometric Validation (Isosceles Triangles & Trapezoids, Degeneracy, Coplanarity)
- STP Suite 7: Visual Continuity & Non-Blocking Step Warning
- CSV Trajectory Parsing (9-point, 12-point, comments, whitespace, malformed rows)
"""

import os
import math
import tempfile
import pytest
import numpy as np

from geometry import (
    Point3D,
    PlaneDefinition,
    validate_triangle,
    validate_trapezoid,
    create_plane,
    parse_csv_line,
    load_trajectory_csv,
    check_trajectory_continuity,
    GeometryValidationError,
    CSVParseError,
    EPSILON,
)


# ============================================================================
# STP Suite 1: 3-Point Validation (Isosceles Triangle)
# ============================================================================

def test_triangle_valid_exact():
    """Verify standard isosceles triangle passes validation."""
    # Apex at (0, 5, 0), base endpoints at (-3, 0, 0) and (3, 0, 0)
    # dist(a, b) = sqrt(3^2 + 5^2) = sqrt(34)
    # dist(a, c) = sqrt(3^2 + 5^2) = sqrt(34)
    a = Point3D(0.0, 5.0, 0.0)
    b = Point3D(-3.0, 0.0, 0.0)
    c = Point3D(3.0, 0.0, 0.0)
    assert validate_triangle(a, b, c) is True


def test_triangle_valid_within_tolerance():
    """Verify triangle with leg difference within epsilon passes."""
    a = Point3D(0.0, 5.0, 0.0)
    b = Point3D(-3.0, 0.0, 0.0)
    # Slight nudge of 0.0004 (< EPSILON = 0.001)
    c = Point3D(3.0, 0.0004, 0.0)
    assert validate_triangle(a, b, c, eps=0.001) is True


def test_triangle_invalid_scalene():
    """Verify non-isosceles triangle raises GeometryValidationError."""
    a = Point3D(0.0, 5.0, 0.0)
    b = Point3D(-3.0, 0.0, 0.0)  # dist = sqrt(34) ~= 5.8309
    c = Point3D(5.0, 0.0, 0.0)   # dist = sqrt(50) ~= 7.0710
    with pytest.raises(GeometryValidationError, match="not isosceles"):
        validate_triangle(a, b, c)


def test_triangle_degenerate_coincident_points():
    """Verify degenerate triangle (coincident points) is rejected."""
    # a and b are identical
    a = Point3D(1.0, 2.0, 3.0)
    b = Point3D(1.0, 2.0, 3.0)
    c = Point3D(4.0, 5.0, 6.0)
    with pytest.raises(GeometryValidationError, match="Degenerate triangle"):
        validate_triangle(a, b, c)

    # b and c are identical (zero base length)
    b2 = Point3D(4.0, 0.0, 0.0)
    c2 = Point3D(4.0, 0.0, 0.0)
    with pytest.raises(GeometryValidationError, match="base length must be positive"):
        validate_triangle(a, b2, c2)


# ============================================================================
# STP Suite 1: 4-Point Validation (Isosceles Trapezoid)
# ============================================================================

def test_trapezoid_valid_symmetric():
    """Verify a standard symmetric isosceles trapezoid passes validation."""
    # Top base: (1, 4, 0) to (3, 4, 0) -> length = 2.0
    # Bottom base: (0, 0, 0) to (4, 0, 0) -> length = 4.0
    # Left leg: dist((1,4,0), (0,0,0)) = sqrt(1 + 16) = sqrt(17)
    # Right leg: dist((3,4,0), (4,0,0)) = sqrt((3-4)^2 + 16) = sqrt(17)
    a1 = Point3D(1.0, 4.0, 0.0)
    a2 = Point3D(3.0, 4.0, 0.0)
    b = Point3D(0.0, 0.0, 0.0)
    c = Point3D(4.0, 0.0, 0.0)
    assert validate_trapezoid(a1, a2, b, c) is True


def test_trapezoid_valid_rectangle():
    """Verify a rectangle (top_width == base_width) passes validation."""
    a1 = Point3D(0.0, 4.0, 0.0)
    a2 = Point3D(5.0, 4.0, 0.0)
    b = Point3D(0.0, 0.0, 0.0)
    c = Point3D(5.0, 0.0, 0.0)
    assert validate_trapezoid(a1, a2, b, c) is True


def test_trapezoid_valid_3d_oblique():
    """Verify an oblique 3D rotated trapezoid in arbitrary 3D orientation passes."""
    # Construct by applying rigid rotation and translation to a known trapezoid
    # Normal to plane: [0, 1, 1] / sqrt(2)
    # Points in plane y = z:
    a1 = Point3D(1.0, 4.0, 4.0)
    a2 = Point3D(3.0, 4.0, 4.0)
    b = Point3D(0.0, 0.0, 0.0)
    c = Point3D(4.0, 0.0, 0.0)
    assert validate_trapezoid(a1, a2, b, c) is True


def test_trapezoid_invalid_unequal_legs():
    """Verify trapezoid with unequal legs is rejected."""
    a1 = Point3D(1.0, 4.0, 0.0)
    a2 = Point3D(3.0, 4.0, 0.0)
    b = Point3D(0.0, 0.0, 0.0)   # leg1 = sqrt(17)
    c = Point3D(5.0, 0.0, 0.0)   # leg2 = sqrt((3-5)^2 + 16) = sqrt(20) != sqrt(17)
    with pytest.raises(GeometryValidationError, match="legs not equal"):
        validate_trapezoid(a1, a2, b, c)


def test_trapezoid_invalid_inverted_bases():
    """Verify trapezoid where top base > bottom base is rejected."""
    # Top base = 6.0, Bottom base = 2.0
    a1 = Point3D(0.0, 4.0, 0.0)
    a2 = Point3D(6.0, 4.0, 0.0)
    b = Point3D(2.0, 0.0, 0.0)
    c = Point3D(4.0, 0.0, 0.0)
    # Leg 1: dist((0,4,0),(2,0,0)) = sqrt(4+16) = sqrt(20)
    # Leg 2: dist((6,4,0),(4,0,0)) = sqrt(4+16) = sqrt(20)
    # Equal legs, coplanar, but top base (6) > bottom base (2)
    with pytest.raises(GeometryValidationError, match="top base .* strictly larger than bottom base"):
        validate_trapezoid(a1, a2, b, c)


def test_trapezoid_invalid_non_coplanar():
    """Verify trapezoid with equal legs and ordered bases but non-coplanar 4th point is rejected."""
    # b=(0,0,0) and c=(10,0,0) -> bottom base = 10.0
    # a1=(3, 4, 0) -> dist(a1, b) = sqrt(9+16) = 5.0
    # a2=(7, 3, sqrt(7)) -> dist(a2, c) = sqrt((7-10)^2 + 3^2 + 7) = sqrt(9+9+7) = 5.0
    # Legs are strictly equal (5.0 == 5.0)
    # Top base: dist(a1, a2) = sqrt(16 + 1 + 7) = sqrt(24) ~= 4.899 <= 10.0 (ordered)
    # But a1, b, c lie in z=0 while a2 has z=sqrt(7) ~= 2.6458 (non-coplanar!)
    sqrt7 = math.sqrt(7.0)
    a1 = Point3D(3.0, 4.0, 0.0)
    a2 = Point3D(7.0, 3.0, sqrt7)
    b = Point3D(0.0, 0.0, 0.0)
    c = Point3D(10.0, 0.0, 0.0)
    with pytest.raises(GeometryValidationError, match="not coplanar"):
        validate_trapezoid(a1, a2, b, c)


def test_trapezoid_invalid_collinear():
    """Verify collinear baseline points are rejected."""
    a1 = Point3D(0.0, 0.0, 0.0)
    a2 = Point3D(1.0, 0.0, 0.0)
    b = Point3D(2.0, 0.0, 0.0)
    c = Point3D(3.0, 0.0, 0.0)
    with pytest.raises(GeometryValidationError, match="collinear|legs not equal|Degenerate"):
        validate_trapezoid(a1, a2, b, c)


# ============================================================================
# CSV Trajectory Parsing (Phase 2.3)
# ============================================================================

def test_create_plane_triangle_and_trapezoid():
    """Verify create_plane factory correctly builds both types."""
    # 9 numbers -> Triangle
    tri_pts = [0.0, 5.0, 0.0, -3.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    p_tri = create_plane(tri_pts)
    assert p_tri.is_triangle is True
    assert p_tri.a1 == p_tri.a2
    assert p_tri.top_width == 0.0
    assert p_tri.leg_length == pytest.approx(math.sqrt(34), abs=1e-5)
    assert p_tri.base_width == pytest.approx(6.0, abs=1e-5)

    # 12 numbers -> Trapezoid
    trap_pts = [1.0, 4.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    p_trap = create_plane(trap_pts)
    assert p_trap.is_triangle is False
    assert p_trap.top_width == pytest.approx(2.0, abs=1e-5)
    assert p_trap.base_width == pytest.approx(4.0, abs=1e-5)
    assert p_trap.leg_length == pytest.approx(math.sqrt(17), abs=1e-5)


def test_create_plane_invalid_count():
    """Verify invalid coordinate counts raise CSVParseError."""
    with pytest.raises(CSVParseError, match="expected 9 or 12"):
        create_plane([1.0, 2.0, 3.0])
    with pytest.raises(CSVParseError, match="expected 9 or 12"):
        create_plane([1.0] * 10)


def test_parse_csv_line():
    """Verify parsing single CSV lines with spaces, comments, and delimiters."""
    # Comment line
    assert parse_csv_line("# This is a comment") is None
    # Empty line
    assert parse_csv_line("   \n") is None

    # Comma-delimited 9-number row
    line9 = " 0.0, 5.0, 0.0, -3.0, 0.0, 0.0, 3.0, 0.0, 0.0 \n"
    p9 = parse_csv_line(line9)
    assert p9 is not None and p9.is_triangle is True

    # Whitespace-delimited 12-number row
    line12 = "1.0 4.0 0.0  3.0 4.0 0.0  0.0 0.0 0.0  4.0 0.0 0.0"
    p12 = parse_csv_line(line12)
    assert p12 is not None and p12.is_triangle is False

    # Malformed row with non-numbers
    with pytest.raises(CSVParseError, match="non-numeric"):
        parse_csv_line("1.0, 2.0, invalid, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0")


def test_load_trajectory_csv_file(tmp_path):
    """Verify loading full CSV trajectory file containing mixed rows and comments."""
    csv_content = """# Sample Trajectory File
# Line 1: Triangle
0.0, 5.0, 0.0, -3.0, 0.0, 0.0, 3.0, 0.0, 0.0

# Line 2: Trapezoid
1.0, 4.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0

# Line 3: Rectangle
0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0
"""
    file_path = tmp_path / "trajectory.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    planes = load_trajectory_csv(str(file_path))
    assert len(planes) == 3
    assert planes[0].is_triangle is True
    assert planes[1].is_triangle is False
    assert planes[2].is_triangle is False


def test_load_trajectory_csv_file_not_found():
    """Verify FileNotFoundError when trajectory CSV path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_trajectory_csv("non_existent_file.csv")


def test_load_trajectory_csv_skip_invalid(tmp_path, capsys):
    """Verify that malformed lines are skipped non-destructively when skip_invalid=True."""
    csv_content = """# Trajectory with one bad row
0.0, 5.0, 0.0, -3.0, 0.0, 0.0, 3.0, 0.0, 0.0
bad, line, with, text, that, cannot, be, parsed, numbers
1.0, 4.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0
"""
    file_path = tmp_path / "partially_invalid.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    planes = load_trajectory_csv(str(file_path), skip_invalid=True)
    assert len(planes) == 2


# ============================================================================
# STP Suite 7: Visual Continuity & Non-Blocking Discontinuity Warning
# ============================================================================

def test_continuity_smooth_trajectory():
    """Verify smooth trajectory triggers no continuity warnings."""
    # Volume size 500 -> threshold = 0.01 * 500 = 5.0
    max_dim = 500.0

    # Frame 0
    p0 = PlaneDefinition(
        a1=Point3D(0.0, 0.0, 0.0),
        a2=Point3D(0.0, 0.0, 0.0),
        b=Point3D(0.0, 5.0, 0.0),
        c=Point3D(5.0, 5.0, 0.0),
        is_triangle=True,
    )
    # Frame 1: shifted by 1.0 (< 5.0)
    p1 = PlaneDefinition(
        a1=Point3D(0.0, 0.0, 1.0),
        a2=Point3D(0.0, 0.0, 1.0),
        b=Point3D(0.0, 5.0, 1.0),
        c=Point3D(5.0, 5.0, 1.0),
        is_triangle=True,
    )

    warnings = check_trajectory_continuity([p0, p1], max_dim)
    assert len(warnings) == 0


def test_continuity_discontinuous_jump_warning():
    """Verify large spatial jump triggers informative non-blocking warning."""
    # Volume size 100 -> threshold = 0.01 * 100 = 1.0
    max_dim = 100.0

    p0 = PlaneDefinition(
        a1=Point3D(0.0, 0.0, 0.0),
        a2=Point3D(0.0, 0.0, 0.0),
        b=Point3D(0.0, 5.0, 0.0),
        c=Point3D(5.0, 5.0, 0.0),
        is_triangle=True,
    )
    # Frame 1: jump of 50.0 in z (>> 1.0)
    p1 = PlaneDefinition(
        a1=Point3D(0.0, 0.0, 50.0),
        a2=Point3D(0.0, 0.0, 50.0),
        b=Point3D(0.0, 5.0, 50.0),
        c=Point3D(5.0, 5.0, 50.0),
        is_triangle=True,
    )

    warnings = check_trajectory_continuity([p0, p1], max_dim)
    assert len(warnings) == 1
    assert "[WARN]" in warnings[0]
    assert "Large spatial delta" in warnings[0]
    assert "between frames 0 and 1" in warnings[0]
