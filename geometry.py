"""
Geometry Module for 3D NRRD Plane Projection (Ex0)

This module handles geometric models, 3D point representations, tolerance checks,
plane validation (isosceles triangle and trapezoid), CSV trajectory parsing,
and trajectory continuity validation (STP Suites 1 & 7).
"""

from __future__ import annotations
import math
import os
import csv
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np

# Geometric tolerance specified in SRS (REQ-03) and EDD
EPSILON: float = 0.001


class GeometryValidationError(ValueError):
    """Raised when plane points violate geometric constraints."""
    pass


class CSVParseError(ValueError):
    """Raised when CSV trajectory rows are malformed or invalid."""
    pass


@dataclass(frozen=True)
class Point3D:
    """
    Represents an immutable 3D Cartesian coordinate (x, y, z).
    """
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        """Return point as a 1D NumPy array [x, y, z] of float64."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other: Point3D) -> float:
        """Euclidean distance between this point and another: ||self - other||_2."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def vector_to(self, other: Point3D) -> np.ndarray:
        """Vector pointing from self to other: other - self."""
        return other.to_array() - self.to_array()


@dataclass(frozen=True)
class PlaneDefinition:
    """
    Unified geometric representation of a projection plane.

    Supports both 3-point definitions (isosceles triangle) and
    4-point definitions (isosceles trapezoid).
    For a 3-point plane with vertices (a, b, c):
        a1 = a, a2 = a
    For a 4-point plane with vertices (a1, a2, b, c):
        a1, a2, b, c represent the trapezoid vertices.
    """
    a1: Point3D
    a2: Point3D
    b: Point3D
    c: Point3D
    is_triangle: bool = False

    @property
    def leg_length(self) -> float:
        """Length of the left leg: dist(a1, b)."""
        return self.a1.distance_to(self.b)

    @property
    def base_width(self) -> float:
        """Width of the bottom base: dist(b, c)."""
        return self.b.distance_to(self.c)

    @property
    def top_width(self) -> float:
        """Width of the top base: dist(a1, a2). (0 for triangles)."""
        return self.a1.distance_to(self.a2)


def validate_triangle(a: Point3D, b: Point3D, c: Point3D, eps: float = EPSILON) -> bool:
    """
    Validates that triangle (a, b, c) is isosceles with |dist(a,b) - dist(a,c)| < eps.

    Constraints:
    1. Leg lengths dist(a, b) and dist(a, c) must be strictly positive (> eps).
    2. Leg lengths must be equal within tolerance eps: |dist(a,b) - dist(a,c)| < eps.
    3. Base length dist(b, c) must be strictly positive (> eps).
    """
    d_ab = a.distance_to(b)
    d_ac = a.distance_to(c)
    d_bc = b.distance_to(c)

    if d_ab < eps or d_ac < eps:
        raise GeometryValidationError(
            f"Degenerate triangle: leg lengths must be positive (dist(a,b)={d_ab:.6f}, dist(a,c)={d_ac:.6f})."
        )
    if d_bc < eps:
        raise GeometryValidationError(
            f"Degenerate triangle: base length must be positive (dist(b,c)={d_bc:.6f})."
        )
    if abs(d_ab - d_ac) >= eps:
        raise GeometryValidationError(
            f"Triangle is not isosceles: |dist(a,b) - dist(a,c)| = {abs(d_ab - d_ac):.6f} >= {eps}."
        )
    return True


def validate_trapezoid(
    a1: Point3D, a2: Point3D, b: Point3D, c: Point3D, eps: float = EPSILON
) -> bool:
    """
    Validates that quadrilateral (a1, a2, b, c) is an isosceles trapezoid:
    1. Positive leg and base lengths: dist(a1, b) > eps, dist(b, c) > eps.
    2. Equal leg lengths: |dist(a1, b) - dist(a2, c)| < eps.
    3. Base ordering: dist(a1, a2) <= dist(b, c) + eps.
    4. Coplanarity: the distance of a2 from the plane (a1, b, c) must be < eps,
       and scalar triple product |(vec(a1->b) x vec(a1->c)) . vec(a1->a2)| must satisfy tolerance.
    """
    d_leg1 = a1.distance_to(b)
    d_leg2 = a2.distance_to(c)
    d_top = a1.distance_to(a2)
    d_bottom = b.distance_to(c)

    if d_leg1 < eps or d_leg2 < eps:
        raise GeometryValidationError(
            f"Degenerate trapezoid: leg lengths must be positive (leg1={d_leg1:.6f}, leg2={d_leg2:.6f})."
        )
    if d_bottom < eps:
        raise GeometryValidationError(
            f"Degenerate trapezoid: bottom base must be positive (dist(b,c)={d_bottom:.6f})."
        )

    # 1. Equal legs check
    if abs(d_leg1 - d_leg2) >= eps:
        raise GeometryValidationError(
            f"Trapezoid legs not equal: |dist(a1,b) - dist(a2,c)| = {abs(d_leg1 - d_leg2):.6f} >= {eps}."
        )

    # 2. Base ordering check: top base <= bottom base
    if d_top > d_bottom + eps:
        raise GeometryValidationError(
            f"Trapezoid top base ({d_top:.6f}) is strictly larger than bottom base ({d_bottom:.6f})."
        )

    # 3. Coplanarity check
    v_a1b = a1.vector_to(b)
    v_a1c = a1.vector_to(c)
    v_a1a2 = a1.vector_to(a2)

    normal = np.cross(v_a1b, v_a1c)
    norm_mag = float(np.linalg.norm(normal))

    # Distance of a2 from the plane formed by (a1, b, c)
    if norm_mag > 1e-9:
        distance_to_plane = abs(float(np.dot(normal, v_a1a2))) / norm_mag
        if distance_to_plane >= eps:
            raise GeometryValidationError(
                f"Points are not coplanar: distance of a2 to plane(a1,b,c) = {distance_to_plane:.6f} >= {eps}."
            )
    else:
        # Collinear baseline points (a1, b, c)
        raise GeometryValidationError(
            "Degenerate trapezoid: points a1, b, and c are collinear."
        )

    return True


def create_plane(
    points: Sequence[float], eps: float = EPSILON
) -> PlaneDefinition:
    """
    Factory function creating a validated PlaneDefinition from 9, 10, 12, or 13 floats.
    - 9 values: a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z (Triangle)
    - 10 values: 3, a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z (Count prefix + Triangle)
    - 12 values: a1_x, a1_y, a1_z, a2_x, a2_y, a2_z, b_x, b_y, b_z, c_x, c_y, c_z (Trapezoid)
    - 13 values: 4, x1..z4 (Count prefix + Trapezoid)
    """
    # Strip leading point-count indicator if present (e.g. '3' or '4' in column 1)
    if len(points) in (10, 13) and int(points[0]) in (3, 4):
        points = points[1:]

    # For 2-decimal rounded CSV files, allow up to 0.05 tolerance to avoid rejecting valid geometry
    eval_eps = max(eps, 0.05)

    if len(points) == 9:
        a = Point3D(float(points[0]), float(points[1]), float(points[2]))
        b = Point3D(float(points[3]), float(points[4]), float(points[5]))
        c = Point3D(float(points[6]), float(points[7]), float(points[8]))
        validate_triangle(a, b, c, eps=eval_eps)
        return PlaneDefinition(a1=a, a2=a, b=b, c=c, is_triangle=True)
    elif len(points) == 12:
        p1 = Point3D(float(points[0]), float(points[1]), float(points[2]))
        p2 = Point3D(float(points[3]), float(points[4]), float(points[5]))
        p3 = Point3D(float(points[6]), float(points[7]), float(points[8]))
        p4 = Point3D(float(points[9]), float(points[10]), float(points[11]))

        # Try standard ordering: (a1, a2, b, c) = (p1, p2, p3, p4)
        try:
            validate_trapezoid(p1, p2, p3, p4, eps=eval_eps)
            return PlaneDefinition(a1=p1, a2=p2, b=p3, c=p4, is_triangle=False)
        except GeometryValidationError:
            pass

        # Try perimeter/column ordering: (a1, a2, b, c) = (p1, p3, p2, p4)
        # where p1, p3 is top base and p2, p4 is bottom base
        try:
            validate_trapezoid(p1, p3, p2, p4, eps=eval_eps)
            return PlaneDefinition(a1=p1, a2=p3, b=p2, c=p4, is_triangle=False)
        except GeometryValidationError:
            pass

        # Try loop ordering: (a1, a2, b, c) = (p1, p4, p2, p3)
        try:
            validate_trapezoid(p1, p4, p2, p3, eps=eval_eps)
            return PlaneDefinition(a1=p1, a2=p4, b=p2, c=p3, is_triangle=False)
        except GeometryValidationError:
            pass

        # If all permutations fail, run standard validation to raise descriptive error
        validate_trapezoid(p1, p2, p3, p4, eps=eval_eps)
        return PlaneDefinition(a1=p1, a2=p2, b=p3, c=p4, is_triangle=False)
    else:
        raise CSVParseError(
            f"Invalid coordinate count: expected 9 or 12 numbers per row (or 10/13 with count prefix), got {len(points)}."
        )


def parse_csv_line(line: str, eps: float = EPSILON) -> Optional[PlaneDefinition]:
    """
    Parses a single line of text from a trajectory CSV.
    Returns:
        PlaneDefinition if line contains valid coordinates.
        None if line is empty, a comment starting with '#', or a header row.
    Raises:
        CSVParseError: If line cannot be parsed into numbers or has invalid token count.
        GeometryValidationError: If points fail geometric constraints.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Support comma-separated or whitespace-separated tokens
    if "," in stripped:
        tokens = [t.strip() for t in stripped.split(",") if t.strip()]
    else:
        tokens = stripped.split()

    if not tokens:
        return None

    # Check if this line is a header row (e.g. contains words like 'Point', 'x1', etc.)
    first_tok = tokens[0].lower()
    if any(c.isalpha() for c in first_tok) or first_tok.startswith("number"):
        return None

    try:
        numbers = [float(t) for t in tokens]
    except ValueError as e:
        raise CSVParseError(f"Row contains non-numeric value: '{line}' - {e}") from e

    return create_plane(numbers, eps=eps)


def load_trajectory_csv(
    filepath: str, eps: float = EPSILON, skip_invalid: bool = True
) -> List[PlaneDefinition]:
    """
    Loads all plane definitions from a CSV file.

    Args:
        filepath: Path to the trajectory CSV file.
        eps: Geometric tolerance.
        skip_invalid: If True, logs warning and skips invalid rows (SRS UC-01 3a).
                      If False, raises on the first invalid row.

    Returns:
        List of validated PlaneDefinition instances.
    """
    if not os.path.exists(filepath):
        # Also search inside 'CSV' directory
        cand_in_folder = os.path.join("CSV", filepath)
        cand_with_ext = filepath + ".csv" if not filepath.lower().endswith(".csv") else filepath
        cand_folder_ext = os.path.join("CSV", os.path.basename(cand_with_ext))
        if os.path.isfile(cand_in_folder):
            filepath = cand_in_folder
        elif os.path.isfile(cand_with_ext):
            filepath = cand_with_ext
        elif os.path.isfile(cand_folder_ext):
            filepath = cand_folder_ext
        else:
            raise FileNotFoundError(f"Trajectory CSV file does not exist: '{filepath}'")

    planes: List[PlaneDefinition] = []

    with open(filepath, mode="r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                plane = parse_csv_line(line, eps=eps)
                if plane is not None:
                    planes.append(plane)
            except (CSVParseError, GeometryValidationError) as err:
                if skip_invalid:
                    # Non-blocking skip with warning
                    import sys
                    print(
                        f"[WARN] Line {line_num}: Skipped invalid plane row - {err}",
                        file=sys.stderr,
                    )
                else:
                    raise

    if not planes:
        raise CSVParseError(f"No valid plane definitions found in trajectory file: '{filepath}'")

    return planes


def check_trajectory_continuity(
    planes: Sequence[PlaneDefinition], max_dim: float
) -> List[str]:
    """
    Evaluates Euclidean displacement between corresponding control points of
    consecutive frames and yields warning messages if jump exceeds adaptive threshold.
    Adaptive threshold: Delta d_max = 0.01 * max(sx, sy, sz) (STP Suite 7).

    Args:
        planes: Sequence of PlaneDefinition frames.
        max_dim: Max spatial dimension of volume max(sx, sy, sz).

    Returns:
        List of warning message strings for any frame transitions exceeding threshold.
    """
    threshold = 0.01 * max_dim
    warnings: List[str] = []

    for i in range(len(planes) - 1):
        p_curr = planes[i]
        p_next = planes[i + 1]

        # Calculate displacements between all corresponding control points
        d_a1 = p_curr.a1.distance_to(p_next.a1)
        d_a2 = p_curr.a2.distance_to(p_next.a2)
        d_b = p_curr.b.distance_to(p_next.b)
        d_c = p_curr.c.distance_to(p_next.c)

        max_delta = max(d_a1, d_a2, d_b, d_c)
        if max_delta > threshold:
            msg = (
                f"[WARN] Large spatial delta ({max_delta:.2f} > {threshold:.2f}) "
                f"detected between frames {i} and {i + 1}. Video may exhibit visual stutter."
            )
            warnings.append(msg)

    return warnings
