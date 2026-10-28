"""
Smoke Test: Environment, Dependencies & Module Architecture Setup (Phase 1)

Verifies:
1. Python version meets specification (>= 3.10).
2. Third-party packages (NumPy, pynrrd, OpenCV, pytest) are correctly installed and importable.
3. Core project modules (geometry, nrrd_loader, projection_engine, video_writer, Ex0)
   load cleanly without syntax or circular import issues.
4. Baseline dataset MRHead_1.nrrd exists on disk.
"""

import sys
import os
import pytest


def test_python_version():
    """Verify Python runtime version >= 3.10."""
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 10), f"Python 3.10+ required, current: {major}.{minor}"


def test_external_dependencies():
    """Verify third-party library imports and versions."""
    import numpy as np
    import nrrd
    import cv2
    import pytest as pt

    assert np.__version__ is not None
    assert nrrd.__version__ is not None
    assert cv2.__version__ is not None
    assert pt.__version__ is not None


def test_internal_module_imports():
    """Verify project modules import cleanly without circular dependencies."""
    import geometry
    import nrrd_loader
    import projection_engine
    import video_writer
    import Ex0

    assert hasattr(geometry, "Point3D")
    assert hasattr(geometry, "PlaneDefinition")
    assert hasattr(geometry, "validate_triangle")
    assert hasattr(geometry, "validate_trapezoid")
    assert hasattr(nrrd_loader, "load_nrrd")
    assert hasattr(projection_engine, "project_plane")
    assert hasattr(video_writer, "MP4VideoWriter")
    assert hasattr(Ex0, "run_pipeline")


def test_mrhead_dataset_present():
    """Verify that sample MRHead_1.nrrd dataset is present in workspace root."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nrrd_path = os.path.join(workspace_root, "MRHead_1.nrrd")
    assert os.path.exists(nrrd_path), f"Expected MRHead_1.nrrd at {nrrd_path}"
    assert os.path.getsize(nrrd_path) > 1000, "MRHead_1.nrrd file appears empty or truncated"


def test_geometry_basic_sanity():
    """Verify basic geometric structures and distances."""
    from geometry import Point3D, PlaneDefinition

    p1 = Point3D(0.0, 0.0, 0.0)
    p2 = Point3D(3.0, 4.0, 0.0)
    assert p1.distance_to(p2) == pytest.approx(5.0, abs=1e-6)

    plane = PlaneDefinition(a1=p1, a2=p1, b=p2, c=Point3D(6.0, 0.0, 0.0), is_triangle=True)
    assert plane.is_triangle is True
    assert plane.leg_length == pytest.approx(5.0, abs=1e-6)
