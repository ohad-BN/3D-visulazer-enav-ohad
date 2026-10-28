"""
Automated Test Suite for NRRD Volume Loader & Data Integrity (Phase 3)
Covers STP Suite 3:
- Ingestion of real dataset (MRHead_1.nrrd)
- Synthetic volumes across supported data types (uint8, uint16, uint32)
- Bounds validation (1 <= sx, sy, sz <= 1024)
- Corrupt, empty, and non-existent file safety
- Memory layout contiguity (C-contiguous)
- Boundary access and out-of-bounds safety
"""

import os
import tempfile
import pytest
import numpy as np
import nrrd

from nrrd_loader import (
    NRRDVolume,
    load_nrrd,
    NRRDLoadError,
    NRRDFormatError,
)


def get_mrhead_path() -> str:
    """Return path to MRHead_1.nrrd in repository root."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "MRHead_1.nrrd")


# ============================================================================
# 1. Real Medical Dataset Ingestion
# ============================================================================

def test_load_real_mrhead_dataset():
    """Verify loading of the provided MRHead_1.nrrd volume."""
    path = get_mrhead_path()
    assert os.path.exists(path), f"MRHead_1.nrrd not found at {path}"

    volume = load_nrrd(path)
    assert isinstance(volume, NRRDVolume)
    assert volume.data.ndim == 3
    assert volume.shape == (256, 256, 130)
    assert volume.sx == 256
    assert volume.sy == 256
    assert volume.sz == 130
    assert volume.max_dimension == 256
    assert volume.data.flags.c_contiguous is True

    # Check non-empty voxel content
    assert volume.data.max() > 0


# ============================================================================
# 2. Synthetic Volume Ingestion across Data Types (uint8, uint16, uint32)
# ============================================================================

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_load_synthetic_volumes(tmp_path, dtype):
    """Verify loading synthetic volumes of uint8, uint16, and uint32 types."""
    shape = (20, 30, 40)
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    nrrd_file = tmp_path / f"synthetic_{dtype.__name__}.nrrd"
    nrrd.write(str(nrrd_file), data)

    volume = load_nrrd(str(nrrd_file))
    assert volume.shape == shape
    assert volume.dtype == np.dtype(dtype)
    assert volume.data.flags.c_contiguous is True
    # Verify exact value preservation
    np.testing.assert_array_equal(volume.data, data)


# ============================================================================
# 3. File Error & Corruption Handling
# ============================================================================

def test_load_non_existent_file():
    """Verify missing file raises NRRDLoadError."""
    with pytest.raises(NRRDLoadError, match="does not exist"):
        load_nrrd("non_existent_volume_file.nrrd")


def test_load_empty_file(tmp_path):
    """Verify empty (0-byte) file raises NRRDLoadError."""
    empty_file = tmp_path / "empty.nrrd"
    empty_file.write_bytes(b"")

    with pytest.raises(NRRDLoadError, match="empty"):
        load_nrrd(str(empty_file))


def test_load_corrupted_file(tmp_path):
    """Verify file with corrupt header/content raises NRRDLoadError."""
    corrupt_file = tmp_path / "corrupt.nrrd"
    corrupt_file.write_bytes(b"NRRD0004\ntype: bad_type\ndata: garbage\n\n\x00\x01\x02")

    with pytest.raises(NRRDLoadError, match="Failed to parse"):
        load_nrrd(str(corrupt_file))


# ============================================================================
# 4. Dimensionality & Boundary Constraints (1 <= dim <= 1024)
# ============================================================================

def test_invalid_dimensionality_2d(tmp_path):
    """Verify 2D dataset raises NRRDFormatError."""
    data_2d = np.zeros((64, 64), dtype=np.uint8)
    nrrd_file = tmp_path / "data_2d.nrrd"
    nrrd.write(str(nrrd_file), data_2d)

    with pytest.raises(NRRDFormatError, match="Expected a 3D volumetric matrix"):
        load_nrrd(str(nrrd_file))


def test_invalid_dimensionality_4d(tmp_path):
    """Verify 4D dataset raises NRRDFormatError."""
    data_4d = np.zeros((10, 10, 10, 2), dtype=np.uint8)
    nrrd_file = tmp_path / "data_4d.nrrd"
    nrrd.write(str(nrrd_file), data_4d)

    with pytest.raises(NRRDFormatError, match="Expected a 3D volumetric matrix"):
        load_nrrd(str(nrrd_file))


def test_dimension_exceeds_1024(tmp_path):
    """Verify dimensions > 1024 raise NRRDFormatError (SRS REQ-01 constraint)."""
    # Create small array with header claiming dimension > 1024
    header = {"type": "uint8", "dimension": 3, "sizes": [1025, 10, 10], "encoding": "raw"}
    # Fake a minimal write by saving actual small array then overriding header if possible,
    # or write a 1025x1x1 array
    data = np.zeros((1025, 1, 1), dtype=np.uint8)
    nrrd_file = tmp_path / "oversized.nrrd"
    nrrd.write(str(nrrd_file), data)

    with pytest.raises(NRRDFormatError, match="violates constraint 1 <= sx <= 1024"):
        load_nrrd(str(nrrd_file))


def test_unsupported_data_type(tmp_path):
    """Verify non-integer types (e.g., float32) raise NRRDFormatError."""
    data_float = np.zeros((5, 5, 5), dtype=np.float32)
    nrrd_file = tmp_path / "float.nrrd"
    nrrd.write(str(nrrd_file), data_float)

    with pytest.raises(NRRDFormatError, match="Unsupported data type"):
        load_nrrd(str(nrrd_file))


# ============================================================================
# 5. Safe Coordinate & Boundary Access
# ============================================================================

def test_safe_voxel_access_and_bounds():
    """Verify is_inside and get_voxel boundary behavior."""
    data = np.zeros((10, 20, 30), dtype=np.uint8)
    data[5, 5, 5] = 42

    vol = NRRDVolume(data=data, header={}, filepath="dummy.nrrd")

    # In-bounds lookup
    assert vol.is_inside(5.0, 5.0, 5.0) is True
    assert vol.get_voxel(5, 5, 5) == 42
    assert vol.get_voxel(0, 0, 0) == 0

    # Boundary edge cases
    assert vol.is_inside(0.0, 0.0, 0.0) is True
    assert vol.is_inside(9.99, 19.99, 29.99) is True
    assert vol.is_inside(10.0, 5.0, 5.0) is False  # out on x
    assert vol.is_inside(-0.1, 5.0, 5.0) is False  # negative x

    # Out-of-bounds safe voxel lookup returns 0 (REQ-05)
    assert vol.get_voxel(-1, 5, 5) == 0
    assert vol.get_voxel(10, 5, 5) == 0
    assert vol.get_voxel(5, 20, 5) == 0
    assert vol.get_voxel(5, 5, 30) == 0
