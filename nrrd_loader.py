"""
NRRD Volume Loader & Validator Module (Ex0)

This module handles loading 3D NRRD datasets, validating dimensionality and constraints
(1 <= sx, sy, sz <= 1024), ensuring contiguous C-order memory layout, and providing
safe voxel access (STP Suite 3).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import nrrd

# Supported volume data types (SRS REQ-01 & standard medical formats)
SUPPORTED_DTYPES = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.int8,
    np.int16,
    np.int32,
)


class NRRDLoadError(RuntimeError):
    """Raised when an NRRD file cannot be read, found, or is corrupt."""
    pass


class NRRDFormatError(ValueError):
    """Raised when an NRRD file violates dimensionality, dtype, or size constraints."""
    pass


@dataclass(frozen=True)
class NRRDVolume:
    """
    Encapsulates a loaded 3D volumetric matrix M and its metadata.
    Guarantees contiguous C-order layout and 1 <= sx, sy, sz <= 1024.
    """
    data: np.ndarray  # Contiguous 3D array (sx, sy, sz)
    header: Dict[str, Any]
    filepath: str

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return dimensions (sx, sy, sz)."""
        return int(self.data.shape[0]), int(self.data.shape[1]), int(self.data.shape[2])

    @property
    def sx(self) -> int:
        return int(self.data.shape[0])

    @property
    def sy(self) -> int:
        return int(self.data.shape[1])

    @property
    def sz(self) -> int:
        return int(self.data.shape[2])

    @property
    def max_dimension(self) -> int:
        """Maximum dimension among sx, sy, sz (used for adaptive thresholds)."""
        return max(self.sx, self.sy, self.sz)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def is_inside(self, x: float, y: float, z: float) -> bool:
        """Returns True if point (x, y, z) is within volumetric matrix bounds [0, s)."""
        return 0.0 <= x < self.sx and 0.0 <= y < self.sy and 0.0 <= z < self.sz

    def get_voxel(self, x: int, y: int, z: int) -> int:
        """
        Safely retrieves voxel value at integer coordinates (x, y, z).
        Returns 0 if coordinates are out of bounds (SRS REQ-05).
        """
        if 0 <= x < self.sx and 0 <= y < self.sy and 0 <= z < self.sz:
            return int(self.data[x, y, z])
        return 0


def load_nrrd(filepath: str) -> NRRDVolume:
    """
    Loads an NRRD file, validates its constraints, and returns an NRRDVolume.

    Validation rules (SRS REQ-01, NFR-03, STP Suite 3):
    1. File must exist, be accessible, and non-empty.
    2. Data must be exactly 3-dimensional.
    3. Dimensions must satisfy 1 <= sx, sy, sz <= 1024.
    4. Data type must be an integer volume representation (uint8, uint16, uint32, int16, etc.).
    5. Memory layout must be contiguous in C-order for high-speed sampling.
    """
    if not os.path.exists(filepath):
        # Also search inside 'NRRD' directory
        cand_in_folder = os.path.join("NRRD", filepath)
        cand_with_ext = filepath + ".nrrd" if not filepath.lower().endswith(".nrrd") else filepath
        cand_folder_ext = os.path.join("NRRD", os.path.basename(cand_with_ext))
        if os.path.isfile(cand_in_folder):
            filepath = cand_in_folder
        elif os.path.isfile(cand_with_ext):
            filepath = cand_with_ext
        elif os.path.isfile(cand_folder_ext):
            filepath = cand_folder_ext
        else:
            raise NRRDLoadError(f"NRRD file does not exist at path: '{filepath}'")

    if os.path.getsize(filepath) == 0:
        raise NRRDLoadError(f"NRRD file is empty (0 bytes): '{filepath}'")

    try:
        data, header = nrrd.read(filepath)
    except Exception as e:
        raise NRRDLoadError(f"Failed to parse NRRD file '{filepath}': {str(e)}") from e

    # 1. Dimensionality check: exactly 3D
    if data.ndim != 3:
        raise NRRDFormatError(
            f"Expected a 3D volumetric matrix, got {data.ndim}D array with shape {data.shape}."
        )

    sx, sy, sz = data.shape

    # 2. Bounding limits check (1 <= dim <= 1024)
    for dim_name, dim_val in (("sx", sx), ("sy", sy), ("sz", sz)):
        if not (1 <= dim_val <= 1024):
            raise NRRDFormatError(
                f"Dimension {dim_name}={dim_val} violates constraint 1 <= {dim_name} <= 1024."
            )

    # 3. Data type validation
    if not any(np.issubdtype(data.dtype, supported) for supported in SUPPORTED_DTYPES):
        raise NRRDFormatError(
            f"Unsupported data type '{data.dtype}'. Supported types: unsigned/signed 8, 16, 32-bit integers."
        )

    # 4. Ensure contiguous C-order layout for CPU cache optimization (EDD Section 3)
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)

    return NRRDVolume(data=data, header=header, filepath=filepath)
