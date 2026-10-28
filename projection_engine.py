"""
2D Plane Projection Engine (Ex0)

This module implements the 3D-to-2D scanline sampling algorithm described in
algorithm_in_detail_edd.md. It samples a 3D volumetric matrix M along a plane
defined by an isosceles triangle or trapezoid, yielding a 2D projection matrix E.

Provides:
- project_plane_baseline: Reference pure-python scanline implementation.
- project_plane_vectorized: High-speed vectorized NumPy implementation (>= 60 FPS target).
- project_plane: Unified dispatcher (defaults to vectorized).
"""

from __future__ import annotations
import math
from typing import Optional
import numpy as np

from geometry import PlaneDefinition
from nrrd_loader import NRRDVolume


def project_plane_baseline(volume: NRRDVolume, plane: PlaneDefinition) -> np.ndarray:
    """
    Reference scanline implementation matching the exact step-by-step pseudo-code
    in algorithm_in_detail_edd.md.

    Args:
        volume: Loaded NRRDVolume containing 3D matrix M.
        plane: Validated PlaneDefinition (triangle or trapezoid).

    Returns:
        2D NumPy array E of shape (H, W) = (ceil(D_ab), ceil(D_bc)), dtype matching volume.
    """
    M = volume.data
    sx, sy, sz = volume.shape

    # Step 1: Dimensions
    D_ab = plane.leg_length
    D_bc = plane.base_width

    if D_ab <= 0 or D_bc <= 0:
        return np.zeros((1, 1), dtype=volume.dtype)

    H = int(math.ceil(D_ab))
    W = int(math.ceil(D_bc))

    E = np.zeros((H, W), dtype=volume.dtype)

    # Step 2: Normalized Side Step Vectors
    a1_arr = plane.a1.to_array()
    a2_arr = plane.a2.to_array()
    b_arr = plane.b.to_array()
    c_arr = plane.c.to_array()

    v_ab_hat = (b_arr - a1_arr) / D_ab
    v_ac_hat = (c_arr - a2_arr) / D_ab

    # Step 3: Outer Scanline Loop (Vertical Axis y)
    y_max = min(int(math.floor(D_ab)), H)
    for y in range(y_max):
        p1 = a1_arr + y * v_ab_hat
        p2 = a2_arr + y * v_ac_hat

        diff = p2 - p1
        d_12 = float(np.linalg.norm(diff))

        if d_12 > 1e-9:
            dp_hat = diff / d_12
        else:
            dp_hat = np.zeros(3, dtype=np.float64)

        # Step 4: Inner Scanline Loop (Horizontal Axis x1)
        x_offset = int(math.floor((D_bc - d_12) / 2.0))
        x1_max = int(math.floor(d_12))

        for x1 in range(x1_max + 1):
            x = x_offset + x1
            if x < 0 or x >= W:
                continue

            p = p1 + x1 * dp_hat
            px = int(math.floor(p[0]))
            py = int(math.floor(p[1]))
            pz = int(math.floor(p[2]))

            # Boundary Handling: assign 0 if outside matrix (REQ-05)
            if 0 <= px < sx and 0 <= py < sy and 0 <= pz < sz:
                E[y, x] = M[px, py, pz]
            else:
                E[y, x] = 0

    return E


def project_plane_vectorized(volume: NRRDVolume, plane: PlaneDefinition) -> np.ndarray:
    """
    High-performance vectorized implementation using NumPy broadcasting and meshgrids.
    Achieves >= 60-300+ FPS by eliminating per-pixel Python loop overhead (NFR-01/02).

    Args:
        volume: Loaded NRRDVolume containing 3D matrix M.
        plane: Validated PlaneDefinition.

    Returns:
        2D NumPy array E of shape (H, W) = (ceil(D_ab), ceil(D_bc)).
    """
    M = volume.data
    sx, sy, sz = volume.shape

    D_ab = plane.leg_length
    D_bc = plane.base_width

    if D_ab <= 0 or D_bc <= 0:
        return np.zeros((1, 1), dtype=volume.dtype)

    H = int(math.ceil(D_ab))
    W = int(math.ceil(D_bc))

    E = np.zeros((H, W), dtype=volume.dtype)

    y_max = min(int(math.floor(D_ab)), H)
    if y_max <= 0:
        return E

    a1_arr = plane.a1.to_array()
    a2_arr = plane.a2.to_array()
    b_arr = plane.b.to_array()
    c_arr = plane.c.to_array()

    v_ab_hat = (b_arr - a1_arr) / D_ab
    v_ac_hat = (c_arr - a2_arr) / D_ab

    # Vectorized vertical axis y: shape (y_max, 1)
    y_arr = np.arange(y_max, dtype=np.float64)[:, None]
    p1 = a1_arr + y_arr * v_ab_hat  # (y_max, 3)
    p2 = a2_arr + y_arr * v_ac_hat  # (y_max, 3)

    diff = p2 - p1  # (y_max, 3)
    d_12 = np.linalg.norm(diff, axis=1)  # (y_max,)

    dp_hat = np.zeros_like(diff)
    nonzero_mask = d_12 > 1e-9
    dp_hat[nonzero_mask] = diff[nonzero_mask] / d_12[nonzero_mask, None]

    x_offset = np.floor((D_bc - d_12) / 2.0).astype(np.int32)  # (y_max,)
    x1_max = np.floor(d_12).astype(np.int32)                    # (y_max,)

    # Horizontal coordinate grid: shape (1, W)
    x_grid = np.arange(W, dtype=np.int32)[None, :]
    x1 = x_grid - x_offset[:, None]  # shape (y_max, W)

    # Valid mask for inner loop range: 0 <= x1 <= floor(d12)
    valid_mask = (x1 >= 0) & (x1 <= x1_max[:, None])  # (y_max, W)

    # Sample coordinates p = p1 + x1 * dp_hat: shape (y_max, W, 3)
    # p1[:, None, :] is (y_max, 1, 3)
    # x1[:, :, None] is (y_max, W, 1)
    # dp_hat[:, None, :] is (y_max, 1, 3)
    p = p1[:, None, :] + x1[:, :, None].astype(np.float64) * dp_hat[:, None, :]

    # Nearest-neighbor integer lookup coordinates
    p_floor = np.floor(p).astype(np.int32)
    px = p_floor[:, :, 0]
    py = p_floor[:, :, 1]
    pz = p_floor[:, :, 2]

    # Boundary handling: check 0 <= px < sx, 0 <= py < sy, 0 <= pz < sz
    in_bounds = (
        valid_mask &
        (px >= 0) & (px < sx) &
        (py >= 0) & (py < sy) &
        (pz >= 0) & (pz < sz)
    )

    # Sample into output frame
    E_sub = np.zeros((y_max, W), dtype=volume.dtype)
    E_sub[in_bounds] = M[px[in_bounds], py[in_bounds], pz[in_bounds]]
    E[:y_max, :] = E_sub

    return E


def project_plane(
    volume: NRRDVolume, plane: PlaneDefinition, vectorized: bool = True
) -> np.ndarray:
    """
    Project 3D volume onto a 2D plane frame.

    Args:
        volume: Loaded NRRDVolume.
        plane: Validated PlaneDefinition.
        vectorized: If True (default), uses the vectorized engine (>= 60 FPS).
                    If False, uses the reference baseline engine.

    Returns:
        2D projected frame E.
    """
    if vectorized:
        return project_plane_vectorized(volume, plane)
    return project_plane_baseline(volume, plane)
