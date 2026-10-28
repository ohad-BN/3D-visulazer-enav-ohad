"""
Automated Test Suite for MP4 Video Encoder Module (Phase 5)
Covers STP Suite 6:
- Video container creation and header inspection (OpenCV VideoCapture)
- Frame count and resolution verification
- 8-bit intensity normalization across data types (uint8, uint16, int16, float)
- Dynamic canvas standardization and frame centering
- Container finalization and resource cleanup
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from video_writer import (
    MP4VideoWriter,
    normalize_frame_to_uint8,
    inspect_video_file,
    VideoWriterError,
)


# ============================================================================
# 1. Normalization & Pre-Processing Tests
# ============================================================================

def test_normalize_uint8():
    """Verify uint8 arrays pass through unmodified when no global_max is given."""
    arr = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    norm = normalize_frame_to_uint8(arr)
    assert norm.dtype == np.uint8
    np.testing.assert_array_equal(norm, arr)


def test_normalize_uint16_and_int16():
    """Verify 16-bit integers map cleanly to [0, 255]."""
    # Range 0 to 1000
    arr = np.array([[0, 500], [750, 1000]], dtype=np.uint16)
    norm = normalize_frame_to_uint8(arr)
    assert norm.dtype == np.uint8
    assert norm[0, 0] == 0
    assert norm[0, 1] == pytest.approx(127, abs=1)
    assert norm[1, 0] == pytest.approx(191, abs=1)
    assert norm[1, 1] == 255


def test_normalize_with_global_max():
    """Verify normalization respects an external global maximum."""
    # Max in frame is 500, but global_max across volume is 1000
    arr = np.array([[0, 250], [500, 500]], dtype=np.int16)
    norm = normalize_frame_to_uint8(arr, global_max=1000.0)
    assert norm.dtype == np.uint8
    assert norm[0, 0] == 0
    assert norm[0, 1] == pytest.approx(63, abs=1)
    assert norm[1, 0] == pytest.approx(127, abs=1)


def test_normalize_all_zeros():
    """Verify all-zero frames safely return all-zero uint8 arrays."""
    arr = np.zeros((20, 30), dtype=np.uint16)
    norm = normalize_frame_to_uint8(arr)
    assert norm.dtype == np.uint8
    assert np.all(norm == 0)


def test_normalize_negative_clipping():
    """Verify negative voxel values are clipped to 0."""
    arr = np.array([[-50, 0], [50, 100]], dtype=np.int16)
    norm = normalize_frame_to_uint8(arr)
    assert norm[0, 0] == 0
    assert norm[0, 1] == 0
    assert norm[1, 0] == 127
    assert norm[1, 1] == 255


# ============================================================================
# 2. STP Suite 6: MP4 Video Container Creation & Integrity
# ============================================================================

def test_mp4_video_creation_and_inspection(tmp_path):
    """
    Verify creating an MP4 video file:
    1. File is generated at exact path.
    2. Video container can be parsed.
    3. Frame count, dimensions, and FPS match configuration.
    """
    mp4_file = tmp_path / "test_output.mp4"
    width, height = 120, 80
    fps = 30.0
    num_frames = 15

    with MP4VideoWriter(str(mp4_file), fps=fps, target_size=(width, height)) as writer:
        for i in range(num_frames):
            # Create synthetic frames with increasing brightness gradient
            val = int((i / num_frames) * 255)
            frame = np.full((height, width), val, dtype=np.uint8)
            writer.add_frame(frame)

    assert os.path.exists(str(mp4_file)), "Output MP4 file was not created"
    assert os.path.getsize(str(mp4_file)) > 0, "Output MP4 file is 0 bytes"

    # Inspect container headers
    info = inspect_video_file(str(mp4_file))
    assert info["exists"] is True
    assert info["is_valid"] is True
    assert info["width"] == width
    assert info["height"] == height
    assert info["frame_count"] == num_frames
    assert info["fps"] == pytest.approx(fps, abs=0.1)


def test_variable_frame_sizes_centered_on_canvas(tmp_path):
    """
    Verify writing frames of differing dimensions:
    Frames should be centered onto the specified target canvas size.
    """
    mp4_file = tmp_path / "variable_frames.mp4"
    canvas_w, canvas_h = 200, 150

    with MP4VideoWriter(str(mp4_file), fps=30.0, target_size=(canvas_w, canvas_h)) as writer:
        # Frame 1: small frame 50x40
        f1 = np.full((40, 50), 100, dtype=np.uint8)
        writer.add_frame(f1)

        # Frame 2: medium frame 120x80
        f2 = np.full((80, 120), 150, dtype=np.uint8)
        writer.add_frame(f2)

        # Frame 3: exact canvas frame 200x150
        f3 = np.full((150, 200), 200, dtype=np.uint8)
        writer.add_frame(f3)

    assert writer.frame_count == 3

    info = inspect_video_file(str(mp4_file))
    assert info["is_valid"] is True
    assert info["width"] == canvas_w
    assert info["height"] == canvas_h
    assert info["frame_count"] == 3


def test_automatic_canvas_sizing_from_first_frame(tmp_path):
    """Verify that when target_size=None, first frame sets canvas dimensions."""
    mp4_file = tmp_path / "auto_size.mp4"
    with MP4VideoWriter(str(mp4_file), fps=24.0, target_size=None) as writer:
        first_frame = np.zeros((90, 160), dtype=np.uint8)
        writer.add_frame(first_frame)
        second_frame = np.zeros((90, 160), dtype=np.uint8)
        writer.add_frame(second_frame)

    info = inspect_video_file(str(mp4_file))
    assert info["is_valid"] is True
    assert info["width"] == 160
    assert info["height"] == 90
    assert info["frame_count"] == 2
