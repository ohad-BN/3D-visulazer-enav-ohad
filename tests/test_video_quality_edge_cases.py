"""
STP Edge Case Test Suite 2: Video & Visual Quality Edge Cases
Covers:
- Intensity Normalization & Flicker Prevention: verifies global_max prevents artificial frame flickering.
- Canvas Jitter & Dynamic Resizing: verifies varying frame dimensions stay centered without coordinate jitter.
- Minimal Video Containers: verifies that a 1-frame MP4 video container finalizes and plays cleanly.
"""

import os
import pytest
import numpy as np
import cv2

from video_writer import (
    MP4VideoWriter,
    normalize_frame_to_uint8,
    inspect_video_file,
)


def test_global_max_anti_flicker_contrast():
    """
    Verify that global_max prevents artificial frame flickering (pulsing contrast).
    If frame 1 has bone (max 2000) and frame 2 has soft tissue (max 200):
      Without global_max: frame 2 is stretched to 255 (blinding flash / flicker).
      With global_max: frame 2 retains true contrast ratio (max value ~25).
    """
    f1_dense = np.full((50, 50), 2000, dtype=np.uint16)
    f2_dim = np.full((50, 50), 200, dtype=np.uint16)

    # 1. Local normalization (no global_max): both stretch to 255
    local_f1 = normalize_frame_to_uint8(f1_dense)
    local_f2 = normalize_frame_to_uint8(f2_dim)
    assert np.max(local_f1) == 255
    assert np.max(local_f2) == 255  # Undesirable flicker!

    # 2. Global normalization with volume maximum (2000.0)
    global_vol_max = 2000.0
    norm_f1 = normalize_frame_to_uint8(f1_dense, global_max=global_vol_max)
    norm_f2 = normalize_frame_to_uint8(f2_dim, global_max=global_vol_max)

    assert np.max(norm_f1) == 255
    # Value should be 200 / 2000 * 255 = 25.5 -> 25
    expected_dim_val = int((200.0 / 2000.0) * 255.0)
    assert np.max(norm_f2) == pytest.approx(expected_dim_val, abs=1)
    # The true 10:1 brightness ratio is preserved, preventing video flicker
    assert np.max(norm_f1) > 9 * np.max(norm_f2)


def test_variable_frame_sizes_centered_on_canvas_jitter_free(tmp_path):
    """
    Verify that frames with dynamic, changing sizes (e.g., zooming in/out)
    are positioned at the exact center of the standardized canvas with zero spatial jitter.
    """
    mp4_file = str(tmp_path / "jitter_free_test.mp4")
    canvas_w, canvas_h = 160, 120
    center_x, center_y = canvas_w // 2, canvas_h // 2  # (80, 60)

    # Define frames of different sizes:
    # Small (40x30), Medium (100x70), Large (160x120)
    frame_sizes = [(30, 40), (70, 100), (120, 160)]

    with MP4VideoWriter(mp4_file, fps=30.0, target_size=(canvas_w, canvas_h)) as writer:
        for h, w in frame_sizes:
            frame = np.zeros((h, w), dtype=np.uint8)
            # Mark the exact center of this frame with value 255
            local_cy, local_cx = h // 2, w // 2
            frame[local_cy, local_cx] = 255
            writer.add_frame(frame)

    assert writer.frame_count == len(frame_sizes)

    # Open the generated video and verify that in every frame,
    # the marker pixel appears at the exact canvas center (80, 60)
    cap = cv2.VideoCapture(mp4_file)
    assert cap.isOpened()

    for idx, (h, w) in enumerate(frame_sizes):
        ret, decoded_frame = cap.read()
        assert ret is True, f"Failed to read frame {idx}"
        assert decoded_frame.shape == (canvas_h, canvas_w, 3)

        # Grayscale channel: verify that pixel at (center_y, center_x) is bright (>= 200 due to compression)
        center_val = decoded_frame[center_y, center_x, 0]
        assert center_val > 200, (
            f"Frame {idx}: Center marker not found at canvas center ({center_x}, {center_y}); got {center_val}"
        )

    cap.release()


def test_single_frame_minimal_mp4_container(tmp_path):
    """
    Verify that an MP4 video containing only a single frame is written and finalized cleanly,
    producing a valid, uncorrupted video container.
    """
    mp4_file = str(tmp_path / "single_frame.mp4")
    w, h = 100, 100

    single_frame = np.full((h, w), 128, dtype=np.uint8)

    with MP4VideoWriter(mp4_file, fps=30.0, target_size=(w, h)) as writer:
        writer.add_frame(single_frame)

    assert os.path.exists(mp4_file)
    assert os.path.getsize(mp4_file) > 0

    # Inspect container metadata
    info = inspect_video_file(mp4_file)
    assert info["exists"] is True
    assert info["is_valid"] is True
    assert info["frame_count"] == 1
    assert info["width"] == w
    assert info["height"] == h

    # Re-read with OpenCV VideoCapture
    cap = cv2.VideoCapture(mp4_file)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret is True
    assert frame.shape == (h, w, 3)
    # Second read must return False (end of stream)
    ret_end, _ = cap.read()
    assert ret_end is False
    cap.release()
