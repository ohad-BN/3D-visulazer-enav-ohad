"""
Video Writer Module (Ex0)

This module handles encoding 2D projected matrices E into an MP4 video container
using OpenCV (cv2.VideoWriter). It provides intensity normalization to 8-bit BGR,
canvas dimension standardization across frames, and video container integrity inspection (STP Suite 6).
"""

from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np


class VideoWriterError(RuntimeError):
    """Raised when video writing fails or container cannot be opened."""
    pass


def normalize_frame_to_uint8(
    frame: np.ndarray, global_max: Optional[float] = None
) -> np.ndarray:
    """
    Normalizes a 2D integer or float matrix to uint8 (0-255).

    Args:
        frame: 2D array of any numeric type (uint8, uint16, int16, uint32, float).
        global_max: Optional maximum value across the entire volume or dataset
                    to prevent frame-to-frame intensity flicker.

    Returns:
        2D NumPy array of dtype uint8 with values in [0, 255].
    """
    if frame.dtype == np.uint8 and global_max is None:
        return frame

    # Clip negative values if any
    data = np.maximum(frame, 0).astype(np.float32)

    denom = float(global_max) if (global_max is not None and global_max > 0) else float(np.max(data))
    if denom <= 0:
        return np.zeros(frame.shape, dtype=np.uint8)

    scaled = (data / denom) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


class MP4VideoWriter:
    """
    Encapsulates OpenCV VideoWriter for creating compliant, uncorrupted MP4 video files.
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        fourcc: str = "mp4v",
        target_size: Optional[Tuple[int, int]] = None,
        global_max: Optional[float] = None,
    ):
        """
        Args:
            output_path: Destination .mp4 file path.
            fps: Frame rate for output video (default: 30.0).
            fourcc: Four-character codec code (default: 'mp4v').
            target_size: (width, height) canvas dimensions for all frames. If None,
                         the first frame's dimensions determine canvas size.
            global_max: Optional global normalization maximum across volume.
        """
        self.output_path = output_path
        self.fps = fps
        self.fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.target_size = target_size  # (W, H)
        self.global_max = global_max
        self._writer: Optional[cv2.VideoWriter] = None
        self.frame_count: int = 0

    def _init_writer(self, width: int, height: int) -> None:
        """Initializes the cv2.VideoWriter once width and height are known."""
        self.target_size = (width, height)
        out_dir = os.path.dirname(os.path.abspath(self.output_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        self._writer = cv2.VideoWriter(
            self.output_path, self.fourcc_code, self.fps, (width, height), isColor=True
        )
        if not self._writer.isOpened():
            raise VideoWriterError(f"Failed to open OpenCV VideoWriter for path: {self.output_path}")

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Normalizes, standardizes, and writes a single 2D projection frame to the video stream.

        Args:
            frame: 2D NumPy array of shape (H, W).
        """
        gray8 = normalize_frame_to_uint8(frame, global_max=self.global_max)
        h, w = gray8.shape

        if self._writer is None:
            canvas_w, canvas_h = self.target_size if self.target_size else (w, h)
            self._init_writer(canvas_w, canvas_h)

        canvas_w, canvas_h = self.target_size  # type: ignore

        # Standardize canvas: center frame on canvas
        if (w, h) != (canvas_w, canvas_h):
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            # Center frame within canvas
            offset_x = max(0, (canvas_w - w) // 2)
            offset_y = max(0, (canvas_h - h) // 2)
            paste_w = min(w, canvas_w - offset_x)
            paste_h = min(h, canvas_h - offset_y)
            canvas[offset_y:offset_y + paste_h, offset_x:offset_x + paste_w] = gray8[:paste_h, :paste_w]
            gray8 = canvas

        # Convert to 3-channel BGR for standard MP4 codec compatibility
        bgr_frame = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        self._writer.write(bgr_frame)
        self.frame_count += 1

    def close(self) -> None:
        """Releases the video writer resource and finalizes the MP4 file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> MP4VideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def inspect_video_file(mp4_path: str) -> Dict[str, Any]:
    """
    Inspects an MP4 file using OpenCV VideoCapture to verify container integrity,
    resolution, frame count, and playability (STP Suite 6).

    Returns:
        Dictionary with keys: 'exists', 'frame_count', 'width', 'height', 'fps', 'is_valid'.
    """
    if not os.path.exists(mp4_path):
        return {"exists": False, "is_valid": False}

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return {"exists": True, "is_valid": False}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    # Test reading at least one frame
    ret, test_frame = cap.read()
    cap.release()

    is_valid = ret and (test_frame is not None) and (frame_count > 0)
    return {
        "exists": True,
        "is_valid": is_valid,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }
