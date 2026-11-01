"""
Performance Profiler & Regression Pinpointer (Intro2SE / Ex0)

This module measures execution times across 4 isolated pipeline stages:
  - Stage A: Geometry Parsing & Validation (geometry.py)
  - Stage B: 3D-to-2D Projection Sampling Math (projection_engine.py)
  - Stage C: Frame Normalization & Canvas Centering (video_writer.py)
  - Stage D: Video Encoding Container I/O (cv2.VideoWriter)

It compares current latency against a Git-tagged baseline (.perf_baseline.json)
and pinpoints the exact component responsible if a regression is detected.
"""

from __future__ import annotations
import os
import sys
import time
import json
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2

from geometry import Point3D, PlaneDefinition, validate_trapezoid, create_plane
from nrrd_loader import NRRDVolume, load_nrrd
from projection_engine import project_plane_vectorized
from video_writer import MP4VideoWriter, normalize_frame_to_uint8

BASELINE_FILE = ".perf_baseline.json"
HISTORY_CSV_FILE = os.path.join("reports", "performance_history.csv")
HISTORY_MD_FILE = os.path.join("reports", "performance_history.md")
REGRESSION_THRESHOLD_PCT = 30.0  # Alert if stage latency increases by > 30%


@dataclass
class StageMetrics:
    name: str
    component: str
    current_ms: float
    baseline_ms: Optional[float] = None
    delta_pct: Optional[float] = None
    status: str = "OK"
    culprit: bool = False
    details: str = ""


@dataclass
class ProfileSummary:
    commit_hash: str
    commit_message: str
    timestamp: str
    stage_a_geometry_ms: float
    stage_b_projection_ms: float
    stage_c_normalization_ms: float
    stage_d_encoding_ms: float
    overall_latency_ms: float
    overall_fps: float


class PerformanceProfiler:
    """Orchestrates stage-by-stage profiling and regression culprit pinpointing."""

    @classmethod
    def get_git_info(cls) -> Tuple[str, str]:
        """Retrieves short commit hash and first-line commit message from Git."""
        try:
            h_proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True
            )
            commit_hash = h_proc.stdout.strip()

            m_proc = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                capture_output=True, text=True, check=True
            )
            commit_msg = m_proc.stdout.strip()
            return commit_hash, commit_msg
        except Exception:
            return "UNCOMMITTED", "Working Directory (No Git Metadata)"

    @classmethod
    def get_git_diff_files(cls) -> List[str]:
        """Returns list of files modified between HEAD and HEAD~1 (or working tree)."""
        try:
            # Check uncommitted modified files first
            diff_proc = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, check=True
            )
            files = [f.strip() for f in diff_proc.stdout.splitlines() if f.strip()]
            if not files:
                # If clean working tree, check last commit
                diff_proc = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                    capture_output=True, text=True, check=True
                )
                files = [f.strip() for f in diff_proc.stdout.splitlines() if f.strip()]
            return files
        except Exception:
            return []

    @classmethod
    def profile_all_stages(cls, num_frames: int = 50) -> ProfileSummary:
        """
        Executes isolated micro-benchmarks for Stages A, B, C, D and returns ProfileSummary.
        """
        # ------------------------------------------------------------------
        # Setup test data
        # ------------------------------------------------------------------
        repo_root = os.path.dirname(os.path.abspath(__file__))
        nrrd_cand = os.path.join(repo_root, "MRHead_1.nrrd")
        if not os.path.isfile(nrrd_cand):
            nrrd_cand = os.path.join(repo_root, "NRRD", "MRHead_1.nrrd")

        if os.path.isfile(nrrd_cand):
            volume = load_nrrd(nrrd_cand)
        else:
            # Fallback synthetic volume
            data = np.zeros((256, 256, 130), dtype=np.int16)
            volume = NRRDVolume(data=data, header={}, filepath="synthetic.nrrd")

        # 50 representative slicing planes
        planes: List[PlaneDefinition] = []
        for i in range(num_frames):
            z = 20.0 + (i * 1.5)
            p = PlaneDefinition(
                a1=Point3D(30.0, 180.0, z),
                a2=Point3D(180.0, 180.0, z),
                b=Point3D(30.0, 30.0, z),
                c=Point3D(180.0, 30.0, z),
                is_triangle=False,
            )
            planes.append(p)

        # ------------------------------------------------------------------
        # Stage A: Geometry Validation & Coordinate Math (geometry.py)
        # ------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        for p in planes:
            _ = validate_trapezoid(p.a1, p.a2, p.b, p.c)
            _ = p.leg_length
            _ = p.base_width
        elapsed_a = (time.perf_counter_ns() - t0) / 1e6  # ms
        stage_a_ms = elapsed_a / num_frames

        # ------------------------------------------------------------------
        # Stage B: 3D-to-2D Projection Sampling Engine (projection_engine.py)
        # ------------------------------------------------------------------
        # Warmup
        _ = project_plane_vectorized(volume, planes[0])

        frames: List[np.ndarray] = []
        t0 = time.perf_counter_ns()
        for p in planes:
            f = project_plane_vectorized(volume, p)
            frames.append(f)
        elapsed_b = (time.perf_counter_ns() - t0) / 1e6
        stage_b_ms = elapsed_b / num_frames

        # ------------------------------------------------------------------
        # Stage C: Frame Normalization & Canvas Centering (video_writer.py)
        # ------------------------------------------------------------------
        canvas_w, canvas_h = 250, 250
        norm_frames: List[np.ndarray] = []
        global_max = float(volume.data.max()) if volume.data.size > 0 else 1000.0

        t0 = time.perf_counter_ns()
        for f in frames:
            gray8 = normalize_frame_to_uint8(f, global_max=global_max)
            h, w = gray8.shape
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            ox = max(0, (canvas_w - w) // 2)
            oy = max(0, (canvas_h - h) // 2)
            pw = min(w, canvas_w - ox)
            ph = min(h, canvas_h - oy)
            canvas[oy:oy + ph, ox:ox + pw] = gray8[:ph, :pw]
            norm_frames.append(canvas)
        elapsed_c = (time.perf_counter_ns() - t0) / 1e6
        stage_c_ms = elapsed_c / num_frames

        # ------------------------------------------------------------------
        # Stage D: Video Encoding Container I/O (cv2.VideoWriter)
        # ------------------------------------------------------------------
        temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_video_fd)

        t0 = time.perf_counter_ns()
        try:
            with MP4VideoWriter(temp_video_path, fps=30.0, target_size=(canvas_w, canvas_h)) as writer:
                for nf in norm_frames:
                    writer.add_frame(nf)
        finally:
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        elapsed_d = (time.perf_counter_ns() - t0) / 1e6
        stage_d_ms = elapsed_d / num_frames

        # Totals
        total_latency_ms = stage_a_ms + stage_b_ms + stage_c_ms + stage_d_ms
        overall_fps = (1000.0 / total_latency_ms) if total_latency_ms > 0 else 0.0

        c_hash, c_msg = cls.get_git_info()
        t_now = time.strftime("%Y-%m-%d %H:%M:%S")

        return ProfileSummary(
            commit_hash=c_hash,
            commit_message=c_msg,
            timestamp=t_now,
            stage_a_geometry_ms=stage_a_ms,
            stage_b_projection_ms=stage_b_ms,
            stage_c_normalization_ms=stage_c_ms,
            stage_d_encoding_ms=stage_d_ms,
            overall_latency_ms=total_latency_ms,
            overall_fps=overall_fps,
        )

    @classmethod
    def save_baseline(cls, summary: ProfileSummary, filepath: str = BASELINE_FILE) -> None:
        """Saves ProfileSummary into JSON baseline file."""
        data = asdict(summary)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[+] Golden baseline saved to: '{filepath}' (Commit {summary.commit_hash})")

    @classmethod
    def load_baseline(cls, filepath: str = BASELINE_FILE) -> Optional[ProfileSummary]:
        """Loads ProfileSummary from JSON baseline file if it exists."""
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProfileSummary(**data)
        except Exception:
            return None

    @classmethod
    def analyze_regression(
        cls, current: ProfileSummary, baseline: Optional[ProfileSummary]
    ) -> List[StageMetrics]:
        """
        Compares each stage against the baseline.
        Flags any stage with latency increase > REGRESSION_THRESHOLD_PCT (+30%).
        """
        modified_files = cls.get_git_diff_files()

        def compute_stage(
            name: str, component: str, curr_val: float, base_val: Optional[float], related_file: str
        ) -> StageMetrics:
            if base_val is None or base_val <= 0:
                return StageMetrics(name=name, component=component, current_ms=curr_val)

            delta = ((curr_val - base_val) / base_val) * 100.0
            is_culprit = delta > REGRESSION_THRESHOLD_PCT
            status = "REGRESSION" if is_culprit else ("FASTER" if delta < -10 else "HEALTHY")

            detail = ""
            if is_culprit:
                file_note = f" (File '{related_file}' was modified!)" if related_file in modified_files else ""
                detail = f"Slowdown +{delta:.1f}% vs baseline ({base_val:.2f}ms -> {curr_val:.2f}ms){file_note}"

            return StageMetrics(
                name=name,
                component=component,
                current_ms=curr_val,
                baseline_ms=base_val,
                delta_pct=delta,
                status=status,
                culprit=is_culprit,
                details=detail,
            )

        stages = [
            compute_stage(
                "Stage A: Geometry Validation",
                "geometry.py",
                current.stage_a_geometry_ms,
                baseline.stage_a_geometry_ms if baseline else None,
                "geometry.py",
            ),
            compute_stage(
                "Stage B: Projection Math Engine",
                "projection_engine.py",
                current.stage_b_projection_ms,
                baseline.stage_b_projection_ms if baseline else None,
                "projection_engine.py",
            ),
            compute_stage(
                "Stage C: Normalization & Centering",
                "video_writer.py",
                current.stage_c_normalization_ms,
                baseline.stage_c_normalization_ms if baseline else None,
                "video_writer.py",
            ),
            compute_stage(
                "Stage D: MP4 Video Encoding (I/O)",
                "video_writer.py",
                current.stage_d_encoding_ms,
                baseline.stage_d_encoding_ms if baseline else None,
                "video_writer.py",
            ),
        ]
        return stages

    @classmethod
    def format_profiling_report(
        cls, current: ProfileSummary, baseline: Optional[ProfileSummary], stages: List[StageMetrics]
    ) -> str:
        """Formats the terminal diagnostic report with stage breakdown and culprit alerts."""
        sep = "=" * 80
        lines = [
            sep,
            "            PIPELINE STAGE PROFILER & PERFORMANCE REGRESSION PINPOINTER",
            sep,
            f"[*] Current Commit:  {current.commit_hash} ('{current.commit_message}')",
        ]

        if baseline:
            lines.append(f"[*] Baseline Commit: {baseline.commit_hash} (Recorded: {baseline.timestamp})")
            fps_delta = ((current.overall_fps - baseline.overall_fps) / baseline.overall_fps) * 100.0
            fps_note = f"({fps_delta:+.1f}% vs baseline {baseline.overall_fps:.1f} FPS)"
        else:
            lines.append("[*] Baseline:        No baseline file found. (Run with --save-baseline to set reference)")
            fps_note = ""

        lines.extend([
            f"[*] Measured Throughput: {current.overall_fps:.1f} FPS {fps_note}",
            "-" * 80,
            f"{'Stage Name':<34} | {'Current':<9} | {'Baseline':<9} | {'Delta':<8} | {'Status'}",
            "-" * 80,
        ])

        culprits = []
        for s in stages:
            curr_str = f"{s.current_ms:6.3f}ms"
            base_str = f"{s.baseline_ms:6.3f}ms" if s.baseline_ms else "   N/A  "
            delta_str = f"{s.delta_pct:+6.1f}%" if s.delta_pct is not None else "  N/A  "
            lines.append(f"{s.name:<34} | {curr_str:<9} | {base_str:<9} | {delta_str:<8} | {s.status}")
            if s.culprit:
                culprits.append(s)

        lines.append("-" * 80)

        # Pinpointed Culprit Callout Box
        if culprits:
            lines.extend([
                "",
                "[!] CRITICAL ALERT: PERFORMANCE REGRESSION PINPOINTED!",
                f"    {len(culprits)} pipeline component(s) exhibited significant latency degradation (> {REGRESSION_THRESHOLD_PCT}%):",
            ])
            for c in culprits:
                lines.extend([
                    f"    -> Component: {c.component} ({c.name})",
                    f"       Diagnosis: {c.details}",
                ])
            lines.extend([
                "",
                "    Recommendation: Inspect the recent modifications in the flagged component above.",
                "                    Verify vectorized operations were not reverted to nested loops.",
                sep,
            ])
        else:
            lines.extend([
                "[+] All 4 pipeline stages are operating within healthy latency limits (Zero Regressions).",
                sep,
            ])

        return "\n".join(lines)

    @classmethod
    def record_history(
        cls,
        summary: ProfileSummary,
        status: str = "HEALTHY",
        csv_path: str = HISTORY_CSV_FILE,
        md_path: str = HISTORY_MD_FILE,
    ) -> None:
        """
        Appends performance run metrics to both CSV and Markdown history logs.
        """
        for p in (csv_path, md_path):
            dir_name = os.path.dirname(os.path.abspath(p))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        # 1. Append to CSV
        csv_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f_csv:
            if not csv_exists:
                f_csv.write(
                    "timestamp,commit_hash,commit_message,overall_fps,stage_a_geometry_ms,stage_b_projection_ms,stage_c_normalization_ms,stage_d_encoding_ms,status\n"
                )
            clean_msg = summary.commit_message.replace('"', '""').replace("\n", " ")
            f_csv.write(
                f"{summary.timestamp},{summary.commit_hash},\"{clean_msg}\",{summary.overall_fps:.1f},"
                f"{summary.stage_a_geometry_ms:.4f},{summary.stage_b_projection_ms:.4f},"
                f"{summary.stage_c_normalization_ms:.4f},{summary.stage_d_encoding_ms:.4f},{status}\n"
            )

        # 2. Append to Markdown Table
        md_exists = os.path.isfile(md_path)
        with open(md_path, "a", encoding="utf-8") as f_md:
            if not md_exists:
                f_md.write(
                    "# Continuous Performance Benchmarking History\n\n"
                    "Historical execution throughput and stage-by-stage latencies recorded over time.\n\n"
                    "| Timestamp | Commit | Message | FPS | Stage A | Stage B | Stage C | Stage D | Status |\n"
                    "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
                )
            short_msg = (clean_msg[:35] + "...") if len(clean_msg) > 35 else clean_msg
            f_md.write(
                f"| {summary.timestamp} | `{summary.commit_hash}` | {short_msg} | **{summary.overall_fps:.1f}** | "
                f"{summary.stage_a_geometry_ms:.3f}ms | {summary.stage_b_projection_ms:.3f}ms | "
                f"{summary.stage_c_normalization_ms:.3f}ms | {summary.stage_d_encoding_ms:.3f}ms | `{status}` |\n"
            )

    @classmethod
    def get_history(cls, csv_path: str = HISTORY_CSV_FILE, limit: int = 10) -> List[Dict[str, str]]:
        """Reads the last N historical records from the CSV log."""
        if not os.path.isfile(csv_path):
            return []
        import csv
        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records[-limit:]

    @classmethod
    def format_history_table(cls, csv_path: str = HISTORY_CSV_FILE, limit: int = 10) -> str:
        """Renders an ASCII table of the most recent performance runs for the terminal."""
        records = cls.get_history(csv_path, limit=limit)
        if not records:
            return "No performance history logged yet. Run python run_all_tests.py to record runs."

        sep = "=" * 90
        lines = [
            sep,
            "                   CONTINUOUS PERFORMANCE TELEMETRY HISTORY (LAST RUNS)",
            sep,
            f"{'Timestamp':<19} | {'Commit':<8} | {'Overall FPS':<12} | {'Stage B (Math)':<14} | {'Status'}",
            "-" * 90,
        ]
        for r in records:
            fps_val = f"{float(r['overall_fps']):.1f} FPS"
            b_val = f"{float(r['stage_b_projection_ms']):.3f}ms"
            lines.append(
                f"{r['timestamp']:<19} | {r['commit_hash']:<8} | {fps_val:<12} | {b_val:<14} | {r['status']}"
            )
        lines.append(sep)
        return "\n".join(lines)


if __name__ == "__main__":
    # Standalone execution for testing
    summary = PerformanceProfiler.profile_all_stages()
    baseline = PerformanceProfiler.load_baseline()
    stages = PerformanceProfiler.analyze_regression(summary, baseline)
    print(PerformanceProfiler.format_profiling_report(summary, baseline, stages))
