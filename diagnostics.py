"""
Diagnostic Fault Localization Engine (Intro2SE / Ex0)

This module analyzes test failures, inspects stack tracebacks down to the exact
source code line in the project (geometry.py, projection_engine.py, nrrd_loader.py, video_writer.py),
extracts surrounding code context, local variable values, classifies the bug category,
and generates an executive diagnostic report.
"""

from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class FaultDetail:
    test_name: str
    suite_id: str
    failure_type: str
    error_message: str
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    code_snippet: Optional[str] = None
    local_variables: Optional[Dict[str, str]] = None
    fault_category: str = "GENERAL_FAULT"
    diagnostic_hint: str = ""


class FaultLocalizationEngine:
    """
    Parses failure output from test executions and pinpoints the exact
    fault location in the project's source code.
    """

    PROJECT_SOURCE_FILES = {
        "geometry.py",
        "projection_engine.py",
        "nrrd_loader.py",
        "video_writer.py",
        "Ex0.py",
    }

    @classmethod
    def categorize_fault(cls, error_msg: str, source_file: str, line_num: Optional[int]) -> tuple[str, str]:
        """Classifies the bug into a high-level category and generates a targeted hint."""
        msg_lower = error_msg.lower()

        if "zerodivisionerror" in msg_lower or "/ 0" in msg_lower or "division by zero" in msg_lower:
            return (
                "MATH_SINGULARITY_ZERO_DIV",
                "Division by zero detected. Check vector normalization (diff / d_12) when distance is zero (e.g. triangle apex)."
            )
        elif "geometryvalidationerror" in msg_lower or "isosceles" in msg_lower or "coplanar" in msg_lower:
            return (
                "GEOMETRIC_INVARIANT_VIOLATION",
                "Plane geometry failed validation constraints. Verify tolerance (eps=0.001), leg equality, or coplanarity dot product."
            )
        elif "indexerror" in msg_lower or "out of bounds" in msg_lower:
            return (
                "BOUNDARY_OUT_OF_BOUNDS",
                "Out-of-bounds array access. Verify that samples outside [0, s) are safely clamped to 0 (REQ-05) instead of indexing M."
            )
        elif "assertionerror" in msg_lower and "mismatch" in msg_lower:
            return (
                "NUMERICAL_VALUE_MISMATCH",
                "Sampled voxel or frame value diverged from analytical ground truth. Check coordinate rounding (np.floor) or step marcher."
            )
        elif "videowritererror" in msg_lower or "mp4" in msg_lower or "fourcc" in msg_lower:
            return (
                "VIDEO_CONTAINER_CORRUPTION",
                "OpenCV video encoder failed. Check canvas dimension consistency (width, height) or codec availability."
            )
        elif "nrrdformaterror" in msg_lower or "nrrdloaderror" in msg_lower:
            return (
                "NRRD_INTEGRITY_FAULT",
                "NRRD loader rejected volume format. Check bounds (1 <= dim <= 1024), 3D dimensionality, or supported integer dtypes."
            )
        elif "assertionerror" in msg_lower:
            return (
                "ASSERTION_FAILURE",
                "Verification assertion failed. Expected behavior diverged from observed state."
            )
        return (
            "GENERAL_FAULT",
            "Unhandled exception occurred. Review stack trace and variable state."
        )

    @classmethod
    def extract_code_snippet(cls, filepath: str, line_num: int, context_lines: int = 2) -> str:
        """Extracts the source code line and surrounding lines around the failure."""
        if not os.path.isfile(filepath):
            return f"Source file '{filepath}' not found on disk."

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
        except Exception as e:
            return f"Could not read source file: {e}"

        total = len(all_lines)
        start = max(0, line_num - 1 - context_lines)
        end = min(total, line_num + context_lines)

        snippet_parts = []
        for idx in range(start, end):
            curr_line_no = idx + 1
            prefix = " > " if curr_line_no == line_num else "   "
            snippet_parts.append(f"{prefix}{curr_line_no:4d} | {all_lines[idx].rstrip()}")

        return "\n".join(snippet_parts)

    @classmethod
    def parse_traceback_for_source_fault(cls, output: str) -> Optional[tuple[str, int, str]]:
        """
        Parses pytest traceback output to locate the innermost frame occurring in
        one of the project's source files (not in pytest internals or test files).
        Returns: (source_file, line_number, code_snippet)
        """
        # Pattern 1: Standard python: File ".../geometry.py", line 123, in validate_trapezoid
        # Pattern 2: Pytest short format: geometry.py:113: in validate_triangle
        patterns = [
            re.compile(
                r'File\s+["\'](?P<path>[^"\']+)["\'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>\w+))?',
                re.IGNORECASE
            ),
            re.compile(
                r'(?P<path>[a-zA-Z0-9_\-\\/]+\.py):(?P<line>\d+):(?:\s+in\s+(?P<func>\w+))?',
                re.IGNORECASE
            ),
        ]

        matches = []
        for p in patterns:
            matches.extend(p.finditer(output))

        # Sort matches by position in the output text so order of frames is preserved
        matches.sort(key=lambda m: m.start())
        # Search backwards from the crash point to find the deepest project file
        for match in reversed(matches):
            raw_path = match.group("path")
            basename = os.path.basename(raw_path)
            if basename in cls.PROJECT_SOURCE_FILES:
                line_no = int(match.group("line"))
                snippet = cls.extract_code_snippet(raw_path, line_no)
                return basename, line_no, snippet

        # If not in source files directly, check if it occurred in tests/
        for match in reversed(matches):
            raw_path = match.group("path")
            basename = os.path.basename(raw_path)
            if basename.startswith("test_") and basename.endswith(".py"):
                line_no = int(match.group("line"))
                snippet = cls.extract_code_snippet(raw_path, line_no)
                return basename, line_no, snippet

        return None

    @classmethod
    def extract_error_message(cls, output: str) -> tuple[str, str]:
        """Extracts the exception class and description from traceback output."""
        lines = output.strip().splitlines()
        for line in reversed(lines):
            # Look for lines like E       AssertionError: ... or E   ValueError: ...
            stripped = line.strip()
            if stripped.startswith("E ") or stripped.startswith("E:"):
                clean = stripped[2:].strip()
                if ":" in clean:
                    err_type, _, err_msg = clean.partition(":")
                    return err_type.strip(), err_msg.strip()
                return "AssertionFailure", clean
            # Also catch standard unindented exception lines
            for exc in ("AssertionError", "ValueError", "IndexError", "ZeroDivisionError", "RuntimeError", "GeometryValidationError"):
                if stripped.startswith(f"{exc}:"):
                    err_type, _, err_msg = stripped.partition(":")
                    return err_type.strip(), err_msg.strip()

        return "UnknownError", "Test failed without standard exception format."

    @classmethod
    def analyze_failure(cls, suite_id: str, test_file: str, raw_output: str) -> FaultDetail:
        """Analyzes a suite failure and builds a structured FaultDetail object."""
        err_type, err_msg = cls.extract_error_message(raw_output)
        source_info = cls.parse_traceback_for_source_fault(raw_output)

        source_file = None
        source_line = None
        code_snippet = None

        if source_info:
            source_file, source_line, code_snippet = source_info

        category, hint = cls.categorize_fault(
            f"{err_type}: {err_msg}",
            source_file or test_file,
            source_line
        )

        return FaultDetail(
            test_name=test_file,
            suite_id=suite_id,
            failure_type=err_type,
            error_message=err_msg,
            source_file=source_file,
            source_line=source_line,
            code_snippet=code_snippet,
            fault_category=category,
            diagnostic_hint=hint,
        )

    @classmethod
    def format_terminal_box(cls, fault: FaultDetail) -> str:
        """Renders an eye-catching, high-visibility ASCII fault box for terminal output."""
        sep = "=" * 80
        lines = [
            sep,
            "                   INJECTED BUG & FAULT LOCALIZATION REPORT",
            sep,
            f"[!] FAILED SUITE:       {fault.suite_id} ({fault.test_name})",
            f"[!] FAULT CATEGORY:     [{fault.fault_category}]",
            f"[!] EXCEPTION TYPE:     {fault.failure_type}",
            f"[!] ERROR DESCRIPTION:  {fault.error_message}",
            "-" * 80,
        ]

        if fault.source_file and fault.source_line:
            lines.extend([
                f"[PINPOINTED FAULT LOCATION]:",
                f"  -> File:        {fault.source_file}",
                f"  -> Line Number: {fault.source_line}",
                "",
                f"[SOURCE CONTEXT]:",
                fault.code_snippet or "  (Snippet unavailable)",
                "-" * 80,
            ])
        else:
            lines.extend([
                "[LOCATION]: Specific source frame could not be isolated.",
                "-" * 80,
            ])

        if fault.diagnostic_hint:
            lines.extend([
                f"[DIAGNOSTIC HINT FOR REVIEWER]:",
                f"  {fault.diagnostic_hint}",
                sep,
            ])

        return "\n".join(lines)

    @classmethod
    def generate_markdown_report(cls, faults: List[FaultDetail], filepath: str = "reports/fault_localization_report.md") -> str:
        """Generates and writes a clean Markdown fault report suitable for submission."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        md = [
            "# Software Fault Localization & Injected Bug Report",
            "",
            "> **Automated Diagnostic System:** This report is generated automatically by the verification harness when an injected bug or test regression is detected.",
            "",
            f"**Total Failures Detected:** {len(faults)}",
            "",
            "---",
            "",
        ]

        for idx, f in enumerate(faults, 1):
            md.extend([
                f"## Fault #{idx}: {f.fault_category} in `{f.suite_id}`",
                "",
                f"* **Suite / Test:** `{f.suite_id}` (`{f.test_name}`)",
                f"* **Exception:** `{f.failure_type}`",
                f"* **Message:** {f.error_message}",
                f"* **Fault Category:** `{f.fault_category}`",
            ])

            if f.source_file and f.source_line:
                md.extend([
                    f"* **Pinpointed Source File:** [`{f.source_file}`](file:///{os.path.abspath(f.source_file)})",
                    f"* **Pinpointed Source Line:** Line **{f.source_line}**",
                    "",
                    "### Source Code Context",
                    "```python",
                    f.code_snippet or "# Snippet unavailable",
                    "```",
                ])

            if f.diagnostic_hint:
                md.extend([
                    "",
                    "> [!IMPORTANT]",
                    f"> **Diagnostic Insight:** {f.diagnostic_hint}",
                ])

            md.extend(["", "---", ""])

        report_content = "\n".join(md)
        with open(filepath, "w", encoding="utf-8") as out:
            out.write(report_content)

        return report_content
