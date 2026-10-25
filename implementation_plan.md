# Implementation & Verification Plan (Checklist)
## 3D NRRD Plane Projection & Video Generation (`Ex0`)

> **Approach & Methodology:**
> We will implement this project incrementally in very small, atomic steps. For every single step, we will:
> 1. Discuss the mathematical and algorithmic rationale.
> 2. Implement the code with detailed, line-by-line explanations.
> 3. Verify it immediately with automated unit/integration tests before proceeding to the next part.
> 4. Keep this checklist updated with progress (`[x]`).

---

### Phase 1: Project Architecture & Environment Setup
- [ ] **1.1 Directory Structure & Module Blueprint**
  - Define clear separation of concerns:
    - `geometry.py`: Geometric models, vector calculations, plane validation, discontinuity checks.
    - `nrrd_loader.py`: NRRD dataset loading, shape & data type validation, normalization.
    - `projection_engine.py`: 3D-to-2D scanline sampling engine (baseline & vectorized for $\ge 60$ FPS).
    - `video_writer.py`: Frame-to-MP4 video encoder using OpenCV.
    - `Ex0.py`: Main CLI entry point orchestrating the pipeline and measuring FPS.
    - `tests/`: STP test suites (Suites 1–7).

---

### Phase 2: Geometric Modeling & CSV Trajectory Parsing (Suites 1 & 7)
- [ ] **2.1 Geometric Data Structures & Tolerances**
  - Define `Point3D` / coordinate representations.
  - Define tolerance constant $\epsilon = 0.001$.
  - Unified internal representation for both 3-point (triangle) and 4-point (trapezoid) definitions.
- [ ] **2.2 Geometric Validation Engine (STP Suite 1)**
  - **3-Point Validation:** Check $|\text{dist}(a, b) - \text{dist}(a, c)| < \epsilon$. Set $a_1 = a, a_2 = a$.
  - **4-Point Validation:**
    - Equal legs: $|\text{dist}(a_1, b) - \text{dist}(a_2, c)| < \epsilon$.
    - Base ordering: $\text{dist}(a_1, a_2) \le \text{dist}(b, c)$.
    - Coplanarity: $|(\vec{a_1b} \times \vec{a_1c}) \cdot \vec{a_1a_2}| < \epsilon$.
- [ ] **2.3 CSV Reader & Row Parser**
  - Parse comma-separated coordinates supporting both 9-value rows $(a, b, c)$ and 12-value rows $(a_1, a_2, b, c)$.
  - Graceful handling of empty lines, whitespace, and descriptive error messages for malformed rows.
- [ ] **2.4 Trajectory Discontinuity & Warning System (STP Suite 7)**
  - Compute Euclidean displacement $\Delta d$ between corresponding control points of consecutive frames.
  - Adaptive threshold: $\Delta d_{\max} = 0.01 \times \max(sx, sy, sz)$.
  - Emit non-blocking `[WARN]` on jump without halting the program.
- [ ] **2.5 Automated Testing for Phase 2**
  - Create and run `tests/test_geometry.py` covering valid/invalid triangles, trapezoids, non-coplanar points, and trajectory warnings.

---

### Phase 3: NRRD Volume Loader & Data Integrity (Suite 3)
- [ ] **3.1 Volume Ingestion with `pynrrd`**
  - Read header and raw voxel data array $M$.
  - Support dimensions $1 \le sx, sy, sz \le 1024$.
  - Support unsigned 8, 16, and 32-bit integer formats (and standard volume representations).
- [ ] **3.2 Memory Layout & Safety Checks**
  - Ensure contiguous C-order memory layout (`np.ascontiguousarray`) for high-speed cache efficiency.
  - Graceful handling of corrupt or missing files.
- [ ] **3.3 Automated Testing for Phase 3**
  - Create and run `tests/test_nrrd_loader.py` with synthetic volumes and verify loading of `MRHead_1.nrrd`.

---

### Phase 4: 2D Plane Projection Engine (Suites 2 & 5)
- [ ] **4.1 Mathematical Step-by-Step Scanline Baseline**
  - Implement the exact scanline math from `algorithm_in_detail_edd.md`:
    - $D_{ab} = \|b - a_1\|_2$, $D_{bc} = \|c - b\|_2$.
    - Output dimensions: width $W = \lceil D_{bc} \rceil$, height $H = \lceil D_{ab} \rceil$.
    - Unit leg vectors: $\hat{v}_{ab} = (b - a_1)/D_{ab}$, $\hat{v}_{ac} = (c - a_2)/D_{ab}$.
    - March endpoints $p_1(y), p_2(y)$, horizontal span $d_{12}$, unit vector $\hat{dp}$, centering offset $x_{\text{offset}}$.
    - Nearest-neighbor sampling: round/floor coordinates to nearest voxel.
- [ ] **4.2 Boundary & Out-of-Bounds Clamping (STP Suite 5)**
  - Assign voxel value $v = 0$ for any sample $(p_x, p_y, p_z)$ outside $[0, sx) \times [0, sy) \times [0, sz)$.
- [ ] **4.3 High-Performance Vectorization ($\ge 60$ FPS Target - NFR-01/02)**
  - Vectorize scanline calculations using NumPy 2D meshgrids / broadcasting.
  - Eliminate slow Python loops per pixel to reach $> 100$ FPS throughput.
- [ ] **4.4 Mathematical Sanity & Precision Testing (STP Suite 2)**
  - Create synthetic test volume where voxel value = spatial coordinate ($M[x, y, z] = x$).
  - Sample axial and diagonal planes, asserting zero deviation from analytical ground truth.
  - Create and run `tests/test_projection.py`.

---

### Phase 5: MP4 Video Encoder Module (Suite 6)
- [ ] **5.1 Frame Formatting & Normalization**
  - Map 16-bit / integer voxel values to 8-bit grayscale ($0\dots 255$) or BGR frames suitable for video codecs.
  - Canvas size standardization: ensure consistent frame dimensions across varying plane sizes (padding/resizing if base dimensions vary, or letterboxing).
- [ ] **5.2 OpenCV VideoWriter Integration**
  - Initialize `cv2.VideoWriter` with `'mp4v'` codec, target FPS (e.g. 30 FPS).
  - Sequentially append frames and finalize video container cleanly.
- [ ] **5.3 Automated Testing for Phase 5**
  - Create and run `tests/test_video_writer.py`: verify `.mp4` container integrity, frame count, and dimensions with OpenCV/FFprobe.

---

### Phase 6: CLI Runner & Pipeline Integration
- [ ] **6.1 CLI Argument Parser (`Ex0.py`)**
  - Implement interface: `python Ex0.py <file.nrrd> <file.csv> <file.mp4>`.
  - Validate file existence and argument syntax.
- [ ] **6.2 End-to-End Orchestration**
  - Load NRRD -> Parse CSV -> Validate Geometry -> Project Planes -> Write MP4.
- [ ] **6.3 High-Resolution Performance Profiler**
  - Measure elapsed time for projection and encoding (excluding initial file load).
  - Output total processing time and average FPS: $\text{FPS}_{\text{avg}} = N / \Delta t$.
  - Verify compliance with $\text{FPS} \ge 30$ (minimum) and $\ge 60$ (target).

---

### Phase 7: Comprehensive STP Verification (Suites 1–7)
- [ ] **7.1 Unified Test Suite Runner (`run_all_tests.py`)**
  - Run all 7 test suites defined in `software_test_plan_stp.md`:
    - Suite 1: Geometric Validation (Triangles & Trapezoids).
    - Suite 2: Mathematical Sanity & Reference Comparison.
    - Suite 3: NRRD File Integrity & Bounds Validation.
    - Suite 4: Performance & High-FPS Benchmarking ($500^3$ benchmark).
    - Suite 5: Boundary & Out-of-Bounds Sampling Safety.
    - Suite 6: Output Video & MP4 Encoding Integrity.
    - Suite 7: Visual Continuity & Non-Blocking Step Warning.

---

### Phase 8: Real Dataset Demo & Final Verification
- [ ] **8.1 Trajectory Generation for `MRHead_1.nrrd`**
  - Generate realistic CSV trajectory (e.g., axial sweep, oblique rotation through the brain scan).
- [ ] **8.2 Execution & Video Generation**
  - Run `python Ex0.py MRHead_1.nrrd trajectory.csv output.mp4`.
  - Confirm video playback, visual quality, and high FPS throughput.
- [ ] **8.3 Final Documentation & Code Review**
  - Review all lines of code, confirm all requirements met.
