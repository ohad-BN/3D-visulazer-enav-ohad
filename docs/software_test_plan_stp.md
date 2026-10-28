# Software Test Plan (STP) - 3D NRRD Plane Projection

## 1. Overview and Scope
* **Objective:** Establish the testing strategy and verification protocols for the 3D-to-2D volumetric projection and MP4 generation system (`Ex0.py`).
* **Test Scope:**
  * Geometric validation of plane inputs (triangle and trapezoid configurations).
  * Mathematical correctness of volumetric sampling against known reference models.
  * Integrity and constraints verification of input NRRD datasets.
  * Runtime performance benchmarks (targeting $\ge 30$ FPS minimum, $\ge 60$ FPS average).
  * Out-of-bounds sampling and numerical boundary safety.
  * Output video format encoding and codec compliance.
  * Visual trajectory continuity and non-blocking heuristic warnings.

## 2. Test Environment and Setup
* **Operating Systems:** Windows 11.
* **Runtimes & Compilers:** Python 3.10+ (NumPy, OpenCV-Python, pynrrd).
* **Hardware Profiles:**
  * Minimal: 4 Cores CPU, 4 GB RAM.
  * Benchmark Target: Intel Core i7, 8 GB RAM.
* **Test Datasets:**
  * Synthetic volumes: Uniform cubes, coordinate gradients ($10^3$, $100^3$, $500^3$).
  * Medical datasets: 3D Slicer MRI head scans (`MR-head.nrrd`).
  * Synthetic plane trajectories: Planar CSV test banks with both compliant and edge-case coordinates.

## 3. Core Testing Strategies (Test Suites 1–7)

### Suite 1: Geometric Validation (Isosceles Triangle & Trapezoid)
* **Goal:** Verify that plane coordinates parsed from the CSV strictly comply with project geometric rules.
* **Test Procedures:**
  * **3-Point Test:** For points $(a, b, c)$, verify $|\text{dist}(a,b) - \text{dist}(a,c)| < 0.001$.
  * **4-Point Test:** For points $(a_1, a_2, b, c)$, verify $|\text{dist}(a_1,b) - \text{dist}(a_2,c)| < 0.001$, $\text{dist}(a_1, a_2) \le \text{dist}(b, c)$, and coplanarity via scalar triple product $|(\vec{a_1b} \times \vec{a_1c}) \cdot \vec{a_1a_2}| < 0.001$.
* **Pass Criteria:** Geometry parser accepts all mathematically valid configurations and rejects degenerate/non-coplanar rows.

### Suite 2: Mathematical Sanity & Reference Comparison (Algorithm Correctness)
* **Goal:** Verify projection math against deterministic ground-truth volumes before processing complex MRI scans.
* **Test Procedures:**
  * Construct a synthetic volume where voxel values equal their spatial indices ($M[x, y, z] = x$).
  * Define an axial plane ($z = k$) where the expected 2D image output $E$ is analytically predictable.
  * Sample the plane and perform pixel-by-pixel assertions ($E[x, y] == E_{\text{expected}}[x, y]$).
* **Pass Criteria:** Zero numerical deviation between projected frame and theoretical values across all valid internal pixels.

### Suite 3: Input NRRD File Integrity & Bounds Validation
* **Goal:** Prevent runtime segmentation faults and memory corruption from malformed datasets.
* **Test Procedures:**
  * Verify parsing of valid NRRD headers (data dimensions $sx, sy, sz \le 1024$, data type unsigned integer 8/16/32-bit).
  * Ingest corrupted, zero-byte, or unsupported dimension formats and assert safe exception raising.
  * Verify clean memory allocation and deallocation under maximum volume boundaries ($1024^3$).
* **Pass Criteria:** System loads valid NRRD files into memory correctly and terminates gracefully with clear error messages on corrupt files.

### Suite 4: Performance & High-FPS Benchmarking
* **Goal:** Verify compliance with strict non-functional throughput requirements.
* **Test Procedures:**
  * Benchmark frame computation throughput over a sequence of 100 consecutive frames on a $500^3$ matrix.
  * Measure total elapsed execution time excluding initial one-time NRRD disk load.
  * Compute average frames per second: $\text{FPS}_{\text{avg}} = \frac{N_{\text{frames}}}{\Delta t_{\text{compute}}}$.
* **Pass Criteria:**
  * Absolute minimum frame rate: $\ge 30$ FPS.
  * Average benchmark throughput: $\ge 60$ FPS (target $\ge 100$ FPS on modern architectures).

### Suite 5: Boundary & Out-of-Bounds Sampling Safety
* **Goal:** Ensure system stability when planes intersect or fully lie outside volume boundaries.
* **Test Procedures:**
  * Feed plane definitions where coordinates extend past matrix boundaries (e.g., $x < 0$ or $x \ge sx$).
  * Test fully exterior planes where no part of the plane intersects the bounding box of $M$.
* **Pass Criteria:**
  * Pixels corresponding to out-of-bounds coordinates evaluate strictly to $0$ (black).
  * Zero out-of-bounds index exceptions, crashes, or memory leaks during processing.

### Suite 6: Output Video & MP4 Encoding Integrity
* **Goal:** Validate that output `.mp4` files are fully compliant and playable.
* **Test Procedures:**
  * Verify that an `.mp4` file is created at the exact output path specified by CLI arguments.
  * Inspect video container headers using FFprobe / OpenCV: confirm container type, dimensions ($D_{bc} \times D_{ab}$), and frame count match the CSV row count.
  * Verify video playback in standard media players (VLC, QuickTime) without playback stutter or visual artifacts.
* **Pass Criteria:** Output file is a valid, uncorrupted MP4 container matching expected frame count and dimensions.

### Suite 7: Visual Continuity & Non-Blocking Step Warning
* **Goal:** Ensure coherent video playback while maintaining strict non-blocking CLI compatibility for automated grading.
* **Test Procedures:**
  * Calculate Euclidean displacement $\Delta d = \|p_i - p_{i+1}\|_2$ for corresponding control points between consecutive CSV rows.
  * **Smooth Trajectory Case:** Verify that smooth sequences ($\Delta d \le \text{threshold}$) execute without console warnings.
  * **Discontinuous Jump Case:** Feed erratic or distant plane coordinates ($\Delta d > \text{threshold}$) representing abrupt spatial cuts.
* **Pass Criteria:**
  * Execution does **not** halt and does **not** throw fatal exceptions on arbitrary test inputs.
  * For large spatial jumps, the system emits a non-blocking diagnostic log message:
    `[WARN] Large spatial delta detected between frames i and i+1 (> threshold). Video may exhibit visual stutter.`
  * All valid frames are still rendered sequentially to the final MP4 file.

  * **Test Strategy & Threshold Definition:**
    * **Adaptive Threshold:** $\Delta d_{\max} = 0.01 \times \max(sx, sy, sz)$ voxels (e.g., $\le 5.12$ voxels for $512^3$, $\le 10.24$ voxels for $1024^3$).
    * **Pass Criteria (Smooth Sequence):** Verify that input trajectories where $\Delta d \le \Delta d_{\max}$ execute cleanly without console warnings.
    * **Pass Criteria (Discontinuous Sequence):** Verify that inputs with $\Delta d > \Delta d_{\max}$ trigger a non-blocking `[WARN]` log while continuing full frame processing and video generation.

## 4. Entry and Exit Criteria
* **Entry Criteria:**
  * Core modules implemented (NRRD Reader, CSV Parser, Projection Engine, MP4 Writer).
  * Synthetic unit test datasets and sample `MR-head.nrrd` accessible in local repository.
* **Exit Criteria:**
  * 100% pass rate on Suites 1, 2, 3, 5, and 6.
  * Performance Suite 4 satisfies $\ge 30$ FPS minimum and $\ge 60$ FPS average on reference hardware.
  * Suite 7 emits informative warnings for discontinuous frames without blocking process completion.