# Software Requirements Specification (SRS)

## 1. Introduction
* **Purpose:** Specify the functional and non-functional requirements for the 3D-to-2D Volumetric Plane Projection and Video Generation system.
* **Scope:** A CLI-driven application that ingests a 3D matrix (`.nrrd`) and a list of plane coordinates (`.csv`), calculates 2D projections for each plane, and outputs a compiled `.mp4` video.
* **Definitions:**
  * $M$: 3D raster data matrix of dimensions $sx \times sy \times sz$.
  * $p$: Projection plane defined by $3$ points (isosceles triangle) or $4$ points (isosceles trapezoid).
  * $E$: Output 2D projected image matrix (frame).
  * $\epsilon$: Floating-point geometric tolerance, defined as $\epsilon = 0.001$.

## 2. Overall Description
* **Product Perspective:** Standalone command-line batch processing tool inspired by 3D Slicer oblique slicing capabilities.
* **User Classes:** Software engineering evaluators, researchers, and developers running automated projection scripts.
* **Operating Environment:** Cross-platform OS (Linux, macOS, Windows) with Python 3.x (or Java/C/C++) and standard numerical/video libraries (e.g., NumPy, pynrrd, OpenCV/FFmpeg).
* **Constraints:**
  * Matrix dimensions bounded by $1 \le sx, sy, sz \le 1024$ ($2^{10}$).
  * Voxel values restricted to $8$, $16$, or $32$-bit unsigned integers.
* **Assumptions:** Input files exist at specified paths and follow standard NRRD and CSV formatting.

## 3. Specific Requirements

### 3.1 External Interfaces
* **CLI Interface:** `python Ex0.py <file.nrrd> <file.csv> <file.mp4>`

### 3.2 Functional Requirements
* **REQ-01 (NRRD Ingestion):** The system shall load a 3D matrix $M$ of size $sx \times sy \times sz$ supporting $8$, $16$, and $32$-bit unsigned integers up to $1024^3$.
* **REQ-02 (CSV Parsing):** The system shall parse a sequence of plane definitions from `<file.csv>`, supporting both $3$-point $(a, b, c)$ and $4$-point $(a_1, a_2, b, c)$ rows.
* **REQ-03 (Plane Validation):** The system shall validate plane geometry using $\epsilon = 0.001$:
  * For $3$ points: $|\text{dist}(a,b) - \text{dist}(a,c)| < \epsilon$.
  * For $4$ points: $|\text{dist}(a_1,b) - \text{dist}(a_2,c)| < \epsilon$, $\text{dist}(a_1,a_2) \le \text{dist}(b,c)$, and vectors $\vec{a_1b}, \vec{a_2c}$ are coplanar.
* **REQ-04 (2D Plane Projection):** For each valid row in the CSV, the system shall compute the 2D projection matrix $E$ of size $D_{ab} \times D_{bc}$ by sampling $M$ along the plane.
* **REQ-05 (Boundary Handling):** The system shall assign a value of $v = 0$ to any sampled 3D coordinate $p$ that falls outside the bounds $[0, sx) \times [0, sy) \times [0, sz)$.
* **REQ-06 (Video Export):** The system shall encode all computed 2D matrices $E$ in sequential order into `<file.mp4>`.

### 3.3 Non-Functional Requirements
* **NFR-01 (Minimum Frame Rate):** The projection computation shall process frames at a strict minimum rate of $\ge 30$ FPS.
* **NFR-02 (Average Frame Rate):** The system shall maintain an average processing throughput of $\ge 60$ FPS across standard test runs (and $\ge 100$ FPS on a $500^3$ byte matrix).
* **NFR-03 (Scalability):** The system shall process volumetric matrices up to $1024 \times 1024 \times 1024$ without memory overflow or crashes.
* **NFR-04 (Stability):** Invalid coordinates or out-of-bounds slices shall be handled gracefully without terminating execution.


......
## 4. Appendices

### 4.1 Use Case: UC-01 Generate Projection Video
* **Actor:** User / Automated Grading Script.
* **Preconditions:** Valid `.nrrd` and `.csv` files are accessible on disk.
* **Main Flow:**
  1. Actor invokes `python Ex0.py file.nrrd file.csv file.mp4`.
  2. System loads `.nrrd` data into 3D matrix $M$.
  3. System reads plane coordinate rows from `file.csv`.
  4. System projects each plane onto a 2D frame $E$ and appends it to the video stream.
  5. System finalizes and saves `file.mp4`.
* **Alternative Flow (3a - Invalid Plane Geometry):** System logs a warning and skips the malformed frame or raises a descriptive validation error.
* **Postconditions:** `file.mp4` is saved on disk containing one frame per valid CSV row.

### 4.2 Traceability Matrix
| Requirement ID | Use Case | Deliverable / Component | Verification Method |
| :--- | :--- | :--- | :--- |
| **REQ-01, REQ-02** | UC-01 | I/O Parser Module | Unit Test (File Loading) |
| **REQ-03, REQ-04, REQ-05** | UC-01 | Projection Math Engine | Unit & Geometric Accuracy Test |
| **REQ-06** | UC-01 | Video Encoder Module | System Demo Test (`Ex0.py`) |
| **NFR-01, NFR-02, NFR-03** | UC-01 | Core Engine | Benchmark Profiler ($\text{FPS} \ge 60$) |