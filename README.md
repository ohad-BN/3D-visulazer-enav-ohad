# 3D NRRD Plane Projection & Video Generator

A high-performance volumetric projection engine that slices 3D NRRD raster volumes along arbitrary planar trajectories and renders them into an MP4 video at over 300 FPS.

---

## 1. Quick Start

### Prerequisites
* Python 3.10+ (tested on Python 3.12)
* FFmpeg / OpenCV-compatible environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ohadbn/Ex0.git
   cd Ex0
   ```

2. Create and activate a virtual environment:
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Run
Execute a full end-to-end projection on the bundled MRI head scan:
```bash
python Ex0.py MRHead_1.nrrd CSV/head.csv output.mp4
```

---

## 2. Usage Examples

### CLI Command Syntax
```bash
python Ex0.py <file.nrrd> <file.csv> <file.mp4>
```

| Argument | Description |
| :--- | :--- |
| `<file.nrrd>` | Path to the 3D NRRD volumetric file (auto-searches inside `NRRD/` if not found in current directory). |
| `<file.csv>` | Path to the trajectory CSV defining slicing planes (auto-searches inside `CSV/`). |
| `<file.mp4>` | Destination path for the generated MP4 video file. |

### Examples

* **Basic Execution with automatic path resolution:**
  ```bash
  python Ex0.py MRHead_1.nrrd head.csv output.mp4
  ```

* **Full Relative Path Execution:**
  ```bash
  python Ex0.py NRRD/croissant.nrrd CSV/croissant.csv croissant_proj.mp4
  ```

* **Display Help:**
  ```bash
  python Ex0.py --help
  ```

### Trajectory CSV Format
Each row in the CSV file represents one video frame slice, formatted as floating-point Cartesian coordinates in volume voxel space:
* **3-Point Definition (Isosceles Triangle):**
  $$a_x, a_y, a_z, \; b_x, b_y, b_z, \; c_x, c_y, c_z$$
  Constraint: $|\text{dist}(a, b) - \text{dist}(a, c)| < 0.001$.
* **4-Point Definition (Isosceles Trapezoid):**
  $$a_{1x}, a_{1y}, a_{1z}, \; a_{2x}, a_{2y}, a_{2z}, \; b_x, b_y, b_z, \; c_x, c_y, c_z$$
  Constraints: $|\text{dist}(a_1, b) - \text{dist}(a_2, c)| < 0.001$, $\text{dist}(a_1, a_2) \le \text{dist}(b, c)$, and coplanarity via scalar triple product.
* Optional prefix count indicators (`3, ...` or `4, ...`), whitespace formatting, and comment lines starting with `#` are automatically supported.

---

## 3. Configuration & Environment

* **Supported Operating Systems:** Windows 10/11, Ubuntu 20.04+, macOS 12+
* **Core Libraries (`requirements.txt`):**
  * `numpy >= 1.24.0` (vectorized matrix computations and coordinate grids)
  * `pynrrd >= 1.0.0` (NRRD file reader and header parser)
  * `opencv-python >= 4.8.0` (video container encoding and frame standardization)
  * `pytest >= 7.4.0` (automated STP verification test runner)
* **System Boundaries & Constraints:**
  * Volume dimensions: $1 \le sx, sy, sz \le 1024$
  * Voxel data types: Unsigned and signed 8-bit, 16-bit, and 32-bit integers
  * Out-of-bounds sampling: Voxel coordinates outside matrix bounds are safely clamped to $0$ (black)
  * Performance Target: Minimum $\ge 30$ FPS; target $\ge 60$ FPS (achieved $> 300$ FPS)

---

## 4. How to Run the Tests

The test suite validates compliance with the Software Test Plan (STP Suites 1–7):

### 1. Unified STP Test Suite Runner
To execute all 7 STP test suites and generate an executive summary table:
```bash
python run_all_tests.py
```

### 2. Running Individual Suites with `pytest`
```bash
# Run all tests with verbose output
pytest -v

# Run 500^3 volumetric performance benchmark (STP Suite 4)
pytest tests/test_benchmark_500.py -s

# Run geometric validation tests (STP Suite 1)
pytest tests/test_geometry.py -v

# Run mathematical correctness and boundary safety (STP Suites 2 & 5)
pytest tests/test_projection.py -v

# Run NRRD integrity tests (STP Suite 3)
pytest tests/test_nrrd_loader.py -v

# Run MP4 video writer integrity tests (STP Suite 6)
pytest tests/test_video_writer.py -v
```

---

## 5. Project Structure

```text
Ex0/
├── README.md                   # Repository entry point and user documentation
├── Ex0.py                      # Main CLI entry point and pipeline orchestrator
├── geometry.py                 # Geometric models, vector math, CSV parser, continuity check
├── nrrd_loader.py              # NRRD volume ingestion, bounds validation, C-contiguous layout
├── projection_engine.py        # Baseline and vectorized 3D-to-2D scanline projection engines
├── video_writer.py             # OpenCV MP4 video encoder, normalization, canvas sizing
├── run_all_tests.py            # Unified runner executing all 7 STP suites
├── requirements.txt            # Python dependencies
│
├── docs/                       # Software Engineering Document Chain
│   ├── scope_of_work_sow.md                        # SOW: Objectives, deliverables, scope, milestones
│   ├── software_requirements_specification_srs.md  # SRS: Functional/non-functional requirements
│   ├── algorithm_in_detail_edd.md                  # EDD: Detailed mathematical & scanline formulation
│   ├── software_test_plan_stp.md                   # STP: Test strategies, criteria, suites 1-7
│   └── implementation_plan.md                      # Development checklist and verification tracking
│
├── tests/                      # Automated Test Suites (STP Suites 1-7)
│   ├── test_geometry.py        # Suite 1 (Geometry) & Suite 7 (Continuity warnings)
│   ├── test_projection.py      # Suite 2 (Analytical correctness) & Suite 5 (Boundaries)
│   ├── test_nrrd_loader.py     # Suite 3 (NRRD bounds, types, corruption safety)
│   ├── test_benchmark_500.py   # Suite 4 (500^3 high-throughput FPS benchmark)
│   ├── test_video_writer.py    # Suite 6 (MP4 container integrity and playability)
│   ├── test_cli.py             # End-to-end CLI arguments and exit code tests
│   └── test_environment.py     # Runtime environment and dependency checks
│
├── CSV/                        # Sample trajectory test banks
│   ├── head.csv                # 280-frame rotation trajectory for MRHead_1
│   ├── croissant.csv           # Slicing trajectory for croissant scan
│   ├── onion.csv               # Slicing trajectory for onion scan
│   ├── biscuit_cake.csv        # Slicing trajectory for biscuit cake scan
│   └── edge_cases.csv          # Degenerate, out-of-bounds, and malformed CSV rows
│
└── NRRD/                       # Sample 3D volumetric datasets
    ├── MRHead_1.nrrd           # 3D Slicer MRI head dataset (256x256x130, int16)
    ├── croissant.nrrd          # Volumetric scan
    ├── onion.nrrd              # Volumetric scan
    └── biscuit_cake.nrrd       # Volumetric scan
```

---

## 6. Software Engineering Documentation

This repository follows the formal Software Engineering document chain:
* **Scope of Work (SOW):** See [docs/scope_of_work_sow.md](docs/scope_of_work_sow.md)
* **Software Requirements Specification (SRS):** See [docs/software_requirements_specification_srs.md](docs/software_requirements_specification_srs.md)
* **Engineering Design Document (EDD):** See [docs/algorithm_in_detail_edd.md](docs/algorithm_in_detail_edd.md)
* **Software Test Plan (STP):** See [docs/software_test_plan_stp.md](docs/software_test_plan_stp.md)

---

## 7. How to Contribute

1. **Branching Strategy:**
   * Create a feature branch from `main`:
     ```bash
     git checkout -b feature/your-feature-name
     ```
2. **Docs-as-Code Rule:**
   * Any change to system behaviour, algorithm, or CLI interfaces must be updated in the corresponding documentation (`docs/`) within the same pull request.
3. **Verification Before Push:**
   * Ensure all automated tests pass with 100% success rate:
     ```bash
     python run_all_tests.py
     ```
4. **Submitting Changes:**
   * Open a pull request against `main` describing the changes and referencing the relevant requirement ID (`REQ-XX` / `NFR-XX`).

---

## 8. Authors & License

* **Authors:**
  * Ohad Ben Nun
  * Anav Shlomov
* **Course:** Introduction to Software Engineering / Object-Oriented Programming, Ariel University
* **Instructor:** Prof. Boaz Ben-Moshe (`benmo@g.ariel.ac.il`)
* **License:** Licensed under the [MIT License](LICENSE).
