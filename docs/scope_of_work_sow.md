# Scope of Work (SOW) - 3D NRRD Plane Projection

## 1. Objectives
* **Context:** Academic assignment (Ex1 - Introduction to Software Engineering).
* **Goal:** Design and implement a robust, runtime-efficient system that loads a 3D volumetric NRRD file, computes 2D plane projections based on given coordinate sequences, and exports the rendered frames into an MP4 video file at $> 30$ FPS.

## 2. Deliverables
* **Documentation:**
  * `README.md` (problem definition, scope, setup, and usage).
  * `SOW.md` (Scope of Work).
  * `SRS.md` (Software Requirements Specification).
  * `Algorithm_Detail.md` (Detailed mathematical and algorithmic specification).
  * Testing strategy and experimental results report.
* **Software & Data:**
  * Source code implementing the 3D-to-2D projection engine and CLI runner (`Ex0.py`).
  * Sample NRRD volumetric dataset and trajectory CSV files.
  * Output demonstration video (`file.mp4`).

## 3. Milestones
* **M1 - Design & Specification:** Repository setup, `SOW.md`, `SRS.md`, and `Algorithm_Detail.md` completion.
* **M2 - Test & Data Prep:** NRRD dataset selection, CSV coordinate generation, and test plan definition.
* **M3 - Core Implementation:** NRRD/CSV parsers, math projection engine, and MP4 video encoder integration.
* **M4 - Optimization & Verification:** Profiling and vectorization to achieve $> 30$ FPS (target $\ge 60$ FPS average), final demo run, and experiment documentation.
