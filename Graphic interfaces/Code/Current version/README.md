# Scarlett Hexapod - 3D Interface Prototype

This is the first prototype of a Python desktop interface for importing and positioning 3D parts of the hexapod.

## Features in this first version

- Empty 3D scene with grid, floor and X/Y/Z axes
- Import of STL files
- Optional STEP/STP import if `cadquery` is installed
- Object tree for selecting imported parts
- Precise translation and rotation controls
- Prusa-like STL display mode with anti-aliasing and sharp edge preservation
- Save/load of a scene layout as JSON
- Placeholder for future alignment/constraint tools

## Install

From this folder:

```powershell
python -m venv scarlett
.\scarlett\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already created the environment and named it `scarlett`, just activate it before installing or running:

```powershell
enable scarlett
pip install -r requirements.txt
```

## Optional STEP/STP Support

STL import works with the main requirements. STEP/STP import needs the optional CAD package `cadquery`.

With the environment activated, install it with:

```powershell
pip install -r requirements-step.txt
```

or directly:

```powershell
pip install cadquery
```

If `cadquery` is difficult to install on your machine, keep the workflow STL-only for now and export high-resolution STL files from your CAD software.

## STL Quality Note

STL files are already triangulated meshes. The app now preserves the original triangles instead of subdividing them, then improves the display with clean normals, sharp-edge handling, lighting and anti-aliasing, closer to the way slicers display mechanical parts.

Use `Sharp CAD view` for normal work. Use `Smooth preview` only if you want a softer visual rendering, and `Facets debug` when you want to inspect the actual triangles.

If a STL was exported with too few triangles, the app cannot recover the original CAD curves perfectly. For best results, export STL files from your CAD software with a fine chord tolerance / deviation and a fine angular tolerance.

## Run

```powershell
python hexapod_3d_interface.py
```

## Next development steps

- Add real alignment constraints: point-to-point, axis-to-axis, plane-to-plane
- Add local pivots for mechanical joints
- Add parent/child hierarchy for body, coxa, femur and tibia parts
- Add joint angle limits
- Later: map joint angles to servo commands

