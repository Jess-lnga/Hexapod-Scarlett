# Scarlett Hexapod - 3D Interface Prototype

This is the first prototype of a Python desktop interface for importing and positioning 3D parts of the hexapod.

## Features in this first version

- Empty 3D scene with grid, floor and X/Y/Z axes
- Import of STL files
- Optional STEP/STP import if `cadquery` is installed
- Object tree for selecting imported parts
- Precise translation and rotation controls
- Save/load of a scene layout as JSON
- Placeholder for future alignment/constraint tools

## Install

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For STEP/STP support, try:

```powershell
pip install cadquery
```

If that fails, keep the first version STL-only for now. STEP support often needs heavier CAD dependencies.

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
