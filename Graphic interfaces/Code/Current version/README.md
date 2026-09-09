# Scarlett Hexapod - 3D Interface Prototype

This is the first prototype of a Python desktop interface for importing and positioning 3D parts of the hexapod.

## Features in this first version

- Empty 3D scene with grid, floor and X/Y/Z axes
- Import of STL files
- Optional STEP/STP import if `cadquery` is installed
- Scene tree for selecting imported parts, with equivalent direct selection by clicking objects in the 3D viewport and compact delete buttons for model removal
- Constraints tree under the scene tree
- `Add constraint` floating dialog for absolute, relative, object-to-axis, object-to-plane and parallel-planes constraints, with live selection in the 3D viewport
- Precise translation and rotation controls
- Direct Coincidence mode in a floating assistant window: hover full detected surfaces or inferred hole axes in yellow, select them in orange, then align them with rotation
- Remove selected model button
- Absolute fixed constraint for parts that must never move
- Relative rigid groups for parts that must move as one assembly
- Related constrained parts highlighted in orange when one part is selected
- Scrollable right-side panel with independently resizable scene tree, constraints tree and controls sections, plus wider spacing from the 3D viewport splitter
- Delete constraints from the constraints tree with the compact delete button
- Prusa-like STL display mode with anti-aliasing and sharp edge preservation
- Save/load of a scene layout as JSON, with backward compatibility for older layout files

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

## Mechanical Constraints

The app has a dedicated constraints tree below the scene tree.

Add a constraint:

- Click `Add constraint` under the constraints tree.
- A floating dialog opens; keep it open while selecting in the 3D viewport.
- Choose a type: `Absolute fixity`, `Relative rigid group`, `Object to axis`, `Object to plane`, `Parallel planes` or `Other`.
- The dialog selection list updates live as you click objects, planes or axes in the viewport.
- The dialog `OK` button stays disabled until the current selection is valid for the selected constraint type.
- Whole selected objects turn blue; hovered selectable objects glow light blue; hovered planes or axes are highlighted in yellow.
- Click `OK` in the dialog to create the constraint.
- Click `Cancel` in the dialog to leave picking mode without creating anything.

Absolute fixed:

- Absolute constraints can include one or more objects selected directly in the 3D viewport.
- These objects can no longer be moved by manual transforms, origin reset, surface coincidence or axis coincidence.

Relative rigid groups:

- Relative constraints can include one or more objects selected directly in the 3D viewport.
- Objects in the same relative constraint share a rigid group name.
- When one non-fixed object in that group is translated or rotated, the same transformation is propagated to the other non-fixed objects in the group.
- When a selected object is constrained with other objects, the related objects are highlighted in orange.

Object-to-axis / object-to-plane:

- These constraints can be created and saved from the constraints panel.
- Object-to-axis and object-to-plane constraints are selected directly in the viewport: first click the constrained object, then click the target axis or plane. For object-to-axis, clicking another non-axis object replaces the constrained object until an axis is selected.
- For now, the actual geometric positioning is still done with the direct `Coincidence` tool; the saved feature data prepares the next persistent constraint solver layer.

Parallel planes:

- Select `Parallel planes` in the constraint dialog.
- Click two planar surfaces on two different objects.
- The two planes are kept parallel when one of the involved objects moves.
- Enable `Fixed distance` before validating if the signed distance between the two planes should stay locked.

Delete a constraint:

- In the constraints tree, click the compact delete button at the right of a constraint title.
- The constraint is removed from the tree and its absolute/relative effects are recalculated from the remaining constraints.

Existing JSON files remain usable. Older layout files that only contain an object list are loaded normally; new saves include `format_version`, `objects` and `constraints`.

Recommended hexapod organization:

- Use rigid groups for physical subassemblies that should behave like one solid part.
- Use absolute fixed on the world/body reference while calibrating other parts.
- For dynamic motion, the clean model should not be `STL drives servo` directly. It should be `joint/segment model drives STL and servo`.
- Define a kinematic skeleton: body frame, coxa joint, femur joint, tibia joint, each with a pivot point, rotation axis, min/max angle and servo mapping.
- Attach visual STL parts to those skeleton segments with fixed local offsets.

This keeps the CAD display, mechanical constraints and servo calibration separated enough that the system can grow without becoming tangled.

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


Viewport selection:

- Outside constraint and coincidence modes, moving the cursor over a 3D object highlights it in light blue.
- Clicking the object selects it, turns it blue, loads its transform controls and highlights constrained related objects in orange.
- Clicking empty viewport space clears the current object selection in normal selection mode.
- The scene tree selection and viewport selection stay synchronized.

## Alignment Tools

There are now two alignment layers.

Direct coincidence:

- Click `Coincidence`.
- A floating assistant window opens with the current picks, `Clear coincidence picks`, `Reverse coincidence` and `Exit coincidence`.
- Move the cursor over the model: a selectable surface is highlighted in yellow, or a detected hole/cylindrical axis is shown as a yellow axis.
- Click the face or axis of the object you want to move: it stays highlighted in orange.
- Click the target face or axis on another object: it is also highlighted in orange.
- The first object is transformed until the two picked features coincide.

This direct mode starts from the STL triangle under the cursor. On planar areas, it expands the selection to connected coplanar triangles so the highlighted element behaves like a full planar surface. On cylindrical or hole-like faceted areas, it analyses neighboring triangle normals to infer an axis, even when the STL is not a perfect circle. To select a hole axis, aim at the inner wall of the hole; clicking empty space at the center of the hole cannot be picked yet.

- `Remove selected model`: removes the selected object. You can also use the compact delete button at the right of a model title in the scene tree.

Persistent object-to-axis and object-to-plane constraints now keep the picked feature data when available, but the automatic re-solve-after-load layer is still a next step.

## STL Quality Note

STL files are already triangulated meshes. The app now preserves the original triangles instead of subdividing them, then improves the display with clean normals, sharp-edge handling, lighting and anti-aliasing, closer to the way slicers display mechanical parts.

Use `Sharp CAD view` for normal work. Use `Smooth preview` only if you want a softer visual rendering, and `Facets debug` when you want to inspect the actual triangles.

If a STL was exported with too few triangles, the app cannot recover the original CAD curves perfectly. For best results, export STL files from your CAD software with a fine chord tolerance / deviation and a fine angular tolerance.

## Run

```powershell
python hexapod_3d_interface.py
```

## Next development steps

- Serialize picked geometric features for persistent point/axis/plane constraints
- Add local pivots for mechanical joints
- Add parent/child hierarchy for body, coxa, femur and tibia parts
- Add joint angle limits
- Later: map joint angles to servo commands





















