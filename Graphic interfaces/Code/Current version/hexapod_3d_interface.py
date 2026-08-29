from __future__ import annotations

import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker


ALIGNMENT_FEATURES = [
    ("X min plane", ("x", "min")),
    ("X center plane", ("x", "center")),
    ("X max plane", ("x", "max")),
    ("Y min plane", ("y", "min")),
    ("Y center plane", ("y", "center")),
    ("Y max plane", ("y", "max")),
    ("Z min plane", ("z", "min")),
    ("Z center plane", ("z", "center")),
    ("Z max plane", ("z", "max")),
]
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
SURFACE_ANGLE_TOLERANCE_DEG = 8.0
SURFACE_PLANE_TOLERANCE_RATIO = 0.002
SURFACE_MIN_PLANE_TOLERANCE = 0.05
AXIS_SMOOTH_ANGLE_TOLERANCE_DEG = 50.0
AXIS_MIN_CELLS = 8


@dataclass
class SceneObject:
    name: str
    file_path: str
    position: list[float]
    rotation: list[float]


@dataclass
class PickedSurface:
    object_name: str
    local_point: np.ndarray
    local_normal: np.ndarray
    cell_id: int
    cell_ids: list[int]
    feature_kind: str = "surface"
    local_axis_point: Optional[np.ndarray] = None
    local_axis_direction: Optional[np.ndarray] = None
    local_axis_length: float = 0.0
    local_axis_radius: float = 0.0


@dataclass
class MeshTopology:
    cell_points: list[np.ndarray]
    cell_centers: np.ndarray
    cell_normals: np.ndarray
    neighbors: list[set[int]]
    diagonal: float


class HexapodModeler(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scarlett Hexapod - 3D calibration workspace")
        self.resize(1450, 900)

        self.objects: dict[str, SceneObject] = {}
        self.meshes: dict[str, pv.PolyData] = {}
        self.mesh_topologies: dict[str, MeshTopology] = {}
        self.actors = {}
        self.selected_name: Optional[str] = None
        self.coincidence_mode = False
        self.coincidence_picks: list[PickedSurface] = []
        self.last_coincidence: Optional[tuple[PickedSurface, PickedSurface]] = None
        self.coincidence_orientation_sign = -1.0
        self.hover_highlight_name = "__coincidence_hover__"
        self.hover_boundary_name = "__coincidence_hover_boundary__"
        self.pick_highlight_names: list[str] = []
        self.pick_boundary_names: list[str] = []
        self.hover_pick_signature: Optional[tuple[str, int, int]] = None
        self._updating_controls = False
        self._updating_alignment_controls = False
        self.display_mode = "sharp"
        self.step_linear_tolerance = 0.02
        self.step_angular_tolerance = 0.05

        self._build_ui()
        self._build_scene()

    def _build_ui(self) -> None:
        import_action = QAction("Import STL/STP", self)
        import_action.triggered.connect(self.import_model)
        save_action = QAction("Save layout", self)
        save_action.triggered.connect(self.save_layout)
        load_action = QAction("Load layout", self)
        load_action.triggered.connect(self.load_layout)
        reset_camera_action = QAction("Reset camera", self)
        reset_camera_action.triggered.connect(self.reset_camera)

        toolbar = self.addToolBar("Main tools")
        toolbar.setMovable(False)
        toolbar.addAction(import_action)
        toolbar.addAction(save_action)
        toolbar.addAction(load_action)
        toolbar.addAction(reset_camera_action)

        self.plotter = QtInteractor(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("3D objects")
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)

        self.position_spins = [self._make_spinbox(-10000, 10000, 0.1) for _ in range(3)]
        self.rotation_spins = [self._make_spinbox(-360, 360, 0.1) for _ in range(3)]
        for spin in [*self.position_spins, *self.rotation_spins]:
            spin.valueChanged.connect(self._on_transform_changed)

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Sharp CAD view", "sharp")
        self.quality_combo.addItem("Smooth preview", "smooth")
        self.quality_combo.addItem("Facets debug", "faceted")
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)

        transform_form = QFormLayout()
        transform_form.addRow(QLabel("Position"))
        transform_form.addRow("X", self.position_spins[0])
        transform_form.addRow("Y", self.position_spins[1])
        transform_form.addRow("Z", self.position_spins[2])
        transform_form.addRow(QLabel("Rotation deg"))
        transform_form.addRow("Rx", self.rotation_spins[0])
        transform_form.addRow("Ry", self.rotation_spins[1])
        transform_form.addRow("Rz", self.rotation_spins[2])
        transform_form.addRow("Display quality", self.quality_combo)

        zero_button = QPushButton("Move selected to origin")
        zero_button.clicked.connect(self.move_selected_to_origin)
        remove_button = QPushButton("Remove selected model")
        remove_button.clicked.connect(self.remove_selected_model)

        self.coincidence_button = QPushButton("Coincidence")
        self.coincidence_button.setCheckable(True)
        self.coincidence_button.clicked.connect(self.toggle_coincidence_mode)
        clear_picks_button = QPushButton("Clear coincidence picks")
        clear_picks_button.clicked.connect(self.clear_coincidence_picks)
        self.reverse_coincidence_button = QPushButton("Reverse coincidence")
        self.reverse_coincidence_button.setEnabled(False)
        self.reverse_coincidence_button.clicked.connect(self.reverse_last_coincidence)
        self.coincidence_status = QLabel("Coincidence: inactive")
        self.coincidence_status.setWordWrap(True)

        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        self.selected_feature_combo = QComboBox()
        self.target_feature_combo = QComboBox()
        for label, data in ALIGNMENT_FEATURES:
            self.selected_feature_combo.addItem(label, data)
            self.target_feature_combo.addItem(label, data)
        self.selected_feature_combo.setEnabled(False)
        self.target_feature_combo.setEnabled(False)


        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(QLabel("Scene tree"))
        side_layout.addWidget(self.tree, stretch=1)
        side_layout.addLayout(transform_form)
        side_layout.addWidget(zero_button)
        side_layout.addWidget(remove_button)
        side_layout.addWidget(QLabel("Direct face constraint"))
        side_layout.addWidget(self.coincidence_button)
        side_layout.addWidget(clear_picks_button)
        side_layout.addWidget(self.reverse_coincidence_button)
        side_layout.addWidget(self.coincidence_status)


        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.plotter)
        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1080, 370])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _build_scene(self) -> None:
        self.plotter.set_background("#1f2329")
        self.plotter.enable_anti_aliasing("fxaa")
        self.plotter.enable_eye_dome_lighting()
        self.plotter.add_axes(line_width=3, labels_off=False)
        self.plotter.show_grid(color="#686f7a", grid="back", location="outer", xtitle="X", ytitle="Y", ztitle="Z")
        self.plotter.add_floor("z", color="#2d333b", lighting=False, pad=1.0)
        self._install_click_picker()
        self.reset_camera()

    def _install_click_picker(self) -> None:
        self.cell_picker = vtkCellPicker()
        self.cell_picker.SetTolerance(0.0008)
        interactor = self._vtk_interactor()
        interactor.AddObserver("MouseMoveEvent", self._on_mouse_move, 1.0)
        interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press, 1.0)

    def _vtk_interactor(self):
        return getattr(self.plotter.iren, "interactor", self.plotter.iren)

    def _make_spinbox(self, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setEnabled(False)
        return spin

    def import_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Import 3D model", "", "3D models (*.stl *.step *.stp);;STL (*.stl);;STEP (*.step *.stp)")
        if not file_path:
            return
        path = Path(file_path)
        try:
            mesh = self._load_mesh(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        name = self._unique_name(path.stem)
        self.objects[name] = SceneObject(name=name, file_path=str(path), position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0])
        self.meshes[name] = mesh
        self.mesh_topologies[name] = self._build_mesh_topology(mesh)
        self.actors[name] = self._add_mesh_actor(name, mesh)
        self._apply_transform(name)
        item = QTreeWidgetItem([name])
        item.setData(0, Qt.UserRole, name)
        self.tree.addTopLevelItem(item)
        self.tree.setCurrentItem(item)
        self._refresh_alignment_targets()
        self.plotter.reset_camera()

    def _load_mesh(self, path: Path) -> pv.PolyData:
        suffix = path.suffix.lower()
        if suffix == ".stl":
            mesh = pv.read(path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_geometry()
            return self._prepare_stl_mesh(mesh)
        if suffix in {".stp", ".step"}:
            return self._load_step_mesh(path)
        raise ValueError(f"Unsupported file format: {suffix}")

    def _load_step_mesh(self, path: Path) -> pv.PolyData:
        try:
            import cadquery as cq
        except ImportError as exc:
            raise RuntimeError("STEP/STP import needs 'cadquery'. Activate your environment, then install it with: pip install cadquery.") from exc
        shape = cq.importers.importStep(str(path))
        vertices = []
        faces = []
        for solid in shape.solids().vals():
            verts, tris = solid.tessellate(self.step_linear_tolerance, self.step_angular_tolerance)
            base = len(vertices)
            vertices.extend([[v.x, v.y, v.z] for v in verts])
            for tri in tris:
                faces.extend([3, base + tri[0], base + tri[1], base + tri[2]])
        if not vertices or not faces:
            raise RuntimeError("STEP file imported, but no mesh could be generated.")
        return self._prepare_stl_mesh(pv.PolyData(np.array(vertices), np.array(faces)))

    def _prepare_stl_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        mesh = mesh.extract_geometry().triangulate().clean()
        return mesh.compute_normals(point_normals=True, cell_normals=True, auto_orient_normals=True, consistent_normals=True, split_vertices=True, feature_angle=35.0)

    def _build_mesh_topology(self, mesh: pv.PolyData) -> MeshTopology:
        cell_points = []
        centers = []
        normals = []
        edge_to_cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for cell_id in range(mesh.n_cells):
            cell = mesh.get_cell(cell_id)
            ids = [int(i) for i in cell.point_ids]
            points = np.array(cell.points)
            normal = self._normal_from_points(points)
            if normal is None:
                normal = np.array([0.0, 0.0, 1.0])
            cell_points.append(points)
            centers.append(points.mean(axis=0))
            normals.append(normal)
            for a, b in zip(ids, ids[1:] + ids[:1]):
                edge_to_cells[tuple(sorted((a, b)))].append(cell_id)

        neighbors = [set() for _ in range(mesh.n_cells)]
        for connected_cells in edge_to_cells.values():
            if len(connected_cells) < 2:
                continue
            for cell_id in connected_cells:
                neighbors[cell_id].update(other for other in connected_cells if other != cell_id)
        bounds = mesh.bounds
        diagonal = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
        return MeshTopology(cell_points, np.array(centers), np.array(normals), neighbors, diagonal)

    def _add_mesh_actor(self, name: str, mesh: pv.PolyData):
        display_mesh = mesh
        smooth_shading = self.display_mode in {"sharp", "smooth"}
        show_edges = self.display_mode == "faceted"
        if self.display_mode == "smooth":
            display_mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True, consistent_normals=True, split_vertices=False)
        return self.plotter.add_mesh(display_mesh, name=name, color="#58a6ff" if name == self.selected_name else "#c9d1d9", smooth_shading=smooth_shading, show_edges=show_edges, edge_color="#20242b", ambient=0.28, diffuse=0.72, specular=0.18, specular_power=28, pickable=True)

    def _on_quality_changed(self) -> None:
        self.display_mode = self.quality_combo.currentData()
        self._rebuild_actors()

    def _rebuild_actors(self) -> None:
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        for name in list(self.actors):
            self.plotter.remove_actor(name)
        self.actors.clear()
        for name, mesh in self.meshes.items():
            self.actors[name] = self._add_mesh_actor(name, mesh)
            self._apply_transform(name)
        self.plotter.render()

    def _unique_name(self, base: str) -> str:
        candidate = base
        index = 2
        while candidate in self.objects:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _on_mouse_move(self, obj, event) -> None:
        if not self.coincidence_mode:
            self._clear_hover_highlight()
            return
        picked = self._pick_surface_at_current_mouse_position()
        if picked is None:
            self._clear_hover_highlight()
            return
        signature = (picked.object_name, picked.cell_id, len(picked.cell_ids))
        if signature != self.hover_pick_signature:
            self._show_hover_highlight(picked)

    def _on_left_button_press(self, obj, event) -> None:
        if not self.coincidence_mode:
            return
        picked = self._pick_surface_at_current_mouse_position()
        if picked is None:
            self.coincidence_status.setText("Coincidence: click on a model surface.")
            return
        self.coincidence_picks.append(picked)
        self._add_pick_highlight(picked)
        self.selected_name = picked.object_name
        self._select_tree_item(picked.object_name)
        self._highlight_selected()
        self._update_coincidence_status()
        if len(self.coincidence_picks) == 2:
            self.last_coincidence = (self.coincidence_picks[0], self.coincidence_picks[1])
            self.coincidence_orientation_sign = self._minimal_rotation_orientation_sign(*self.last_coincidence)
            self.reverse_coincidence_button.setEnabled(True)
            self._apply_direct_plane_coincidence(*self.last_coincidence)

    def _pick_surface_at_current_mouse_position(self) -> Optional[PickedSurface]:
        click_x, click_y = self._vtk_interactor().GetEventPosition()
        self.cell_picker.Pick(click_x, click_y, 0, self.plotter.renderer)
        actor = self.cell_picker.GetActor()
        cell_id = self.cell_picker.GetCellId()
        object_name = self._object_name_for_actor(actor)
        if object_name is None or cell_id < 0:
            return None
        return self._picked_surface_from_cell(object_name, cell_id)

    def _object_name_for_actor(self, picked_actor) -> Optional[str]:
        if picked_actor is None:
            return None
        for name, actor in self.actors.items():
            if actor is picked_actor or actor == picked_actor:
                return name
        return None

    def _picked_surface_from_cell(self, object_name: str, cell_id: int) -> Optional[PickedSurface]:
        topology = self.mesh_topologies[object_name]
        if cell_id >= len(topology.cell_points):
            return None
        normal = topology.cell_normals[cell_id]
        if np.linalg.norm(normal) <= 1e-9:
            return None
        axis_feature = self._axis_feature_from_cell(object_name, cell_id)
        if axis_feature is not None:
            return axis_feature

        cell_ids = self._coplanar_connected_cells(object_name, cell_id)
        all_points = np.vstack([topology.cell_points[i] for i in cell_ids])
        local_point = all_points.mean(axis=0)
        local_normal = self._average_surface_normal(topology.cell_normals[cell_ids], normal)
        return PickedSurface(object_name, local_point, local_normal, cell_id, cell_ids)

    def _axis_feature_from_cell(self, object_name: str, seed_cell_id: int) -> Optional[PickedSurface]:
        topology = self.mesh_topologies[object_name]
        cell_ids = self._smooth_connected_cells(object_name, seed_cell_id)
        if len(cell_ids) < AXIS_MIN_CELLS:
            return None

        normals = topology.cell_normals[cell_ids]
        normal_covariance = np.cov(normals.T)
        eigenvalues, eigenvectors = np.linalg.eigh(normal_covariance)
        axis_direction = eigenvectors[:, int(np.argmin(eigenvalues))]
        axis_direction = axis_direction / np.linalg.norm(axis_direction)

        normal_spread = float(np.trace(normal_covariance))
        if normal_spread < 0.04:
            return None
        if float(np.mean(np.abs(normals @ axis_direction))) > 0.28:
            return None

        centers = topology.cell_centers[cell_ids]
        axis_point = centers.mean(axis=0)
        axial = (centers - axis_point) @ axis_direction
        radial_vectors = centers - axis_point - np.outer(axial, axis_direction)
        radii = np.linalg.norm(radial_vectors, axis=1)
        mean_radius = float(np.mean(radii))
        if mean_radius <= 1e-6:
            return None
        if float(np.std(radii) / mean_radius) > 0.35:
            return None

        axis_length = max(float(np.max(axial) - np.min(axial)), topology.diagonal * 0.25)
        return PickedSurface(
            object_name=object_name,
            local_point=axis_point,
            local_normal=axis_direction,
            cell_id=seed_cell_id,
            cell_ids=cell_ids,
            feature_kind="axis",
            local_axis_point=axis_point,
            local_axis_direction=axis_direction,
            local_axis_length=axis_length,
            local_axis_radius=mean_radius,
        )

    def _smooth_connected_cells(self, object_name: str, seed_cell_id: int) -> list[int]:
        topology = self.mesh_topologies[object_name]
        cos_angle = math.cos(math.radians(AXIS_SMOOTH_ANGLE_TOLERANCE_DEG))
        surface_ids = []
        visited = {seed_cell_id}
        queue = deque([seed_cell_id])
        while queue:
            current = queue.popleft()
            surface_ids.append(current)
            current_normal = topology.cell_normals[current]
            for neighbor in topology.neighbors[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_normal = topology.cell_normals[neighbor]
                if float(np.dot(current_normal, neighbor_normal)) >= cos_angle:
                    queue.append(neighbor)
        return surface_ids

    def _coplanar_connected_cells(self, object_name: str, seed_cell_id: int) -> list[int]:
        topology = self.mesh_topologies[object_name]
        seed_normal = topology.cell_normals[seed_cell_id]
        seed_point = topology.cell_centers[seed_cell_id]
        cos_angle = math.cos(math.radians(SURFACE_ANGLE_TOLERANCE_DEG))
        plane_tolerance = max(topology.diagonal * SURFACE_PLANE_TOLERANCE_RATIO, SURFACE_MIN_PLANE_TOLERANCE)
        surface_ids = []
        visited = {seed_cell_id}
        queue = deque([seed_cell_id])
        while queue:
            current = queue.popleft()
            surface_ids.append(current)
            for neighbor in topology.neighbors[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                normal = topology.cell_normals[neighbor]
                if float(np.dot(seed_normal, normal)) < cos_angle:
                    continue
                plane_distance = abs(float(np.dot(topology.cell_centers[neighbor] - seed_point, seed_normal)))
                if plane_distance > plane_tolerance:
                    continue
                queue.append(neighbor)
        return surface_ids

    def _average_surface_normal(self, normals: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = np.array([normal if np.dot(normal, reference) >= 0 else -normal for normal in normals])
        average = aligned.mean(axis=0)
        length = np.linalg.norm(average)
        return reference if length <= 1e-9 else average / length

    def _transform_points(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        homogeneous = np.c_[points, np.ones(len(points))]
        return (homogeneous @ matrix.T)[:, :3]

    def _normal_from_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        origin = points[0]
        for i in range(1, len(points) - 1):
            normal = np.cross(points[i] - origin, points[i + 1] - origin)
            length = np.linalg.norm(normal)
            if length > 1e-9:
                return normal / length
        return None

    def _world_point_for_surface(self, surface: PickedSurface) -> np.ndarray:
        return self._transform_points(np.array([surface.local_point]), self._transform_matrix_for(surface.object_name))[0]

    def _world_normal_for_surface(self, surface: PickedSurface) -> np.ndarray:
        normal = self._rotation_matrix_for(surface.object_name) @ surface.local_normal
        length = np.linalg.norm(normal)
        return np.array([0.0, 0.0, 1.0]) if length <= 1e-9 else normal / length

    def _highlight_offset_for(self, picked: PickedSurface) -> float:
        bounds = self._world_bounds_for(picked.object_name)
        diagonal = np.linalg.norm(np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
        return max(float(diagonal) * 0.0015, 0.02)

    def _highlight_mesh_for(self, picked: PickedSurface) -> pv.PolyData:
        if picked.feature_kind == "axis":
            point, direction, length, radius = self._world_axis_for(picked)
            p0 = point - direction * length * 0.65
            p1 = point + direction * length * 0.65
            line = pv.Line(p0, p1)
            tube_radius = max(radius * 0.08, self._highlight_offset_for(picked) * 1.5)
            return line.tube(radius=tube_radius, n_sides=16)

        mesh = self.meshes[picked.object_name].extract_cells(picked.cell_ids).extract_geometry().triangulate()
        world_points = self._transform_points(np.array(mesh.points), self._transform_matrix_for(picked.object_name))
        world_points = world_points + self._world_normal_for_surface(picked) * self._highlight_offset_for(picked)
        return pv.PolyData(world_points, mesh.faces).clean()

    def _world_axis_for(self, picked: PickedSurface) -> tuple[np.ndarray, np.ndarray, float, float]:
        if picked.local_axis_point is None or picked.local_axis_direction is None:
            point = self._world_point_for_surface(picked)
            direction = self._world_normal_for_surface(picked)
            return point, direction, self.mesh_topologies[picked.object_name].diagonal * 0.5, 1.0
        point = self._transform_points(np.array([picked.local_axis_point]), self._transform_matrix_for(picked.object_name))[0]
        direction = self._rotation_matrix_for(picked.object_name) @ picked.local_axis_direction
        direction = direction / np.linalg.norm(direction)
        return point, direction, picked.local_axis_length, picked.local_axis_radius

    def _highlight_boundary_for(self, highlighted: pv.PolyData) -> pv.PolyData:
        return highlighted.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )

    def _show_hover_highlight(self, picked: PickedSurface) -> None:
        self._clear_hover_highlight(render=False)
        self.hover_pick_signature = (picked.object_name, picked.cell_id, len(picked.cell_ids))
        highlighted = self._highlight_mesh_for(picked)
        self.plotter.add_mesh(
            highlighted,
            name=self.hover_highlight_name,
            color="#f2cc60",
            opacity=0.58,
            show_edges=False,
            pickable=False,
        )
        if picked.feature_kind == "surface":
            self.plotter.add_mesh(
                self._highlight_boundary_for(highlighted),
                name=self.hover_boundary_name,
                color="#fff4b0",
                line_width=4,
                pickable=False,
            )
        self.plotter.render()

    def _clear_hover_highlight(self, render: bool = True) -> None:
        if self.hover_pick_signature is None:
            return
        for actor_name in (self.hover_highlight_name, self.hover_boundary_name):
            try:
                self.plotter.remove_actor(actor_name)
            except ValueError:
                pass
        self.hover_pick_signature = None
        if render:
            self.plotter.render()

    def _add_pick_highlight(self, picked: PickedSurface) -> None:
        name = f"__coincidence_pick_{len(self.pick_highlight_names)}__"
        boundary_name = f"__coincidence_pick_boundary_{len(self.pick_boundary_names)}__"
        self.pick_highlight_names.append(name)
        self.pick_boundary_names.append(boundary_name)
        highlighted = self._highlight_mesh_for(picked)
        self.plotter.add_mesh(
            highlighted,
            name=name,
            color="#ff9f1c",
            opacity=0.72,
            show_edges=False,
            pickable=False,
        )
        if picked.feature_kind == "surface":
            self.plotter.add_mesh(
                self._highlight_boundary_for(highlighted),
                name=boundary_name,
                color="#ffd29a",
                line_width=5,
                pickable=False,
            )
        self._clear_hover_highlight(render=False)
        self.plotter.render()

    def _clear_pick_highlights(self, render: bool = True) -> None:
        for name in [*self.pick_highlight_names, *self.pick_boundary_names]:
            try:
                self.plotter.remove_actor(name)
            except ValueError:
                pass
        self.pick_highlight_names.clear()
        self.pick_boundary_names.clear()
        if render:
            self.plotter.render()

    def toggle_coincidence_mode(self) -> None:
        self.coincidence_mode = self.coincidence_button.isChecked()
        self.clear_coincidence_picks(keep_mode=True)
        if self.coincidence_mode:
            self.coincidence_status.setText("Coincidence: hover a surface, then click the moving surface and the target surface.")
        else:
            self.coincidence_status.setText("Coincidence: inactive")

    def clear_coincidence_picks(self, keep_mode: bool = False) -> None:
        self.coincidence_picks.clear()
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        if not keep_mode:
            self.coincidence_mode = False
            self.coincidence_button.setChecked(False)
            self.coincidence_status.setText("Coincidence: inactive")
        elif self.coincidence_mode:
            self.coincidence_status.setText("Coincidence: hover a surface, then click the moving surface and the target surface.")
        self.plotter.render()

    def _update_coincidence_status(self) -> None:
        count = len(self.coincidence_picks)
        if count == 1:
            first = self.coincidence_picks[0]
            self.coincidence_status.setText(f"Coincidence: first surface selected on '{first.object_name}' ({len(first.cell_ids)} triangles). Click target surface.")
        elif count == 2:
            first, second = self.coincidence_picks
            self.coincidence_status.setText(f"Coincidence: aligning '{first.object_name}' to '{second.object_name}'.")

    def reverse_last_coincidence(self) -> None:
        if self.last_coincidence is None:
            return
        self.coincidence_orientation_sign *= -1.0
        self._apply_direct_plane_coincidence(*self.last_coincidence, clear_selection=False)
        self.coincidence_status.setText("Coincidence: reversed last orientation.")

    def _minimal_rotation_orientation_sign(self, moving: PickedSurface, target: PickedSurface) -> float:
        if moving.feature_kind == "axis" and target.feature_kind == "axis":
            _, moving_direction, _, _ = self._world_axis_for(moving)
            _, target_direction, _, _ = self._world_axis_for(target)
            return 1.0 if float(np.dot(moving_direction, target_direction)) >= 0.0 else -1.0

        moving_normal = self._world_normal_for_surface(moving)
        target_normal = self._world_normal_for_surface(target)
        return 1.0 if float(np.dot(moving_normal, target_normal)) >= 0.0 else -1.0
    def _apply_direct_plane_coincidence(self, moving: PickedSurface, target: PickedSurface, clear_selection: bool = True) -> None:
        if moving.object_name == target.object_name:
            self.coincidence_status.setText("Coincidence: choose two different objects.")
            self.coincidence_picks.clear()
            self._clear_pick_highlights()
            return

        if moving.feature_kind == "axis" and target.feature_kind == "axis":
            self._apply_axis_coincidence(moving, target, clear_selection)
            return
        if moving.feature_kind != target.feature_kind:
            self.coincidence_status.setText("Coincidence: surface-to-axis constraints are not supported yet.")
            self.coincidence_picks.clear()
            self._clear_pick_highlights()
            return

        moving_normal = self._world_normal_for_surface(moving)
        target_normal = self._world_normal_for_surface(target) * self.coincidence_orientation_sign
        rotation_delta = self._rotation_between_vectors(moving_normal, target_normal)
        current_rotation = self._rotation_matrix_for(moving.object_name)
        new_rotation = rotation_delta @ current_rotation

        obj = self.objects[moving.object_name]
        obj.rotation = self._euler_degrees_from_rotation_matrix(new_rotation)
        self._apply_transform(moving.object_name)

        moved_point = self._world_point_for_surface(moving)
        target_point = self._world_point_for_surface(target)
        actual_target_normal = self._world_normal_for_surface(target)
        delta_distance = float(np.dot(target_point - moved_point, actual_target_normal))
        delta = actual_target_normal * delta_distance
        obj.position = [obj.position[i] + float(delta[i]) for i in range(3)]

        self.selected_name = moving.object_name
        self._select_tree_item(moving.object_name)
        self._apply_transform(moving.object_name)
        self._load_selected_into_controls()
        if clear_selection:
            self.coincidence_picks.clear()
            self._clear_pick_highlights(render=False)
        self.coincidence_status.setText(f"Coincidence: '{moving.object_name}' moved onto '{target.object_name}'.")
        self.plotter.render()

    def _apply_axis_coincidence(self, moving: PickedSurface, target: PickedSurface, clear_selection: bool = True) -> None:
        moving_point, moving_direction, _, _ = self._world_axis_for(moving)
        target_point, target_direction, _, _ = self._world_axis_for(target)
        target_direction = target_direction * self.coincidence_orientation_sign
        rotation_delta = self._rotation_between_vectors(moving_direction, target_direction)
        current_rotation = self._rotation_matrix_for(moving.object_name)
        new_rotation = rotation_delta @ current_rotation

        obj = self.objects[moving.object_name]
        obj.rotation = self._euler_degrees_from_rotation_matrix(new_rotation)
        self._apply_transform(moving.object_name)

        moved_point, _, _, _ = self._world_axis_for(moving)
        target_point, target_direction, _, _ = self._world_axis_for(target)
        between_axes = target_point - moved_point
        delta = between_axes - np.dot(between_axes, target_direction) * target_direction
        obj.position = [obj.position[i] + float(delta[i]) for i in range(3)]

        self.selected_name = moving.object_name
        self._select_tree_item(moving.object_name)
        self._apply_transform(moving.object_name)
        self._load_selected_into_controls()
        if clear_selection:
            self.coincidence_picks.clear()
            self._clear_pick_highlights(render=False)
        self.coincidence_status.setText(f"Coincidence: axis of '{moving.object_name}' aligned to axis of '{target.object_name}'.")
        self.plotter.render()
    def _rotation_between_vectors(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        cross = np.cross(source, target)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if dot > 0.999999:
            return np.identity(3)
        if dot < -0.999999:
            axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
            return self._rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis), math.pi)
        skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
        return np.identity(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))

    def _rotation_matrix_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])

    def _euler_degrees_from_rotation_matrix(self, matrix: np.ndarray) -> list[float]:
        sy = -matrix[2, 0]
        cy = math.sqrt(max(0.0, 1.0 - sy * sy))
        if cy > 1e-8:
            rx = math.atan2(matrix[2, 1], matrix[2, 2])
            ry = math.asin(sy)
            rz = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            rx = math.atan2(-matrix[1, 2], matrix[1, 1])
            ry = math.asin(sy)
            rz = 0.0
        return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]

    def _select_tree_item(self, name: str) -> None:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.data(0, Qt.UserRole) == name:
                self.tree.setCurrentItem(item)
                return

    def _on_tree_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self.selected_name = None
            self._set_controls_enabled(False)
            self._refresh_alignment_targets()
            return
        self.selected_name = items[0].data(0, Qt.UserRole)
        self._load_selected_into_controls()
        self._highlight_selected()
        self._refresh_alignment_targets()

    def _load_selected_into_controls(self) -> None:
        if not self.selected_name:
            return
        obj = self.objects[self.selected_name]
        self._updating_controls = True
        for spin, value in zip(self.position_spins, obj.position):
            spin.setValue(value)
        for spin, value in zip(self.rotation_spins, obj.rotation):
            spin.setValue(value)
        self._updating_controls = False
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for spin in [*self.position_spins, *self.rotation_spins]:
            spin.setEnabled(enabled)
        self.selected_feature_combo.setEnabled(enabled)
        self.target_feature_combo.setEnabled(enabled and self.target_combo.count() > 0)
        self.target_combo.setEnabled(enabled and self.target_combo.count() > 0)

    def _on_transform_changed(self) -> None:
        if self._updating_controls or not self.selected_name:
            return
        obj = self.objects[self.selected_name]
        obj.position = [spin.value() for spin in self.position_spins]
        obj.rotation = [spin.value() for spin in self.rotation_spins]
        self._apply_transform(self.selected_name)
        self.plotter.render()

    def _apply_transform(self, name: str) -> None:
        self.actors[name].user_matrix = self._transform_matrix_for(name)

    def _rotation_matrix_for(self, name: str) -> np.ndarray:
        obj = self.objects[name]
        rx, ry, rz = [math.radians(v) for v in obj.rotation]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rot_z @ rot_y @ rot_x

    def _transform_matrix_for(self, name: str) -> np.ndarray:
        obj = self.objects[name]
        matrix = np.identity(4)
        matrix[:3, :3] = self._rotation_matrix_for(name)
        matrix[:3, 3] = obj.position
        return matrix

    def _world_bounds_for(self, name: str) -> tuple[float, float, float, float, float, float]:
        mesh = self.meshes[name]
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        corners = np.array([[xmin, ymin, zmin, 1.0], [xmin, ymin, zmax, 1.0], [xmin, ymax, zmin, 1.0], [xmin, ymax, zmax, 1.0], [xmax, ymin, zmin, 1.0], [xmax, ymin, zmax, 1.0], [xmax, ymax, zmin, 1.0], [xmax, ymax, zmax, 1.0]])
        transformed = corners @ self._transform_matrix_for(name).T
        mins = transformed[:, :3].min(axis=0)
        maxs = transformed[:, :3].max(axis=0)
        return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

    def _feature_value(self, name: str, feature: tuple[str, str]) -> float:
        axis, side = feature
        bounds = self._world_bounds_for(name)
        axis_bounds = {"x": (bounds[0], bounds[1]), "y": (bounds[2], bounds[3]), "z": (bounds[4], bounds[5])}[axis]
        if side == "min":
            return axis_bounds[0]
        if side == "max":
            return axis_bounds[1]
        return (axis_bounds[0] + axis_bounds[1]) / 2.0

    def _world_center_for(self, name: str) -> np.ndarray:
        bounds = self._world_bounds_for(name)
        return np.array([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0])

    def _selected_and_target(self) -> tuple[Optional[str], Optional[str]]:
        selected = self.selected_name
        target = self.target_combo.currentData()
        if not selected or not target or selected == target:
            return None, None
        return selected, target

    def _refresh_alignment_targets(self) -> None:
        if self._updating_alignment_controls:
            return
        previous_target = self.target_combo.currentData()
        self._updating_alignment_controls = True
        self.target_combo.clear()
        for name in self.objects:
            if name != self.selected_name:
                self.target_combo.addItem(name, name)
        if previous_target:
            index = self.target_combo.findData(previous_target)
            if index >= 0:
                self.target_combo.setCurrentIndex(index)
        self._updating_alignment_controls = False
        has_target = self.target_combo.count() > 0 and self.selected_name is not None
        self.target_combo.setEnabled(has_target)
        self.selected_feature_combo.setEnabled(self.selected_name is not None)
        self.target_feature_combo.setEnabled(has_target)

    def align_selected_plane_to_target(self) -> None:
        selected, target = self._selected_and_target()
        if not selected or not target:
            QMessageBox.information(self, "Alignment", "Select one object, then choose a different target object.")
            return
        selected_feature = self.selected_feature_combo.currentData()
        target_feature = self.target_feature_combo.currentData()
        selected_axis, _ = selected_feature
        target_axis, _ = target_feature
        if selected_axis != target_axis:
            QMessageBox.information(self, "Alignment", "This first alignment tool works on parallel bounding planes. Choose features on the same axis.")
            return
        axis_index = AXIS_INDEX[selected_axis]
        delta = self._feature_value(target, target_feature) - self._feature_value(selected, selected_feature)
        self.objects[selected].position[axis_index] += delta
        self._apply_transform(selected)
        self._load_selected_into_controls()
        self.plotter.render()

    def align_selected_center_to_target(self) -> None:
        selected, target = self._selected_and_target()
        if not selected or not target:
            QMessageBox.information(self, "Alignment", "Select one object, then choose a different target object.")
            return
        delta = self._world_center_for(target) - self._world_center_for(selected)
        obj = self.objects[selected]
        obj.position = [obj.position[i] + float(delta[i]) for i in range(3)]
        self._apply_transform(selected)
        self._load_selected_into_controls()
        self.plotter.render()

    def copy_target_rotation_to_selected(self) -> None:
        selected, target = self._selected_and_target()
        if not selected or not target:
            QMessageBox.information(self, "Alignment", "Select one object, then choose a different target object.")
            return
        self.objects[selected].rotation = list(self.objects[target].rotation)
        self._apply_transform(selected)
        self._load_selected_into_controls()
        self.plotter.render()

    def _highlight_selected(self) -> None:
        for name, actor in self.actors.items():
            actor.prop.color = "#58a6ff" if name == self.selected_name else "#c9d1d9"
        self.plotter.render()

    def move_selected_to_origin(self) -> None:
        if not self.selected_name:
            return
        self.objects[self.selected_name].position = [0.0, 0.0, 0.0]
        self._apply_transform(self.selected_name)
        self._load_selected_into_controls()
        self.plotter.render()

    def remove_selected_model(self) -> None:
        if not self.selected_name:
            return
        name = self.selected_name
        self.plotter.remove_actor(name)
        self.objects.pop(name, None)
        self.meshes.pop(name, None)
        self.mesh_topologies.pop(name, None)
        self.actors.pop(name, None)
        self.coincidence_picks = [pick for pick in self.coincidence_picks if pick.object_name != name]
        if self.last_coincidence and name in {self.last_coincidence[0].object_name, self.last_coincidence[1].object_name}:
            self.last_coincidence = None
            self.reverse_coincidence_button.setEnabled(False)
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        item = self.tree.currentItem()
        if item is not None:
            index = self.tree.indexOfTopLevelItem(item)
            if index >= 0:
                self.tree.takeTopLevelItem(index)
        self.selected_name = None
        self._set_controls_enabled(False)
        self._refresh_alignment_targets()
        self.plotter.render()

    def save_layout(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save layout", "hexapod_scene.json", "JSON (*.json)")
        if not file_path:
            return
        data = [asdict(obj) for obj in self.objects.values()]
        Path(file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_layout(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load layout", "", "JSON (*.json)")
        if not file_path:
            return
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.clear_scene()
        for raw in data:
            obj = SceneObject(**raw)
            path = Path(obj.file_path)
            mesh = self._load_mesh(path)
            obj.name = self._unique_name(obj.name)
            self.objects[obj.name] = obj
            self.meshes[obj.name] = mesh
            self.mesh_topologies[obj.name] = self._build_mesh_topology(mesh)
            self.actors[obj.name] = self._add_mesh_actor(obj.name, mesh)
            self._apply_transform(obj.name)
            item = QTreeWidgetItem([obj.name])
            item.setData(0, Qt.UserRole, obj.name)
            self.tree.addTopLevelItem(item)
        self._refresh_alignment_targets()
        self.plotter.reset_camera()

    def clear_scene(self) -> None:
        for name in list(self.objects):
            self.plotter.remove_actor(name)
        self.objects.clear()
        self.meshes.clear()
        self.mesh_topologies.clear()
        self.actors.clear()
        self.tree.clear()
        self.selected_name = None
        self.coincidence_picks.clear()
        self.last_coincidence = None
        self.reverse_coincidence_button.setEnabled(False)
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        self._set_controls_enabled(False)
        self._refresh_alignment_targets()

    def reset_camera(self) -> None:
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HexapodModeler()
    window.show()
    sys.exit(app.exec())











