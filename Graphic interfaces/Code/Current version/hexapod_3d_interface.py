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
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QToolButton,
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
    fixed_absolute: bool = False
    rigid_group: str = ""


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


@dataclass
class ConstraintRecord:
    id: int
    type: str
    name: str
    objects: list[str]
    parameters: dict


@dataclass
class PoseAction:
    label: str
    before: dict[str, dict[str, list[float]]]
    after: dict[str, dict[str, list[float]]]


class ResizeHandle(QWidget):
    def __init__(self, section: "ResizableSection") -> None:
        super().__init__()
        self.section = section
        self.drag_start_y: Optional[float] = None
        self.start_height = 0
        self.setFixedHeight(10)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to resize this section")
        self.setStyleSheet(
            "QWidget { background: #30363d; border-radius: 2px; margin: 3px 38px; }"
            "QWidget:hover { background: #4b5563; }"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self.drag_start_y = event.globalPosition().y()
        self.start_height = self.section.height()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.drag_start_y is None:
            return
        delta = int(event.globalPosition().y() - self.drag_start_y)
        self.section.set_section_height(self.start_height + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.drag_start_y = None
        event.accept()


class ResizableSection(QWidget):
    def __init__(self, title: str, widgets: list[QWidget], initial_height: int, minimum_height: int = 120) -> None:
        super().__init__()
        self.minimum_section_height = minimum_height
        self.setObjectName("resizableSection")
        self.setStyleSheet("QWidget#resizableSection { border-bottom: 1px solid #343a42; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 10, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel(title))
        for widget in widgets:
            layout.addWidget(widget)
        layout.addWidget(ResizeHandle(self))
        self.set_section_height(initial_height)

    def set_section_height(self, height: int) -> None:
        self.setFixedHeight(max(self.minimum_section_height, height))


class AddConstraintDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add constraint")
        self.resize(430, 360)

        self.type_combo = QComboBox()
        self.type_combo.addItem("Absolute fixity", "absolute")
        self.type_combo.addItem("Relative rigid group", "relative")
        self.type_combo.addItem("Object to axis", "object_to_axis")
        self.type_combo.addItem("Object to plane", "object_to_plane")
        self.type_combo.addItem("Parallel planes", "parallel_planes")
        self.type_combo.addItem("Dynamic rotation constraint", "dynamic_rotation")
        self.type_combo.addItem("Other / placeholder", "other")
        self.type_combo.currentIndexChanged.connect(self._update_help_text)

        self.group_name = QLineEdit()
        self.group_name.setPlaceholderText("Optional group/constraint name")

        self.fixed_distance_checkbox = QCheckBox("Fixed distance")
        self.fixed_distance_checkbox.setVisible(False)

        self.help_label = QLabel()
        self.help_label.setWordWrap(True)

        self.selection_list = QListWidget()
        self.selection_list.setSelectionMode(QAbstractItemView.NoSelection)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Type", self.type_combo)
        form.addRow("Name", self.group_name)
        form.addRow(self.fixed_distance_checkbox)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.help_label)
        layout.addWidget(QLabel("Current selection"))
        layout.addWidget(self.selection_list)
        layout.addWidget(buttons)
        self._update_help_text()

    def selected_type(self) -> str:
        return self.type_combo.currentData()

    def entered_name(self) -> str:
        return self.group_name.text().strip()

    def fixed_distance(self) -> bool:
        return self.fixed_distance_checkbox.isChecked()

    def set_selection_summary(self, entries: list[str], can_accept: bool) -> None:
        self.selection_list.clear()
        for entry in entries:
            QListWidgetItem(entry, self.selection_list)
        self.ok_button.setEnabled(can_accept)

    def _update_help_text(self) -> None:
        help_by_type = {
            "absolute": "Select one or more whole 3D objects directly in the viewport, then validate with OK.",
            "relative": "Select the objects that must move as one rigid assembly, then validate with OK.",
            "object_to_axis": "Click the constrained object, then click the target hole/cylindrical axis in the viewport. Validate with OK.",
            "object_to_plane": "Click the constrained object, then click the target plane in the viewport. Validate with OK.",
            "parallel_planes": "Select two planes on two different objects. Enable Fixed distance to keep the initial plane distance locked.",
            "dynamic_rotation": "Click the core object, click its live rotation axis, then click every object that must rotate with that core. Validate with OK.",
            "other": "Select one or more objects directly in the viewport, then validate with OK.",
        }
        self.fixed_distance_checkbox.setVisible(self.selected_type() == "parallel_planes")
        self.help_label.setText(help_by_type[self.selected_type()])


class CoincidenceDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Coincidence")
        self.resize(420, 260)

        self.status_label = QLabel("Coincidence: inactive")
        self.status_label.setWordWrap(True)
        self.selection_list = QListWidget()
        self.selection_list.setSelectionMode(QAbstractItemView.NoSelection)

        self.clear_button = QPushButton("Clear coincidence picks")
        self.reverse_button = QPushButton("Reverse coincidence")
        self.close_button = QPushButton("Exit coincidence")
        self.reverse_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Current picks"))
        layout.addWidget(self.selection_list)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.reverse_button)
        layout.addWidget(self.close_button)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_picks(self, picks: list[PickedSurface]) -> None:
        self.selection_list.clear()
        for index, pick in enumerate(picks, start=1):
            label = "Axis" if pick.feature_kind == "axis" else "Plane"
            QListWidgetItem(f"{index}. {label}: {pick.object_name} ({len(pick.cell_ids)} cells)", self.selection_list)

class HexapodModeler(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scarlett Hexapod - 3D calibration workspace")
        self.resize(1450, 900)

        self.objects: dict[str, SceneObject] = {}
        self.constraints: list[ConstraintRecord] = []
        self.next_constraint_id = 1
        self.constraint_dialog: Optional[AddConstraintDialog] = None
        self.constraint_pick_mode = False
        self.pending_constraint_type = ""
        self.pending_constraint_name = ""
        self.pending_constraint_objects: list[str] = []
        self.pending_constraint_features: list[PickedSurface] = []
        self.meshes: dict[str, pv.PolyData] = {}
        self.mesh_topologies: dict[str, MeshTopology] = {}
        self.actors = {}
        self.selected_name: Optional[str] = None
        self.coincidence_mode = False
        self.coincidence_dialog: Optional[CoincidenceDialog] = None
        self.coincidence_picks: list[PickedSurface] = []
        self.last_coincidence: Optional[tuple[PickedSurface, PickedSurface]] = None
        self.coincidence_orientation_sign = -1.0
        self.hover_highlight_name = "__coincidence_hover__"
        self.hover_boundary_name = "__coincidence_hover_boundary__"
        self.pick_highlight_names: list[str] = []
        self.pick_boundary_names: list[str] = []
        self.hover_pick_signature: Optional[tuple[str, int, int]] = None
        self.constraint_hover_object_name: Optional[str] = None
        self._updating_controls = False
        self._updating_alignment_controls = False
        self._solving_constraints = False
        self._restoring_history = False
        self._updating_dynamic_axis_angle = False
        self._last_dynamic_axis_angle = 0.0
        self.undo_stack: list[PoseAction] = []
        self.redo_stack: list[PoseAction] = []
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
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo_last_action)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo_last_action)

        toolbar = self.addToolBar("Main tools")
        toolbar.setMovable(False)
        toolbar.addAction(import_action)
        toolbar.addAction(save_action)
        toolbar.addAction(load_action)
        toolbar.addAction(reset_camera_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)

        self.plotter = QtInteractor(self)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["3D objects", ""])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.setColumnWidth(1, 24)
        self.tree.setMinimumHeight(90)
        self.tree.setTextElideMode(Qt.ElideRight)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tree.setStyleSheet("QTreeWidget::item { min-height: 22px; }")
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)

        self.constraints_tree = QTreeWidget()
        self.constraints_tree.setColumnCount(2)
        self.constraints_tree.setHeaderLabels(["Constraints", ""])
        self.constraints_tree.header().setStretchLastSection(False)
        self.constraints_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.constraints_tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.constraints_tree.setColumnWidth(1, 24)
        self.constraints_tree.setMinimumHeight(90)
        self.constraints_tree.setTextElideMode(Qt.ElideRight)
        self.constraints_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.constraints_tree.setStyleSheet("QTreeWidget::item { min-height: 22px; }")
        add_constraint_button = QPushButton("Add constraint")
        add_constraint_button.clicked.connect(self.add_constraint)
        self.finish_constraint_button = QPushButton("Finish constraint")
        self.finish_constraint_button.setEnabled(False)
        self.finish_constraint_button.clicked.connect(self.finish_constraint_picking)
        self.cancel_constraint_button = QPushButton("Cancel constraint")
        self.cancel_constraint_button.setEnabled(False)
        self.cancel_constraint_button.clicked.connect(self.cancel_constraint_picking)

        self.position_spins = [self._make_spinbox(-10000, 10000, 0.1) for _ in range(3)]
        self.rotation_spins = [self._make_spinbox(-360, 360, 0.1) for _ in range(3)]
        for spin in [*self.position_spins, *self.rotation_spins]:
            spin.valueChanged.connect(self._on_transform_changed)
        self.dynamic_axis_angle_spin = self._make_spinbox(-360, 360, 1.0)
        self.dynamic_axis_angle_spin.setToolTip("Angle to apply around the selected core object's live dynamic rotation axis.")
        self.dynamic_axis_angle_spin.valueChanged.connect(self._on_dynamic_axis_angle_changed)
        self.dynamic_axis_button = QPushButton("Reset dynamic angle")
        self.dynamic_axis_button.setEnabled(False)
        self.dynamic_axis_button.clicked.connect(self.reset_dynamic_axis_angle)

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
        transform_form.addRow("Dynamic axis deg", self.dynamic_axis_angle_spin)
        transform_form.addRow("Display quality", self.quality_combo)

        zero_button = QPushButton("Move selected to origin")
        zero_button.clicked.connect(self.move_selected_to_origin)
        remove_button = QPushButton("Remove selected model")
        remove_button.clicked.connect(self.remove_selected_model)

        self.fixed_absolute_checkbox = QCheckBox("Fixed absolute")
        self.fixed_absolute_checkbox.setEnabled(False)
        self.fixed_absolute_checkbox.stateChanged.connect(self._on_fixed_absolute_changed)

        self.rigid_group_edit = QLineEdit()
        self.rigid_group_edit.setPlaceholderText("Example: body, front_left_coxa")
        self.rigid_group_edit.setEnabled(False)
        self.rigid_group_edit.editingFinished.connect(self._on_rigid_group_changed)

        constraints_form = QFormLayout()
        constraints_form.addRow(QLabel("Constraints"))
        constraints_form.addRow(self.fixed_absolute_checkbox)
        constraints_form.addRow("Rigid group", self.rigid_group_edit)

        self.coincidence_button = QPushButton("Coincidence")
        self.coincidence_button.setCheckable(True)
        self.coincidence_button.clicked.connect(self.toggle_coincidence_mode)
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

        scene_tree_panel = ResizableSection("Scene tree", [self.tree], initial_height=230)
        constraints_tree_panel = ResizableSection("Constraints", [self.constraints_tree, add_constraint_button], initial_height=270)

        controls_panel = QWidget()
        controls_panel.setObjectName("controlsPanel")
        controls_panel.setStyleSheet("QWidget#controlsPanel { border-bottom: 1px solid #343a42; }")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(12, 8, 10, 8)
        controls_layout.addLayout(transform_form)
        controls_layout.addLayout(constraints_form)
        controls_layout.addWidget(self.dynamic_axis_button)
        controls_layout.addWidget(zero_button)
        controls_layout.addWidget(remove_button)
        controls_layout.addWidget(self.coincidence_button)
        controls_layout.addWidget(self.coincidence_status)
        controls_layout.addStretch(1)
        controls_panel.setMinimumHeight(420)

        side_panel = QWidget()
        side_panel.setMinimumHeight(800)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(scene_tree_panel)
        side_layout.addWidget(constraints_tree_panel)
        side_layout.addWidget(controls_panel)
        side_layout.addStretch(1)


        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setStyleSheet("QSplitter::handle:horizontal { background: #30363d; margin: 0 2px; }")
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QScrollArea.NoFrame)
        side_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        side_scroll.setWidget(side_panel)

        splitter.addWidget(self.plotter)
        splitter.addWidget(side_scroll)
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

    def _can_render(self) -> bool:
        if self.plotter.width() <= 0 or self.plotter.height() <= 0:
            return False
        render_window = getattr(self.plotter, "render_window", None)
        if render_window is not None:
            width, height = render_window.GetSize()
            if width <= 0 or height <= 0:
                return False
        return True

    def _render(self) -> None:
        if not self._can_render():
            return
        self.plotter.render()

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
        item = self._add_scene_tree_item(name)
        self.tree.setCurrentItem(item)
        self._refresh_alignment_targets()
        self.reset_camera()

    def _load_mesh(self, path: Path) -> pv.PolyData:
        suffix = path.suffix.lower()
        if suffix == ".stl":
            mesh = pv.read(path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
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
        mesh = mesh.extract_surface().triangulate().clean()
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
        self._render()

    def _unique_name(self, base: str) -> str:
        candidate = base
        index = 2
        while candidate in self.objects:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _on_mouse_move(self, obj, event) -> None:
        if self.constraint_pick_mode:
            self._update_constraint_hover()
            return
        if not self.coincidence_mode:
            self._update_object_hover()
            return
        self.constraint_hover_object_name = None
        picked = self._pick_surface_at_current_mouse_position()
        if picked is None:
            self._clear_hover_highlight()
            return
        signature = (picked.object_name, picked.cell_id, len(picked.cell_ids))
        if signature != self.hover_pick_signature:
            self._show_hover_highlight(picked)

    def _on_left_button_press(self, obj, event) -> None:
        if self.constraint_pick_mode:
            self._handle_constraint_pick_click()
            return
        if not self.coincidence_mode:
            self._select_object_from_viewport()
            return
        picked = self._pick_surface_at_current_mouse_position()
        if picked is None:
            self._set_coincidence_status("Coincidence: click on a model surface.")
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
            self._refresh_coincidence_dialog()
            self._apply_direct_plane_coincidence(*self.last_coincidence)

    def _update_object_hover(self) -> None:
        self._clear_hover_highlight(render=False)
        object_name = self._pick_object_name_at_current_mouse_position()
        if object_name != self.constraint_hover_object_name:
            self.constraint_hover_object_name = object_name
            self._highlight_selected()

    def _select_object_from_viewport(self) -> None:
        object_name = self._pick_object_name_at_current_mouse_position()
        if object_name is None:
            self._clear_object_selection()
            return
        self.selected_name = object_name
        self._select_tree_item(object_name)
        self._load_selected_into_controls()
        self._highlight_selected()
        self._refresh_alignment_targets()

    def _clear_object_selection(self) -> None:
        self.selected_name = None
        self.constraint_hover_object_name = None
        self.tree.clearSelection()
        self._set_controls_enabled(False)
        self._refresh_alignment_targets()
        self._highlight_selected()

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

        mesh = self.meshes[picked.object_name].extract_cells(picked.cell_ids).extract_surface().triangulate()
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
        self._render()

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
            self._render()

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
        self._render()

    def _clear_pick_highlights(self, render: bool = True) -> None:
        for name in [*self.pick_highlight_names, *self.pick_boundary_names]:
            try:
                self.plotter.remove_actor(name)
            except ValueError:
                pass
        self.pick_highlight_names.clear()
        self.pick_boundary_names.clear()
        if render:
            self._render()

    def toggle_coincidence_mode(self) -> None:
        if self.constraint_pick_mode:
            self._reset_constraint_picking(clear_highlights=True, close_dialog=True)
        if self.coincidence_mode:
            self.exit_coincidence_mode()
            return
        self.coincidence_mode = True
        self.coincidence_button.setChecked(True)
        self.clear_coincidence_picks(keep_mode=True)
        self._show_coincidence_dialog()
        self._set_coincidence_status("Coincidence: hover a surface or axis, then click the moving feature and the target feature.")

    def _show_coincidence_dialog(self) -> None:
        if self.coincidence_dialog is not None:
            self.coincidence_dialog.raise_()
            self.coincidence_dialog.activateWindow()
            return
        dialog = CoincidenceDialog(self)
        self.coincidence_dialog = dialog
        dialog.clear_button.clicked.connect(lambda: self.clear_coincidence_picks(keep_mode=True))
        dialog.reverse_button.clicked.connect(self.reverse_last_coincidence)
        dialog.close_button.clicked.connect(self.exit_coincidence_mode)
        dialog.rejected.connect(self.exit_coincidence_mode)
        dialog.destroyed.connect(lambda: self._clear_coincidence_dialog_reference(dialog))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._refresh_coincidence_dialog()

    def _clear_coincidence_dialog_reference(self, dialog: CoincidenceDialog) -> None:
        if self.coincidence_dialog is dialog:
            self.coincidence_dialog = None

    def _set_coincidence_status(self, text: str) -> None:
        self.coincidence_status.setText(text)
        if self.coincidence_dialog is not None:
            self.coincidence_dialog.set_status(text)

    def _refresh_coincidence_dialog(self) -> None:
        if self.coincidence_dialog is None:
            return
        self.coincidence_dialog.set_picks(self.coincidence_picks)
        self.coincidence_dialog.reverse_button.setEnabled(self.last_coincidence is not None)

    def exit_coincidence_mode(self) -> None:
        dialog = self.coincidence_dialog
        self.coincidence_dialog = None
        self.coincidence_mode = False
        self.coincidence_button.setChecked(False)
        self.clear_coincidence_picks(keep_mode=False)
        if dialog is not None:
            dialog.blockSignals(True)
            dialog.reject()
            dialog.blockSignals(False)
        self._set_coincidence_status("Coincidence: inactive")

    def clear_coincidence_picks(self, keep_mode: bool = False) -> None:
        self.coincidence_picks.clear()
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        if not keep_mode:
            self.coincidence_mode = False
            self.coincidence_button.setChecked(False)
            self._set_coincidence_status("Coincidence: inactive")
        elif self.coincidence_mode:
            self._set_coincidence_status("Coincidence: hover a surface or axis, then click the moving feature and the target feature.")
        self._refresh_coincidence_dialog()
        self._render()

    def _update_coincidence_status(self) -> None:
        count = len(self.coincidence_picks)
        if count == 1:
            first = self.coincidence_picks[0]
            label = "axis" if first.feature_kind == "axis" else "surface"
            self._set_coincidence_status(f"Coincidence: first {label} selected on '{first.object_name}' ({len(first.cell_ids)} triangles). Click target feature.")
        elif count == 2:
            first, second = self.coincidence_picks
            self._set_coincidence_status(f"Coincidence: aligning '{first.object_name}' to '{second.object_name}'.")
        self._refresh_coincidence_dialog()

    def reverse_last_coincidence(self) -> None:
        if self.last_coincidence is None:
            return
        self.coincidence_orientation_sign *= -1.0
        self._apply_direct_plane_coincidence(*self.last_coincidence, clear_selection=False)
        self._set_coincidence_status("Coincidence: reversed last orientation.")
        self._refresh_coincidence_dialog()

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
            self._set_coincidence_status("Coincidence: choose two different objects.")
            self.coincidence_picks.clear()
            self._clear_pick_highlights()
            return

        if moving.feature_kind == "axis" and target.feature_kind == "axis":
            self._apply_axis_coincidence(moving, target, clear_selection)
            return
        if moving.feature_kind != target.feature_kind:
            self._set_coincidence_status("Coincidence: surface-to-axis constraints are not supported yet.")
            self.coincidence_picks.clear()
            self._clear_pick_highlights()
            return

        if self.objects[moving.object_name].fixed_absolute:
            self._set_coincidence_status(f"Constraint: '{moving.object_name}' is fixed absolute and cannot be moved.")
            self.coincidence_picks.clear()
            self._clear_pick_highlights()
            return
        affected_names = self._movement_affected_names(moving.object_name)
        before = self._snapshot_objects(affected_names)
        old_matrix = self._transform_matrix_for(moving.object_name)

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
        self._propagate_rigid_group_delta(moving.object_name, old_matrix)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Coincidence {moving.object_name}", before, after)
        if clear_selection:
            self.coincidence_picks.clear()
            self._clear_pick_highlights(render=False)
        self._set_coincidence_status(f"Coincidence: '{moving.object_name}' moved onto '{target.object_name}'.")
        self._render()

    def _apply_axis_coincidence(self, moving: PickedSurface, target: PickedSurface, clear_selection: bool = True) -> None:
        if self.objects[moving.object_name].fixed_absolute:
            self._set_coincidence_status(f"Constraint: '{moving.object_name}' is fixed absolute and cannot be moved.")
            self.coincidence_picks.clear()
            self._clear_pick_highlights()
            return
        affected_names = self._movement_affected_names(moving.object_name)
        before = self._snapshot_objects(affected_names)
        old_matrix = self._transform_matrix_for(moving.object_name)

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
        self._propagate_rigid_group_delta(moving.object_name, old_matrix)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Axis coincidence {moving.object_name}", before, after)
        if clear_selection:
            self.coincidence_picks.clear()
            self._clear_pick_highlights(render=False)
        self._set_coincidence_status(f"Coincidence: axis of '{moving.object_name}' aligned to axis of '{target.object_name}'.")
        self._render()
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
        sy = float(np.clip(-matrix[2, 0], -1.0, 1.0))
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

    def _add_scene_tree_item(self, name: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name, ""])
        item.setData(0, Qt.UserRole, name)
        self.tree.addTopLevelItem(item)
        self.tree.setItemWidget(item, 1, self._scene_tree_delete_button(name))
        return item

    def _scene_tree_delete_button(self, name: str) -> QToolButton:
        delete_button = self._make_delete_button("Delete model")
        delete_button.clicked.connect(lambda _checked=False, object_name=name: self.remove_model(object_name))
        return delete_button
    def _make_delete_button(self, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText("✕")
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setFixedSize(18, 18)
        button.setStyleSheet(
            "QToolButton { border: none; color: #8b949e; font-size: 11px; font-weight: 600; padding: 0; }"
            "QToolButton:hover { color: #ff7b72; background: rgba(248, 81, 73, 0.12); border-radius: 9px; }"
        )
        return button
    def _select_tree_item(self, name: str) -> None:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.data(0, Qt.UserRole) == name:
                self.tree.setCurrentItem(item)
                return

    def add_constraint(self) -> None:
        if not self.objects:
            QMessageBox.information(self, "Add constraint", "Import at least one 3D object first.")
            return
        if self.constraint_dialog is not None:
            self.constraint_dialog.raise_()
            self.constraint_dialog.activateWindow()
            return

        dialog = AddConstraintDialog(self)
        self.constraint_dialog = dialog
        dialog.type_combo.currentIndexChanged.connect(lambda _index: self._on_constraint_dialog_changed(reset_selection=True))
        dialog.group_name.textChanged.connect(lambda _text: self._on_constraint_dialog_changed(reset_selection=False))
        dialog.fixed_distance_checkbox.stateChanged.connect(lambda _state: self._on_constraint_dialog_changed(reset_selection=False))
        dialog.accepted.connect(self.finish_constraint_picking)
        dialog.rejected.connect(lambda: self._reset_constraint_picking(clear_highlights=True, close_dialog=False))
        dialog.destroyed.connect(lambda: self._clear_constraint_dialog_reference(dialog))
        self._begin_constraint_picking(dialog.selected_type(), dialog.entered_name())
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _clear_constraint_dialog_reference(self, dialog: AddConstraintDialog) -> None:
        if self.constraint_dialog is dialog:
            self.constraint_dialog = None

    def _begin_constraint_picking(self, constraint_type: str, name: str) -> None:
        if self.coincidence_mode:
            self.clear_coincidence_picks(keep_mode=False)
        self.constraint_pick_mode = True
        self.pending_constraint_type = constraint_type
        self.pending_constraint_name = name
        self.pending_constraint_objects.clear()
        self.pending_constraint_features.clear()
        self.constraint_hover_object_name = None
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        self.finish_constraint_button.setEnabled(False)
        self.cancel_constraint_button.setEnabled(True)
        self._refresh_constraint_dialog_state()
        self._highlight_selected()

    def _on_constraint_dialog_changed(self, reset_selection: bool) -> None:
        if self.constraint_dialog is None:
            return
        self.pending_constraint_type = self.constraint_dialog.selected_type()
        self.pending_constraint_name = self.constraint_dialog.entered_name()
        if reset_selection:
            self.pending_constraint_objects.clear()
            self.pending_constraint_features.clear()
            self.constraint_hover_object_name = None
            self._clear_hover_highlight(render=False)
            self._clear_pick_highlights(render=False)
            self._highlight_selected()
        self._refresh_constraint_dialog_state()

    def _constraint_pick_instruction(self) -> str:
        if self.pending_constraint_type in {"absolute", "relative", "other"}:
            count = len(self.pending_constraint_objects)
            return f"Constraint pick: click whole 3D objects ({count} selected). OK enables when at least one object is selected."
        if self.pending_constraint_type == "object_to_axis":
            if not self.pending_constraint_objects:
                return "Constraint pick: click the constrained 3D object."
            if not self.pending_constraint_features:
                return f"Constraint pick: constrained object is '{self.pending_constraint_objects[0]}'. Click a target axis; clicking another plain object replaces the constrained object."
            return "Constraint pick: object and axis selected. OK is enabled; click another object or axis to replace the current choice."
        if self.pending_constraint_type == "object_to_plane":
            if not self.pending_constraint_objects:
                return "Constraint pick: click the constrained 3D object."
            if not self.pending_constraint_features:
                return f"Constraint pick: constrained object is '{self.pending_constraint_objects[0]}'. Click the target plane."
            return "Constraint pick: object and plane selected. OK is enabled; click another plane to replace the current target."
        if self.pending_constraint_type == "parallel_planes":
            count = len(self.pending_constraint_features)
            return f"Constraint pick: click two planes on two different objects ({count}/2 selected). OK enables when both planes are valid."
        if self.pending_constraint_type == "dynamic_rotation":
            if not self.pending_constraint_objects:
                return "Constraint pick: click the core object that drives this dynamic rotation."
            if not self.pending_constraint_features:
                return f"Constraint pick: core is '{self.pending_constraint_objects[0]}'. Click its live rotation axis on any object."
            attached_count = max(0, len(self.pending_constraint_objects) - 1)
            return f"Constraint pick: click attached objects that rotate with '{self.pending_constraint_objects[0]}' ({attached_count} attached). OK enables when at least one attached object is selected."
        return "Constraint pick: select elements in the 3D view."

    def _required_pending_feature_kind(self) -> Optional[str]:
        if self.pending_constraint_type == "object_to_axis":
            return "axis"
        if self.pending_constraint_type == "dynamic_rotation" and self.pending_constraint_objects and not self.pending_constraint_features:
            return "axis"
        if self.pending_constraint_type in {"object_to_plane", "parallel_planes"}:
            return "surface"
        return None

    def _refresh_constraint_dialog_state(self) -> None:
        can_accept = self._pending_constraint_is_valid()
        self.finish_constraint_button.setEnabled(can_accept)
        self.cancel_constraint_button.setEnabled(self.constraint_pick_mode)
        if self.constraint_dialog is not None:
            self.constraint_dialog.set_selection_summary(self._pending_constraint_entries(), can_accept)
        self.coincidence_status.setText(self._constraint_pick_instruction())

    def _pending_constraint_is_valid(self) -> bool:
        if self.pending_constraint_type in {"absolute", "relative", "other"}:
            return bool(self.pending_constraint_objects)
        if self.pending_constraint_type in {"object_to_axis", "object_to_plane"}:
            return bool(self.pending_constraint_objects and self.pending_constraint_features)
        if self.pending_constraint_type == "parallel_planes":
            feature_objects = {feature.object_name for feature in self.pending_constraint_features}
            return len(self.pending_constraint_features) == 2 and len(feature_objects) == 2
        if self.pending_constraint_type == "dynamic_rotation":
            return len(self.pending_constraint_objects) >= 2 and len(self.pending_constraint_features) == 1
        return False

    def _pending_constraint_entries(self) -> list[str]:
        if self.pending_constraint_type == "dynamic_rotation":
            entries = []
            if self.pending_constraint_objects:
                entries.append(f"Core: {self.pending_constraint_objects[0]}")
            for name in self.pending_constraint_objects[1:]:
                entries.append(f"Attached: {name}")
        else:
            entries = [f"Object: {name}" for name in self.pending_constraint_objects]
        for feature in self.pending_constraint_features:
            label = "Axis" if feature.feature_kind == "axis" else "Plane"
            entries.append(f"{label}: {feature.object_name} ({len(feature.cell_ids)} cells)")
        return entries

    def _update_constraint_hover(self) -> None:
        required_kind = self._required_pending_feature_kind()
        needs_object_hover = required_kind is None or (not self.pending_constraint_objects and self.pending_constraint_type != "parallel_planes")
        if self.pending_constraint_type == "dynamic_rotation" and self.pending_constraint_features:
            needs_object_hover = True
        if needs_object_hover:
            self._clear_hover_highlight(render=False)
            object_name = self._pick_object_name_at_current_mouse_position()
            if object_name != self.constraint_hover_object_name:
                self.constraint_hover_object_name = object_name
                self._highlight_selected()
            return

        picked = self._pick_surface_at_current_mouse_position()
        if picked is None:
            self._clear_hover_highlight()
            if self.constraint_hover_object_name is not None:
                self.constraint_hover_object_name = None
                self._highlight_selected()
            return
        if picked.feature_kind == required_kind:
            if self.constraint_hover_object_name is not None:
                self.constraint_hover_object_name = None
                self._highlight_selected()
            signature = (picked.object_name, picked.cell_id, len(picked.cell_ids))
            if signature != self.hover_pick_signature:
                self._show_hover_highlight(picked)
            return

        self._clear_hover_highlight(render=False)
        object_name = picked.object_name if self.pending_constraint_type == "object_to_axis" else None
        if object_name != self.constraint_hover_object_name:
            self.constraint_hover_object_name = object_name
            self._highlight_selected()

    def _pick_object_name_at_current_mouse_position(self) -> Optional[str]:
        click_x, click_y = self._vtk_interactor().GetEventPosition()
        self.cell_picker.Pick(click_x, click_y, 0, self.plotter.renderer)
        actor = self.cell_picker.GetActor()
        cell_id = self.cell_picker.GetCellId()
        if cell_id < 0:
            return None
        return self._object_name_for_actor(actor)

    def _handle_constraint_pick_click(self) -> None:
        if self.pending_constraint_type in {"absolute", "relative", "other"}:
            self._handle_object_set_constraint_click()
            return
        if self.pending_constraint_type == "parallel_planes":
            self._handle_parallel_planes_constraint_click()
            return
        if self.pending_constraint_type == "dynamic_rotation":
            self._handle_dynamic_rotation_constraint_click()
            return
        self._handle_object_to_feature_constraint_click()

    def _handle_object_set_constraint_click(self) -> None:
        object_name = self._pick_object_name_at_current_mouse_position()
        if object_name is None:
            self.coincidence_status.setText("Constraint pick: click directly on a 3D object.")
            return
        if object_name in self.pending_constraint_objects:
            self.pending_constraint_objects.remove(object_name)
        else:
            self.pending_constraint_objects.append(object_name)
        self.selected_name = object_name
        self._select_tree_item(object_name)
        self._highlight_selected()
        self._refresh_constraint_dialog_state()

    def _handle_parallel_planes_constraint_click(self) -> None:
        picked = self._pick_surface_at_current_mouse_position()
        if picked is None or picked.feature_kind != "surface":
            self.coincidence_status.setText("Constraint pick: Parallel planes needs planar surface selections.")
            return
        if self.pending_constraint_features and picked.object_name == self.pending_constraint_features[0].object_name:
            self.pending_constraint_features[0] = picked
        elif len(self.pending_constraint_features) < 2:
            self.pending_constraint_features.append(picked)
        else:
            self.pending_constraint_features[1] = picked
        self.pending_constraint_objects = []
        for feature in self.pending_constraint_features:
            if feature.object_name not in self.pending_constraint_objects:
                self.pending_constraint_objects.append(feature.object_name)
        self._clear_pick_highlights(render=False)
        for feature in self.pending_constraint_features:
            self._add_pick_highlight(feature)
        self.selected_name = picked.object_name
        self._select_tree_item(picked.object_name)
        self._highlight_selected()
        self._refresh_constraint_dialog_state()
    def _handle_object_to_feature_constraint_click(self) -> None:
        required_kind = self._required_pending_feature_kind()
        if not self.pending_constraint_objects:
            self._replace_pending_constrained_object_from_click()
            return

        picked = self._pick_surface_at_current_mouse_position()
        if picked is not None and picked.feature_kind == required_kind:
            self.pending_constraint_features = [picked]
            self._clear_pick_highlights(render=False)
            self._add_pick_highlight(picked)
            self._refresh_constraint_dialog_state()
            return

        if self.pending_constraint_type == "object_to_axis":
            self._replace_pending_constrained_object_from_click()
            return

        self.coincidence_status.setText("Constraint pick: click a compatible target feature for this constraint.")

    def _replace_pending_constrained_object_from_click(self) -> None:
        object_name = self._pick_object_name_at_current_mouse_position()
        if object_name is None:
            self.coincidence_status.setText("Constraint pick: click the constrained 3D object first.")
            return
        self.pending_constraint_objects = [object_name]
        self.pending_constraint_features.clear()
        self._clear_pick_highlights(render=False)
        self.selected_name = object_name
        self._select_tree_item(object_name)
        self._highlight_selected()
        self._refresh_constraint_dialog_state()

    def _handle_dynamic_rotation_constraint_click(self) -> None:
        if not self.pending_constraint_objects:
            object_name = self._pick_object_name_at_current_mouse_position()
            if object_name is None:
                self.coincidence_status.setText("Constraint pick: click the core 3D object first.")
                return
            self.pending_constraint_objects = [object_name]
            self.selected_name = object_name
            self._select_tree_item(object_name)
            self._highlight_selected()
            self._refresh_constraint_dialog_state()
            return

        if not self.pending_constraint_features:
            picked = self._pick_surface_at_current_mouse_position()
            if picked is None or picked.feature_kind != "axis":
                self.coincidence_status.setText("Constraint pick: click a detected hole/cylindrical axis for the core.")
                return
            self.pending_constraint_features = [picked]
            self._clear_pick_highlights(render=False)
            self._add_pick_highlight(picked)
            self._refresh_constraint_dialog_state()
            return

        object_name = self._pick_object_name_at_current_mouse_position()
        if object_name is None:
            self.coincidence_status.setText("Constraint pick: click attached 3D objects after selecting the axis.")
            return
        core_name = self.pending_constraint_objects[0]
        if object_name == core_name:
            self.coincidence_status.setText("Constraint pick: the core is already selected. Click attached objects.")
            return
        if object_name in self.pending_constraint_objects:
            self.pending_constraint_objects.remove(object_name)
        else:
            self.pending_constraint_objects.append(object_name)
        self.selected_name = object_name
        self._select_tree_item(object_name)
        self._highlight_selected()
        self._refresh_constraint_dialog_state()

    def finish_constraint_picking(self) -> None:
        if not self.constraint_pick_mode:
            return
        if not self._pending_constraint_is_valid():
            QMessageBox.information(self, "Add constraint", "The current viewport selection is not complete for this constraint type.")
            self._refresh_constraint_dialog_state()
            return

        objects = list(self.pending_constraint_objects)
        for feature in self.pending_constraint_features:
            if feature.object_name not in objects:
                objects.append(feature.object_name)
        constraint = ConstraintRecord(
            id=self.next_constraint_id,
            type=self.pending_constraint_type,
            name=self.pending_constraint_name or self._default_constraint_name(self.pending_constraint_type),
            objects=objects,
            parameters=self._pending_constraint_parameters(),
        )
        self.next_constraint_id += 1
        self.constraints.append(constraint)
        self._apply_constraint_record(constraint)
        self._rebuild_constraints_tree()
        self._reset_constraint_picking(clear_highlights=True, close_dialog=self.sender() is self.finish_constraint_button)
        self._highlight_selected()
        self._update_dynamic_axis_controls()

    def cancel_constraint_picking(self) -> None:
        if not self.constraint_pick_mode:
            return
        self._reset_constraint_picking(clear_highlights=True, close_dialog=True)
        self.coincidence_status.setText("Constraint pick: cancelled.")
        self._highlight_selected()

    def _reset_constraint_picking(self, clear_highlights: bool, close_dialog: bool = False) -> None:
        dialog = self.constraint_dialog
        self.constraint_dialog = None
        self.constraint_pick_mode = False
        self.pending_constraint_type = ""
        self.pending_constraint_name = ""
        self.pending_constraint_objects.clear()
        self.pending_constraint_features.clear()
        self.constraint_hover_object_name = None
        self.finish_constraint_button.setEnabled(False)
        self.cancel_constraint_button.setEnabled(False)
        self._clear_hover_highlight(render=False)
        if clear_highlights:
            self._clear_pick_highlights(render=False)
        if close_dialog and dialog is not None:
            dialog.reject()
        self._render()

    def _pending_constraint_parameters(self) -> dict:
        if self.pending_constraint_type == "relative":
            return {"rigid_group": self.pending_constraint_name or self._default_constraint_name("relative")}
        if self.pending_constraint_type in {"object_to_axis", "object_to_plane"}:
            return {
                "status": "linked_to_viewport_pick",
                "constrained_object": self.pending_constraint_objects[0],
                "picked_features": [self._serialize_pick_for_constraint(pick) for pick in self.pending_constraint_features],
            }
        if self.pending_constraint_type == "parallel_planes":
            features = [self._serialize_pick_for_constraint(pick) for pick in self.pending_constraint_features]
            fixed_distance = bool(self.constraint_dialog.fixed_distance() if self.constraint_dialog is not None else False)
            parameters = {"status": "active", "fixed_distance": fixed_distance, "picked_features": features}
            if fixed_distance and len(self.pending_constraint_features) == 2:
                p0 = self._world_point_for_surface(self.pending_constraint_features[0])
                p1 = self._world_point_for_surface(self.pending_constraint_features[1])
                n1 = self._world_normal_for_surface(self.pending_constraint_features[1])
                parameters["plane_distance"] = float(np.dot(p0 - p1, n1))
            return parameters
        if self.pending_constraint_type == "dynamic_rotation":
            return {
                "status": "active",
                "core_object": self.pending_constraint_objects[0],
                "axis_feature": self._serialize_pick_for_constraint(self.pending_constraint_features[0]),
                "driven_objects": list(self.pending_constraint_objects[1:]),
            }
        return {}

    def _serialize_pick_for_constraint(self, pick: PickedSurface) -> dict:
        data = {
            "object": pick.object_name,
            "kind": pick.feature_kind,
            "cell_id": int(pick.cell_id),
            "cell_ids": [int(cell_id) for cell_id in pick.cell_ids],
            "local_point": [float(value) for value in pick.local_point],
            "local_normal": [float(value) for value in pick.local_normal],
        }
        if pick.local_axis_point is not None:
            data["local_axis_point"] = [float(value) for value in pick.local_axis_point]
        if pick.local_axis_direction is not None:
            data["local_axis_direction"] = [float(value) for value in pick.local_axis_direction]
        if pick.local_axis_radius:
            data["local_axis_radius"] = float(pick.local_axis_radius)
        if pick.local_axis_length:
            data["local_axis_length"] = float(pick.local_axis_length)
        return data
    def _default_constraint_name(self, constraint_type: str) -> str:
        same_type_count = sum(1 for constraint in self.constraints if constraint.type == constraint_type) + 1
        labels = {
            "absolute": "Absolute",
            "relative": "Relative",
            "object_to_axis": "ObjectToAxis",
            "object_to_plane": "ObjectToPlane",
            "parallel_planes": "ParallelPlanes",
            "dynamic_rotation": "DynamicRotation",
            "other": "Other",
        }
        return f"{labels.get(constraint_type, constraint_type)} {same_type_count}"

    def _apply_constraint_record(self, constraint: ConstraintRecord) -> None:
        existing_objects = [name for name in constraint.objects if name in self.objects]
        if constraint.type == "absolute":
            for name in existing_objects:
                self.objects[name].fixed_absolute = True
        elif constraint.type == "relative":
            group_name = constraint.parameters.get("rigid_group") or constraint.name.strip() or f"relative_{constraint.id}"
            constraint.parameters["rigid_group"] = group_name
            for name in existing_objects:
                self.objects[name].rigid_group = group_name
        elif constraint.type in {"object_to_axis", "object_to_plane", "parallel_planes"}:
            constraint.parameters.setdefault("status", "recorded")
            constraint.parameters.setdefault(
                "note",
                "Recorded from the constraints panel. Use direct Coincidence to create picked geometry alignment.",
            )
        elif constraint.type == "dynamic_rotation":
            constraint.parameters.setdefault("status", "active")
            constraint.parameters.setdefault("note", "When the core object rotates, the core and driven objects rotate around the current live axis.")
        if self.selected_name:
            self._load_selected_into_controls()

    def _constraint_tree_expansion_state(self) -> dict[int, dict[str, bool]]:
        state: dict[int, dict[str, bool]] = {}
        for index in range(self.constraints_tree.topLevelItemCount()):
            item = self.constraints_tree.topLevelItem(index)
            constraint_id = item.data(0, Qt.UserRole)
            if constraint_id is None:
                continue
            item_state = {"__root__": item.isExpanded()}
            for child_index in range(item.childCount()):
                child = item.child(child_index)
                item_state[child.text(0)] = child.isExpanded()
            state[int(constraint_id)] = item_state
        return state

    def _rebuild_constraints_tree(self) -> None:
        expansion_state = self._constraint_tree_expansion_state()
        self.constraints_tree.clear()
        for constraint in self.constraints:
            item_state = expansion_state.get(constraint.id, {})
            item = QTreeWidgetItem([f"{constraint.name} ({constraint.type})", ""])
            item.setData(0, Qt.UserRole, constraint.id)
            objects_item = QTreeWidgetItem(["Objects"])
            for object_name in constraint.objects:
                objects_item.addChild(QTreeWidgetItem([object_name]))
            item.addChild(objects_item)

            features = constraint.parameters.get("picked_features", [])
            if features:
                features_item = QTreeWidgetItem(["Picked features"])
                for feature in features:
                    kind = feature.get("kind", "feature")
                    object_name = feature.get("object", "unknown")
                    cell_count = len(feature.get("cell_ids", []))
                    features_item.addChild(QTreeWidgetItem([f"{kind} on {object_name} ({cell_count} cells)"]))
                item.addChild(features_item)
                features_item.setExpanded(item_state.get("Picked features", False))

            axis_feature = constraint.parameters.get("axis_feature")
            if axis_feature:
                axis_object = axis_feature.get("object", "unknown")
                cell_count = len(axis_feature.get("cell_ids", []))
                item.addChild(QTreeWidgetItem([f"Live axis: {axis_object} ({cell_count} cells)"]))

            for key, value in constraint.parameters.items():
                if key in {"picked_features", "axis_feature"}:
                    continue
                item.addChild(QTreeWidgetItem([f"{key}: {value}"]))
            self.constraints_tree.addTopLevelItem(item)
            self.constraints_tree.setItemWidget(item, 1, self._constraint_tree_delete_button(constraint))
            item.setExpanded(item_state.get("__root__", True))
            objects_item.setExpanded(item_state.get("Objects", True))

    def _constraint_tree_delete_button(self, constraint: ConstraintRecord) -> QToolButton:
        delete_button = self._make_delete_button("Delete constraint")
        delete_button.clicked.connect(lambda _checked=False, constraint_id=constraint.id: self.delete_constraint(constraint_id))
        return delete_button

    def delete_constraint(self, constraint_id: int) -> None:
        self.constraints = [constraint for constraint in self.constraints if constraint.id != constraint_id]
        self._reapply_constraint_records()
        self._rebuild_constraints_tree()
        self._highlight_selected()
        self._update_dynamic_axis_controls()

    def _reapply_constraint_records(self) -> None:
        for obj in self.objects.values():
            obj.fixed_absolute = False
            obj.rigid_group = ""
        for constraint in self.constraints:
            self._apply_constraint_record(constraint)
        if self.selected_name:
            self._load_selected_into_controls()
    def _rigid_component_for(self, object_name: str) -> set[str]:
        if object_name not in self.objects:
            return set()
        component = {object_name}
        queue = [object_name]
        while queue:
            current = queue.pop(0)
            current_group = self.objects[current].rigid_group.strip()
            linked: set[str] = set()
            if current_group:
                linked.update(name for name, obj in self.objects.items() if obj.rigid_group.strip() == current_group)
            for constraint in self.constraints:
                if constraint.type == "relative" and current in constraint.objects:
                    linked.update(name for name in constraint.objects if name in self.objects)
            for name in linked:
                if name not in component:
                    component.add(name)
                    queue.append(name)
        return component

    def _related_objects_for(self, object_name: str) -> set[str]:
        if object_name not in self.objects:
            return set()
        related = self._rigid_component_for(object_name) - {object_name}
        for constraint in self.constraints:
            if object_name in constraint.objects:
                related.update(name for name in constraint.objects if name != object_name and name in self.objects)
        return related

    def _remove_object_from_constraints(self, object_name: str) -> None:
        for constraint in self.constraints:
            constraint.objects = [name for name in constraint.objects if name != object_name]
            if constraint.type == "dynamic_rotation":
                if constraint.parameters.get("core_object") == object_name:
                    constraint.objects.clear()
                    continue
                axis_feature = constraint.parameters.get("axis_feature", {})
                if axis_feature.get("object") == object_name:
                    constraint.objects.clear()
                    continue
                constraint.parameters["driven_objects"] = [
                    name for name in constraint.parameters.get("driven_objects", []) if name != object_name
                ]
                if not constraint.parameters["driven_objects"]:
                    constraint.objects.clear()
        self.constraints = [constraint for constraint in self.constraints if constraint.objects]
        self._rebuild_constraints_tree()

    def _on_tree_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self.selected_name = None
            self._set_controls_enabled(False)
            self._refresh_alignment_targets()
            self._highlight_selected()
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
        self.fixed_absolute_checkbox.setChecked(obj.fixed_absolute)
        self.rigid_group_edit.setText(obj.rigid_group)
        self._updating_dynamic_axis_angle = True
        self.dynamic_axis_angle_spin.setValue(0.0)
        self._last_dynamic_axis_angle = 0.0
        self._updating_dynamic_axis_angle = False
        self._updating_controls = False
        self._set_controls_enabled(True)
        self._update_dynamic_axis_controls()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for spin in [*self.position_spins, *self.rotation_spins]:
            spin.setEnabled(enabled)
        self.selected_feature_combo.setEnabled(enabled)
        self.target_feature_combo.setEnabled(enabled and self.target_combo.count() > 0)
        self.target_combo.setEnabled(enabled and self.target_combo.count() > 0)
        self.fixed_absolute_checkbox.setEnabled(enabled)
        self.rigid_group_edit.setEnabled(enabled)
        self._update_dynamic_axis_controls()

    def _update_dynamic_axis_controls(self) -> None:
        enabled = (
            self.selected_name is not None
            and self._dynamic_rotation_constraint_for(self.selected_name) is not None
            and not self.objects[self.selected_name].fixed_absolute
        )
        self.dynamic_axis_angle_spin.setEnabled(enabled)
        self.dynamic_axis_button.setEnabled(enabled)

    def _on_fixed_absolute_changed(self) -> None:
        if self._updating_controls or not self.selected_name:
            return
        self.objects[self.selected_name].fixed_absolute = self.fixed_absolute_checkbox.isChecked()

    def _on_rigid_group_changed(self) -> None:
        if self._updating_controls or not self.selected_name:
            return
        self.objects[self.selected_name].rigid_group = self.rigid_group_edit.text().strip()

    def _dynamic_rotation_constraint_for(self, core_name: Optional[str]) -> Optional[ConstraintRecord]:
        if not core_name:
            return None
        for constraint in self.constraints:
            if constraint.type == "dynamic_rotation" and constraint.parameters.get("core_object") == core_name:
                return constraint
        return None

    def _movement_affected_names(self, moved_name: str) -> set[str]:
        names = {moved_name}
        names.update(self._rigid_component_for(moved_name))
        dynamic_constraint = self._dynamic_rotation_constraint_for(moved_name)
        if dynamic_constraint is not None:
            names.update(name for name in dynamic_constraint.parameters.get("driven_objects", []) if name in self.objects)
        return {name for name in names if name in self.objects}

    def _snapshot_objects(self, names: set[str]) -> dict[str, dict[str, list[float]]]:
        snapshot = {}
        for name in names:
            if name not in self.objects:
                continue
            obj = self.objects[name]
            snapshot[name] = {
                "position": [float(value) for value in obj.position],
                "rotation": [float(value) for value in obj.rotation],
            }
        return snapshot

    def _push_pose_history(self, label: str, before: dict[str, dict[str, list[float]]], after: dict[str, dict[str, list[float]]]) -> None:
        if self._restoring_history or before == after:
            return
        self.undo_stack.append(PoseAction(label=label, before=before, after=after))
        self.redo_stack.clear()
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))

    def _restore_pose_snapshot(self, snapshot: dict[str, dict[str, list[float]]]) -> None:
        self._restoring_history = True
        try:
            for name, pose in snapshot.items():
                if name not in self.objects:
                    continue
                obj = self.objects[name]
                obj.position = [float(value) for value in pose["position"]]
                obj.rotation = [float(value) for value in pose["rotation"]]
                self._apply_transform(name)
            self._load_selected_into_controls()
            self._highlight_selected()
            self._render()
        finally:
            self._restoring_history = False

    def undo_last_action(self) -> None:
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        self._restore_pose_snapshot(action.before)
        self._update_history_actions()

    def redo_last_action(self) -> None:
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        self.undo_stack.append(action)
        self._restore_pose_snapshot(action.after)
        self._update_history_actions()

    def _on_transform_changed(self) -> None:
        if self._updating_controls or not self.selected_name:
            return
        obj = self.objects[self.selected_name]
        if obj.fixed_absolute:
            self.coincidence_status.setText(f"Constraint: '{obj.name}' is fixed absolute and cannot be moved.")
            self._load_selected_into_controls()
            return
        affected_names = self._movement_affected_names(self.selected_name)
        before = self._snapshot_objects(affected_names)
        old_matrix = self._transform_matrix_for(self.selected_name)
        obj.position = [spin.value() for spin in self.position_spins]
        obj.rotation = [spin.value() for spin in self.rotation_spins]
        self._apply_transform(self.selected_name)
        dynamic_rotation_applied = self._apply_dynamic_rotation_constraints(self.selected_name, old_matrix)
        if not dynamic_rotation_applied:
            self._propagate_rigid_group_delta(self.selected_name, old_matrix)
        self._enforce_constraints_after_move(self.selected_name)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Move {self.selected_name}", before, after)
        self._render()

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

    def _set_pose_from_matrix(self, name: str, matrix: np.ndarray) -> None:
        obj = self.objects[name]
        obj.position = [float(value) for value in matrix[:3, 3]]
        obj.rotation = self._euler_degrees_from_rotation_matrix(matrix[:3, :3])
        self._apply_transform(name)

    def _propagate_rigid_group_delta(self, source_name: str, old_source_matrix: np.ndarray) -> None:
        peers = self._rigid_component_for(source_name) - {source_name}
        if not peers:
            return

        new_source_matrix = self._transform_matrix_for(source_name)
        delta_matrix = new_source_matrix @ np.linalg.inv(old_source_matrix)
        for name in peers:
            obj = self.objects[name]
            if obj.fixed_absolute:
                continue
            self._set_pose_from_matrix(name, delta_matrix @ self._transform_matrix_for(name))
    def _world_point_from_feature_data(self, feature: dict) -> np.ndarray:
        object_name = feature.get("object")
        local_point = np.array(feature.get("local_point", [0.0, 0.0, 0.0]), dtype=float)
        return self._transform_points(np.array([local_point]), self._transform_matrix_for(object_name))[0]

    def _world_normal_from_feature_data(self, feature: dict) -> np.ndarray:
        object_name = feature.get("object")
        local_normal = np.array(feature.get("local_normal", [0.0, 0.0, 1.0]), dtype=float)
        normal = self._rotation_matrix_for(object_name) @ local_normal
        length = np.linalg.norm(normal)
        return np.array([0.0, 0.0, 1.0]) if length <= 1e-9 else normal / length

    def _world_axis_from_feature_data(self, feature: dict, matrix_overrides: Optional[dict[str, np.ndarray]] = None) -> Optional[tuple[np.ndarray, np.ndarray]]:
        object_name = feature.get("object")
        if object_name not in self.objects:
            return None
        matrix = matrix_overrides.get(object_name) if matrix_overrides else None
        if matrix is None:
            matrix = self._transform_matrix_for(object_name)
        local_point = np.array(feature.get("local_axis_point", feature.get("local_point", [0.0, 0.0, 0.0])), dtype=float)
        local_direction = np.array(feature.get("local_axis_direction", feature.get("local_normal", [0.0, 0.0, 1.0])), dtype=float)
        point = self._transform_points(np.array([local_point]), matrix)[0]
        direction = matrix[:3, :3] @ local_direction
        length = np.linalg.norm(direction)
        if length <= 1e-9:
            return None
        return point, direction / length

    def _rotation_angle_about_axis(self, rotation_delta: np.ndarray, axis_direction: np.ndarray) -> float:
        axis_direction = axis_direction / np.linalg.norm(axis_direction)
        rotation_vector = np.array(
            [
                rotation_delta[2, 1] - rotation_delta[1, 2],
                rotation_delta[0, 2] - rotation_delta[2, 0],
                rotation_delta[1, 0] - rotation_delta[0, 1],
            ]
        )
        sin_angle = float(np.dot(rotation_vector, axis_direction) * 0.5)
        cos_angle = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
        return math.atan2(sin_angle, cos_angle)

    def _transform_about_world_axis(self, axis_point: np.ndarray, axis_direction: np.ndarray, angle: float) -> np.ndarray:
        rotation = self._rotation_matrix_from_axis_angle(axis_direction / np.linalg.norm(axis_direction), angle)
        matrix = np.identity(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = axis_point - rotation @ axis_point
        return matrix

    def _on_dynamic_axis_angle_changed(self, value: float) -> None:
        if self._updating_controls or self._updating_dynamic_axis_angle or not self.selected_name:
            return
        delta_degrees = float(value) - self._last_dynamic_axis_angle
        self._last_dynamic_axis_angle = float(value)
        if abs(delta_degrees) <= 1e-9:
            return
        if not self._rotate_selected_around_dynamic_axis_by(delta_degrees):
            self._updating_dynamic_axis_angle = True
            self.dynamic_axis_angle_spin.setValue(value - delta_degrees)
            self._last_dynamic_axis_angle = float(value - delta_degrees)
            self._updating_dynamic_axis_angle = False

    def reset_dynamic_axis_angle(self) -> None:
        self._updating_dynamic_axis_angle = True
        self.dynamic_axis_angle_spin.setValue(0.0)
        self._last_dynamic_axis_angle = 0.0
        self._updating_dynamic_axis_angle = False

    def _rotate_selected_around_dynamic_axis_by(self, angle_degrees: float) -> bool:
        if not self.selected_name:
            return False
        constraint = self._dynamic_rotation_constraint_for(self.selected_name)
        if constraint is None:
            self.coincidence_status.setText("Dynamic rotation: selected object has no dynamic rotation constraint.")
            return False
        if self.objects[self.selected_name].fixed_absolute:
            self.coincidence_status.setText(f"Constraint: '{self.selected_name}' is fixed absolute and cannot be moved.")
            return False
        if abs(angle_degrees) <= 1e-9:
            return False
        axis_feature = constraint.parameters.get("axis_feature")
        world_axis = self._world_axis_from_feature_data(axis_feature) if axis_feature else None
        if world_axis is None:
            self.coincidence_status.setText("Dynamic rotation: live axis is unavailable.")
            return False

        affected_names = self._movement_affected_names(self.selected_name)
        before = self._snapshot_objects(affected_names)
        axis_point, axis_direction = world_axis
        axis_transform = self._transform_about_world_axis(axis_point, axis_direction, math.radians(angle_degrees))
        self._set_pose_from_matrix(self.selected_name, axis_transform @ self._transform_matrix_for(self.selected_name))
        for name in constraint.parameters.get("driven_objects", []):
            if name not in self.objects or name == self.selected_name:
                continue
            if self.objects[name].fixed_absolute:
                continue
            self._set_pose_from_matrix(name, axis_transform @ self._transform_matrix_for(name))
        self._enforce_constraints_after_move(self.selected_name)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Dynamic rotate {self.selected_name}", before, after)
        self.coincidence_status.setText(f"Dynamic rotation: '{self.selected_name}' rotated {angle_degrees:.3f} deg around its live axis.")
        self._render()
        return True

    def _apply_dynamic_rotation_constraints(self, moved_name: str, old_moved_matrix: np.ndarray) -> bool:
        applied = False
        requested_matrix = self._transform_matrix_for(moved_name)
        old_rotation = old_moved_matrix[:3, :3]
        requested_rotation = requested_matrix[:3, :3]
        rotation_delta = requested_rotation @ old_rotation.T

        for constraint in self.constraints:
            if constraint.type != "dynamic_rotation":
                continue
            if constraint.parameters.get("core_object") != moved_name:
                continue
            axis_feature = constraint.parameters.get("axis_feature")
            if not axis_feature:
                continue
            axis_matrix_overrides = {moved_name: old_moved_matrix}
            world_axis = self._world_axis_from_feature_data(axis_feature, axis_matrix_overrides)
            if world_axis is None:
                continue
            axis_point, axis_direction = world_axis
            angle = self._rotation_angle_about_axis(rotation_delta, axis_direction)
            if abs(angle) <= 1e-9:
                continue

            axis_transform = self._transform_about_world_axis(axis_point, axis_direction, angle)
            self._set_pose_from_matrix(moved_name, axis_transform @ old_moved_matrix)
            for name in constraint.parameters.get("driven_objects", []):
                if name not in self.objects or name == moved_name:
                    continue
                if self.objects[name].fixed_absolute:
                    continue
                self._set_pose_from_matrix(name, axis_transform @ self._transform_matrix_for(name))
            return True
        return applied

    def _enforce_constraints_after_move(self, moved_name: str) -> None:
        if self._solving_constraints:
            return
        self._solving_constraints = True
        try:
            for constraint in self.constraints:
                if constraint.type == "parallel_planes":
                    self._enforce_parallel_planes_constraint(constraint, moved_name)
        finally:
            self._solving_constraints = False

    def _enforce_parallel_planes_constraint(self, constraint: ConstraintRecord, moved_name: str) -> None:
        features = constraint.parameters.get("picked_features", [])
        if len(features) != 2:
            return
        if moved_name not in {features[0].get("object"), features[1].get("object")}:
            return
        moving_index = 0 if features[0].get("object") == moved_name else 1
        target_index = 1 - moving_index
        moving_feature = features[moving_index]
        target_feature = features[target_index]
        moving_object = moving_feature.get("object")
        target_object = target_feature.get("object")
        if moving_object not in self.objects or target_object not in self.objects:
            return
        if self.objects[moving_object].fixed_absolute:
            return

        old_matrix = self._transform_matrix_for(moving_object)
        moving_normal = self._world_normal_from_feature_data(moving_feature)
        target_normal = self._world_normal_from_feature_data(target_feature)
        if float(np.dot(moving_normal, target_normal)) < 0.0:
            target_normal = -target_normal
        rotation_delta = self._rotation_between_vectors(moving_normal, target_normal)
        new_rotation = rotation_delta @ self._rotation_matrix_for(moving_object)
        self.objects[moving_object].rotation = self._euler_degrees_from_rotation_matrix(new_rotation)
        self._apply_transform(moving_object)

        if constraint.parameters.get("fixed_distance"):
            moving_point = self._world_point_from_feature_data(moving_feature)
            target_point = self._world_point_from_feature_data(target_feature)
            target_normal = self._world_normal_from_feature_data(target_feature)
            desired_distance = float(constraint.parameters.get("plane_distance", 0.0))
            current_distance = float(np.dot(moving_point - target_point, target_normal))
            delta = target_normal * (desired_distance - current_distance)
            obj = self.objects[moving_object]
            obj.position = [obj.position[i] + float(delta[i]) for i in range(3)]
            self._apply_transform(moving_object)

        self._propagate_rigid_group_delta(moving_object, old_matrix)
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
        if self.objects[selected].fixed_absolute:
            self.coincidence_status.setText(f"Constraint: '{selected}' is fixed absolute and cannot be moved.")
            return
        affected_names = self._movement_affected_names(selected)
        before = self._snapshot_objects(affected_names)
        axis_index = AXIS_INDEX[selected_axis]
        delta = self._feature_value(target, target_feature) - self._feature_value(selected, selected_feature)
        old_matrix = self._transform_matrix_for(selected)
        self.objects[selected].position[axis_index] += delta
        self._apply_transform(selected)
        self._propagate_rigid_group_delta(selected, old_matrix)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Align {selected}", before, after)
        self._render()

    def align_selected_center_to_target(self) -> None:
        selected, target = self._selected_and_target()
        if not selected or not target:
            QMessageBox.information(self, "Alignment", "Select one object, then choose a different target object.")
            return
        if self.objects[selected].fixed_absolute:
            self.coincidence_status.setText(f"Constraint: '{selected}' is fixed absolute and cannot be moved.")
            return
        affected_names = self._movement_affected_names(selected)
        before = self._snapshot_objects(affected_names)
        old_matrix = self._transform_matrix_for(selected)
        delta = self._world_center_for(target) - self._world_center_for(selected)
        obj = self.objects[selected]
        obj.position = [obj.position[i] + float(delta[i]) for i in range(3)]
        self._apply_transform(selected)
        self._propagate_rigid_group_delta(selected, old_matrix)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Center align {selected}", before, after)
        self._render()

    def copy_target_rotation_to_selected(self) -> None:
        selected, target = self._selected_and_target()
        if not selected or not target:
            QMessageBox.information(self, "Alignment", "Select one object, then choose a different target object.")
            return
        if self.objects[selected].fixed_absolute:
            self.coincidence_status.setText(f"Constraint: '{selected}' is fixed absolute and cannot be moved.")
            return
        affected_names = self._movement_affected_names(selected)
        before = self._snapshot_objects(affected_names)
        old_matrix = self._transform_matrix_for(selected)
        self.objects[selected].rotation = list(self.objects[target].rotation)
        self._apply_transform(selected)
        self._propagate_rigid_group_delta(selected, old_matrix)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Copy rotation {selected}", before, after)
        self._render()

    def _highlight_selected(self) -> None:
        related = self._related_objects_for(self.selected_name) if self.selected_name else set()
        pending_objects = set(self.pending_constraint_objects) if self.constraint_pick_mode else set()
        for name, actor in self.actors.items():
            if name == self.constraint_hover_object_name and name not in pending_objects:
                actor.prop.color = "#8bd3ff"
            elif name == self.selected_name or name in pending_objects:
                actor.prop.color = "#58a6ff"
            elif name in related:
                actor.prop.color = "#ff9f1c"
            else:
                actor.prop.color = "#c9d1d9"
        self._render()

    def move_selected_to_origin(self) -> None:
        if not self.selected_name:
            return
        if self.objects[self.selected_name].fixed_absolute:
            self.coincidence_status.setText(f"Constraint: '{self.selected_name}' is fixed absolute and cannot be moved.")
            return
        affected_names = self._movement_affected_names(self.selected_name)
        before = self._snapshot_objects(affected_names)
        old_matrix = self._transform_matrix_for(self.selected_name)
        self.objects[self.selected_name].position = [0.0, 0.0, 0.0]
        self._apply_transform(self.selected_name)
        self._propagate_rigid_group_delta(self.selected_name, old_matrix)
        self._enforce_constraints_after_move(self.selected_name)
        self._load_selected_into_controls()
        after = self._snapshot_objects(affected_names)
        self._push_pose_history(f"Move {self.selected_name} to origin", before, after)
        self._render()

    def remove_selected_model(self) -> None:
        if self.selected_name:
            self.remove_model(self.selected_name)

    def remove_model(self, name: str) -> None:
        if name not in self.objects:
            return
        self.plotter.remove_actor(name)
        self.objects.pop(name, None)
        self.meshes.pop(name, None)
        self.mesh_topologies.pop(name, None)
        self.actors.pop(name, None)
        self.coincidence_picks = [pick for pick in self.coincidence_picks if pick.object_name != name]
        self._remove_object_from_constraints(name)
        if self.last_coincidence and name in {self.last_coincidence[0].object_name, self.last_coincidence[1].object_name}:
            self.last_coincidence = None
            self.reverse_coincidence_button.setEnabled(False)
            self._refresh_coincidence_dialog()
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.data(0, Qt.UserRole) == name:
                self.tree.takeTopLevelItem(index)
                break
        if self.selected_name == name:
            self.selected_name = None
            self._set_controls_enabled(False)
        if self.constraint_hover_object_name == name:
            self.constraint_hover_object_name = None
        self._refresh_alignment_targets()
        self._update_dynamic_axis_controls()
        self._highlight_selected()
        self._render()

    def save_layout(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save layout", "hexapod_scene.json", "JSON (*.json)")
        if not file_path:
            return
        data = {
            "format_version": 2,
            "objects": [asdict(obj) for obj in self.objects.values()],
            "constraints": [asdict(constraint) for constraint in self.constraints],
        }
        Path(file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_layout(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load layout", "", "JSON (*.json)")
        if not file_path:
            return
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        object_data = data if isinstance(data, list) else data.get("objects", [])
        constraint_data = [] if isinstance(data, list) else data.get("constraints", [])

        self.clear_scene()
        name_map: dict[str, str] = {}
        for raw in object_data:
            obj = SceneObject(**raw)
            original_name = obj.name
            path = Path(obj.file_path)
            mesh = self._load_mesh(path)
            obj.name = self._unique_name(obj.name)
            name_map[original_name] = obj.name
            self.objects[obj.name] = obj
            self.meshes[obj.name] = mesh
            self.mesh_topologies[obj.name] = self._build_mesh_topology(mesh)
            self.actors[obj.name] = self._add_mesh_actor(obj.name, mesh)
            self._apply_transform(obj.name)
            self._add_scene_tree_item(obj.name)

        self.constraints.clear()
        self.next_constraint_id = 1
        for raw in constraint_data:
            remapped_objects = [name_map.get(name, name) for name in raw.get("objects", []) if name_map.get(name, name) in self.objects]
            parameters = self._remap_constraint_parameters(raw.get("parameters", {}), name_map)
            constraint = ConstraintRecord(
                id=int(raw.get("id", self.next_constraint_id)),
                type=raw.get("type", "other"),
                name=raw.get("name", "Constraint"),
                objects=remapped_objects,
                parameters=parameters,
            )
            self.constraints.append(constraint)
            self.next_constraint_id = max(self.next_constraint_id, constraint.id + 1)
            self._apply_constraint_record(constraint)

        self._rebuild_constraints_tree()
        self._refresh_alignment_targets()
        self._update_dynamic_axis_controls()
        self.reset_camera()

    def _remap_constraint_parameters(self, parameters: dict, name_map: dict[str, str]) -> dict:
        remapped = json.loads(json.dumps(parameters))

        def remap_name(name: str) -> str:
            return name_map.get(name, name)

        if "constrained_object" in remapped:
            remapped["constrained_object"] = remap_name(remapped["constrained_object"])
        if "core_object" in remapped:
            remapped["core_object"] = remap_name(remapped["core_object"])
        if "driven_objects" in remapped:
            remapped["driven_objects"] = [remap_name(name) for name in remapped["driven_objects"]]

        for key in ("picked_features",):
            for feature in remapped.get(key, []):
                if "object" in feature:
                    feature["object"] = remap_name(feature["object"])
        axis_feature = remapped.get("axis_feature")
        if axis_feature and "object" in axis_feature:
            axis_feature["object"] = remap_name(axis_feature["object"])
        return remapped

    def clear_scene(self) -> None:
        for name in list(self.objects):
            self.plotter.remove_actor(name)
        self.objects.clear()
        self.meshes.clear()
        self.mesh_topologies.clear()
        self.actors.clear()
        self.tree.clear()
        self.constraints.clear()
        self.next_constraint_id = 1
        self.constraints_tree.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_actions()
        dialog = self.constraint_dialog
        self.constraint_dialog = None
        if dialog is not None:
            dialog.blockSignals(True)
            dialog.reject()
            dialog.blockSignals(False)
        coincidence_dialog = self.coincidence_dialog
        self.coincidence_dialog = None
        if coincidence_dialog is not None:
            coincidence_dialog.blockSignals(True)
            coincidence_dialog.reject()
            coincidence_dialog.blockSignals(False)
        self.coincidence_mode = False
        self.coincidence_button.setChecked(False)
        self.constraint_pick_mode = False
        self.pending_constraint_type = ""
        self.pending_constraint_name = ""
        self.pending_constraint_objects.clear()
        self.pending_constraint_features.clear()
        self.constraint_hover_object_name = None
        self.finish_constraint_button.setEnabled(False)
        self.cancel_constraint_button.setEnabled(False)
        self.selected_name = None
        self.coincidence_picks.clear()
        self.last_coincidence = None
        self.reverse_coincidence_button.setEnabled(False)
        self._clear_hover_highlight(render=False)
        self._clear_pick_highlights(render=False)
        self._set_controls_enabled(False)
        self._refresh_alignment_targets()
        self._update_dynamic_axis_controls()

    def reset_camera(self) -> None:
        self.plotter.camera_position = "iso"
        if not self._can_render():
            return
        self.plotter.reset_camera()

    def closeEvent(self, event) -> None:
        try:
            self.plotter.disable_eye_dome_lighting()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HexapodModeler()
    window.show()
    sys.exit(app.exec())


































































