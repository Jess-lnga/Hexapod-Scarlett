from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
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


@dataclass
class SceneObject:
    name: str
    file_path: str
    position: list[float]
    rotation: list[float]


class HexapodModeler(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scarlett Hexapod - 3D calibration workspace")
        self.resize(1400, 850)

        self.objects: dict[str, SceneObject] = {}
        self.meshes: dict[str, pv.PolyData] = {}
        self.actors = {}
        self.selected_name: Optional[str] = None
        self._updating_controls = False
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
        self.quality_combo.setCurrentIndex(0)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)

        form = QFormLayout()
        form.addRow(QLabel("Position"))
        form.addRow("X", self.position_spins[0])
        form.addRow("Y", self.position_spins[1])
        form.addRow("Z", self.position_spins[2])
        form.addRow(QLabel("Rotation deg"))
        form.addRow("Rx", self.rotation_spins[0])
        form.addRow("Ry", self.rotation_spins[1])
        form.addRow("Rz", self.rotation_spins[2])
        form.addRow("Display quality", self.quality_combo)

        zero_button = QPushButton("Move selected to origin")
        zero_button.clicked.connect(self.move_selected_to_origin)

        align_button = QPushButton("Alignment tools placeholder")
        align_button.clicked.connect(self.show_alignment_placeholder)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(QLabel("Scene tree"))
        side_layout.addWidget(self.tree, stretch=1)
        side_layout.addLayout(form)
        side_layout.addWidget(zero_button)
        side_layout.addWidget(align_button)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.plotter)
        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1050, 350])

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
        self.plotter.show_grid(
            color="#686f7a",
            grid="back",
            location="outer",
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        )
        self.plotter.add_floor("z", color="#2d333b", lighting=False, pad=1.0)
        self.reset_camera()

    def _make_spinbox(self, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setEnabled(False)
        return spin

    def import_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import 3D model",
            "",
            "3D models (*.stl *.step *.stp);;STL (*.stl);;STEP (*.step *.stp)",
        )
        if not file_path:
            return

        path = Path(file_path)
        try:
            mesh = self._load_mesh(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        name = self._unique_name(path.stem)
        obj = SceneObject(name=name, file_path=str(path), position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0])
        self.objects[name] = obj
        self.meshes[name] = mesh
        self.actors[name] = self._add_mesh_actor(name, mesh)
        self._apply_transform(name)

        item = QTreeWidgetItem([name])
        item.setData(0, Qt.UserRole, name)
        self.tree.addTopLevelItem(item)
        self.tree.setCurrentItem(item)
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
            raise RuntimeError(
                "STEP/STP import needs the optional 'cadquery' package. "
                "Activate your environment, then install it with: pip install cadquery. "
                "If installation is difficult, export the part as a high-resolution STL from your CAD software for now."
            ) from exc

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
        # Keep STL geometry untouched. Display quality is handled with normals and rendering,
        # not destructive subdivision, so mechanical edges stay accurate.
        mesh = mesh.extract_geometry().triangulate().clean()
        return mesh.compute_normals(
            point_normals=True,
            cell_normals=True,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=True,
            feature_angle=35.0,
        )

    def _add_mesh_actor(self, name: str, mesh: pv.PolyData):
        display_mesh = mesh
        smooth_shading = self.display_mode in {"sharp", "smooth"}
        show_edges = self.display_mode == "faceted"

        if self.display_mode == "smooth":
            display_mesh = mesh.compute_normals(
                point_normals=True,
                cell_normals=False,
                auto_orient_normals=True,
                consistent_normals=True,
                split_vertices=False,
            )

        return self.plotter.add_mesh(
            display_mesh,
            name=name,
            color="#58a6ff" if name == self.selected_name else "#c9d1d9",
            smooth_shading=smooth_shading,
            show_edges=show_edges,
            edge_color="#20242b",
            ambient=0.28,
            diffuse=0.72,
            specular=0.18,
            specular_power=28,
        )

    def _on_quality_changed(self) -> None:
        self.display_mode = self.quality_combo.currentData()
        self._rebuild_actors()

    def _rebuild_actors(self) -> None:
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

    def _on_tree_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self.selected_name = None
            self._set_controls_enabled(False)
            return

        self.selected_name = items[0].data(0, Qt.UserRole)
        self._load_selected_into_controls()
        self._highlight_selected()

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

    def _on_transform_changed(self) -> None:
        if self._updating_controls or not self.selected_name:
            return

        obj = self.objects[self.selected_name]
        obj.position = [spin.value() for spin in self.position_spins]
        obj.rotation = [spin.value() for spin in self.rotation_spins]
        self._apply_transform(self.selected_name)
        self.plotter.render()

    def _apply_transform(self, name: str) -> None:
        obj = self.objects[name]
        rx, ry, rz = [math.radians(v) for v in obj.rotation]
        tx, ty, tz = obj.position

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        rot_x = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
        rot_y = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
        rot_z = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        translate = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

        matrix = translate @ rot_z @ rot_y @ rot_x
        self.actors[name].user_matrix = matrix

    def _highlight_selected(self) -> None:
        for name, actor in self.actors.items():
            actor.prop.color = "#58a6ff" if name == self.selected_name else "#c9d1d9"
        self.plotter.render()

    def move_selected_to_origin(self) -> None:
        if not self.selected_name:
            return
        obj = self.objects[self.selected_name]
        obj.position = [0.0, 0.0, 0.0]
        self._apply_transform(self.selected_name)
        self._load_selected_into_controls()
        self.plotter.render()

    def show_alignment_placeholder(self) -> None:
        QMessageBox.information(
            self,
            "Alignment tools",
            "This first version has the scene, import, selection, translation and rotation. "
            "The next layer can add Catia-like constraints: axis coincidence, plane coincidence, "
            "point coincidence and joint limits.",
        )

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
            self.actors[obj.name] = self._add_mesh_actor(obj.name, mesh)
            self._apply_transform(obj.name)
            item = QTreeWidgetItem([obj.name])
            item.setData(0, Qt.UserRole, obj.name)
            self.tree.addTopLevelItem(item)
        self.plotter.reset_camera()

    def clear_scene(self) -> None:
        for name in list(self.objects):
            self.plotter.remove_actor(name)
        self.objects.clear()
        self.meshes.clear()
        self.actors.clear()
        self.tree.clear()
        self.selected_name = None
        self._set_controls_enabled(False)

    def reset_camera(self) -> None:
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HexapodModeler()
    window.show()
    sys.exit(app.exec())
