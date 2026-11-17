from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from math import isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter

from PyQt6.QtCore import QSignalBlocker, QTimer, Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox,
    QStyle, QTabWidget, QVBoxLayout, QWidget,
)
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


SUPPORT_LAB = Path(__file__).resolve().parents[1] / "1027"
DEFAULT_MODEL = SUPPORT_LAB / "ORIGAMI_Chat_Free.obj"
SPIDER_MAN_NAME = "spider-man brand new day"
DEMO_HEIGHT_MM = 80
BRIDGE_SCRIPT = SUPPORT_LAB / "prusa_support_bridge.py"
PRUSA_EXE = Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe")
READERS = {".obj": vtk.vtkOBJReader, ".stl": vtk.vtkSTLReader}
PATH_STYLES = {
    "wall": ("红色外壁", (1.0, 0.26, 0.30)),
    "infill": ("蓝色内部填充", (0.28, 0.62, 1.0)),
    "solid": ("黄色实心层与补缝", (1.0, 0.76, 0.23)),
    "support": ("绿色支撑", (0.12, 0.90, 0.35)),
}
PLAYBACK_SECONDS = 25
BED_SIZE = 220.0  # Matches the virtual printer in prusa_support_bridge.py.
BED_HALF = BED_SIZE / 2
MIN_PRINT_HEIGHT = 0.3


def read_mesh(path):
    reader_type = READERS.get(path.suffix.lower())
    if reader_type is None:
        raise ValueError("请选择 OBJ 或 STL 格式的模型。")
    if not path.is_file():
        raise FileNotFoundError("找不到该模型文件。")

    reader = reader_type()
    errors = []
    reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if errors or reader.GetErrorCode() or not mesh.GetNumberOfPolys():
        raise ValueError("无法读取模型，或文件中没有可显示的表面。")
    if not all(isfinite(value) for value in mesh.GetBounds()):
        raise ValueError("模型的坐标无效。")
    return mesh


def slice_support(mesh, angle, gap, style, cancel, *, density=10, pattern=None):
    spec = spec_from_file_location("prusa_support_bridge", BRIDGE_SCRIPT)
    bridge = module_from_spec(spec)
    spec.loader.exec_module(bridge)
    # A separate directory for each job also isolates multiple viewer instances.
    with TemporaryDirectory(prefix="python3dprint-prusa-") as directory:
        job = bridge.prepare_job(mesh, directory, angle, gap, style,
                                 density=density, pattern=pattern)
        if cancel.is_set():
            return None
        result = bridge.run_job(job, cancel=cancel, include_print=True)
    if cancel.is_set():
        return None
    result["support_count"] = len(result.pop("segments"))
    moves = result.pop("print_moves")
    starts = result.pop("print_layers")
    ends = [index for index, _ in starts[1:]] + [len(moves)]
    result["layers"] = [(z, end) for (_, z), end in zip(starts, ends)]
    indices = {role: [] for role in PATH_STYLES}
    points = {role: [] for role in PATH_STYLES}
    for index, (role, start, end) in enumerate(moves):
        indices[role].append(index)
        points[role].extend((start, end))
    result["tracks"] = {
        role: PrintTrack(indices[role], points[role]) for role in PATH_STYLES
    }
    return result


def printable_mesh(mesh, model_actor):
    transform = vtk.vtkTransform()
    transform.SetMatrix(model_actor.GetMatrix())
    posed = vtk.vtkTransformFilter()
    posed.SetInputData(mesh)
    posed.SetTransform(transform)

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(posed.GetOutputPort())
    triangles.PassVertsOff()
    triangles.PassLinesOff()
    triangles.Update()
    return triangles.GetOutput()


class PrintTrack:
    def __init__(self, indices, coordinates):
        self.indices = np.asarray(indices, dtype=np.int32)
        self.connectivity = np.arange(len(coordinates), dtype=np.int32)
        self.visible_range = (0, 0)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(coordinates, dtype=np.float32).reshape(-1, 3),
                                    deep=True))
        self.lines = vtk.vtkCellArray()
        self.data = vtk.vtkPolyData()
        self.data.SetPoints(points)
        self.data.SetLines(self.lines)

    def show_range(self, first, end):
        start, stop = map(int, self.indices.searchsorted((first, end)))
        if (start, stop) != self.visible_range:
            # Only change the visible connectivity. The point buffer is shared.
            self.lines.SetData(2, numpy_to_vtk(self.connectivity[2 * start:2 * stop]))
            self.data.Modified()
            self.visible_range = (start, stop)


class ModelViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D 打印模型查看器")
        self.resize(1000, 680)
        self.setMinimumSize(600, 420)
        self.current_path = None
        self.prusa_available = BRIDGE_SCRIPT.is_file() and PRUSA_EXE.is_file()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PrusaSupport")
        self.future = None
        self.autoplay = False
        self.cancel_event = None
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self.poll_support)
        self.layers = []
        self.layer_ends = []
        self.tracks = {}
        self.active_layer = 1
        self.play_position = 0.0
        self.platform_fits = False
        self.reslice_required = False
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self.advance_print)

        file_menu = self.menuBar().addMenu("文件")
        view_menu = self.menuBar().addMenu("视图")
        toolbar = self.addToolBar("模型工具")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        open_action = file_menu.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "打开模型",
        )
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_model)
        toolbar.addAction(open_action)

        self.reset_action = view_menu.addAction("重置视角")
        self.reset_action.setEnabled(False)
        self.reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(self.reset_action)
        self.preview_action = view_menu.addAction("切片预览")
        self.preview_action.setEnabled(False)
        self.preview_action.triggered.connect(self.toggle_preview)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        workspace = QWidget(self)
        self.setCentralWidget(workspace)
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout = QHBoxLayout()
        workspace_layout.addLayout(canvas_layout, 1)
        self.viewport = QVTKRenderWindowInteractor(workspace)
        canvas_layout.addWidget(self.viewport, 1)
        self.render_window = self.viewport.GetRenderWindow()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.10, 0.13, 0.18)
        self.renderer.SetBackground2(0.22, 0.27, 0.34)
        self.renderer.GradientBackgroundOn()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.GetInteractor().SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(0.92, 0.16, 0.20)
        self.actor.GetProperty().SetInterpolationToFlat()
        self.add_build_plate()

        self.path_actors = {}
        for role, (_, color) in PATH_STYLES.items():
            mapper = vtk.vtkPolyDataMapper()
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(2)
            actor.GetProperty().LightingOff()
            actor.VisibilityOff()
            self.renderer.AddActor(actor)
            self.path_actors[role] = actor
        self.add_playback_controls(workspace_layout, canvas_layout)
        self.add_support_panel()
        self.statusBar().showMessage("尚未加载模型")

    def add_build_plate(self):
        z = -0.05
        points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        for coordinate in range(-100, 101, 10):
            for a, b in (((-BED_HALF, coordinate, z), (BED_HALF, coordinate, z)),
                         ((coordinate, -BED_HALF, z), (coordinate, BED_HALF, z))):
                lines.InsertNextCell(2)
                lines.InsertCellPoint(points.InsertNextPoint(*a))
                lines.InsertCellPoint(points.InsertNextPoint(*b))
        grid = vtk.vtkPolyData()
        grid.SetPoints(points)
        grid.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(grid)
        self.bed_grid = vtk.vtkActor()
        self.bed_grid.SetMapper(mapper)
        self.bed_grid.GetProperty().SetColor(0.30, 0.36, 0.44)
        self.bed_grid.PickableOff()
        self.bed_grid.VisibilityOff()
        self.renderer.AddActor(self.bed_grid)

        self.bed_edges = {}
        for side, a, b in (
            ("X−", (-BED_HALF, -BED_HALF), (-BED_HALF, BED_HALF)),
            ("X+", (BED_HALF, -BED_HALF), (BED_HALF, BED_HALF)),
            ("Y−", (-BED_HALF, -BED_HALF), (BED_HALF, -BED_HALF)),
            ("Y+", (-BED_HALF, BED_HALF), (BED_HALF, BED_HALF)),
        ):
            line = vtk.vtkLineSource()
            line.SetPoint1(*a, z)
            line.SetPoint2(*b, z)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())
            edge = vtk.vtkActor()
            edge.SetMapper(mapper)
            edge.GetProperty().SetColor(0.67, 0.76, 0.87)
            edge.GetProperty().SetLineWidth(2)
            edge.PickableOff()
            edge.VisibilityOff()
            self.renderer.AddActor(edge)
            self.bed_edges[side] = edge

    def add_playback_controls(self, workspace, canvas):
        self.layer_rail = QWidget()
        self.layer_rail.setMinimumWidth(76)
        layer_layout = QVBoxLayout(self.layer_rail)
        layer_layout.setContentsMargins(5, 5, 5, 5)
        layer_layout.addWidget(QLabel("层号"), alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layer_number = QSpinBox()
        self.layer_number.setMinimumWidth(66)
        self.layer_number.setStyleSheet(
            "QSpinBox::up-button { subcontrol-position: top right; width: 12px; }"
            "QSpinBox::down-button { subcontrol-position: bottom right; width: 12px; }"
        )
        self.layer_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layer_number.setToolTip("输入层号后按回车，也可以用上下方向键切层")
        self.layer_number.setKeyboardTracking(False)
        self.layer_number.valueChanged.connect(self.seek_layer)
        layer_layout.addWidget(self.layer_number)
        self.layer_total = QLabel()
        layer_layout.addWidget(self.layer_total, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layer_slider = QSlider(Qt.Orientation.Vertical)
        self.layer_slider.setToolTip("选择层号")
        self.layer_slider.valueChanged.connect(self.seek_layer)
        layer_layout.addWidget(self.layer_slider, 1, Qt.AlignmentFlag.AlignHCenter)
        canvas.addWidget(self.layer_rail)

        self.timeline = QWidget()
        controls = QHBoxLayout(self.timeline)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)
        self.speed = QComboBox()
        self.speed.setToolTip("播放速度")
        for multiplier in (0.25, 0.5, 1, 2, 4):
            self.speed.addItem(f"{multiplier:g}×", multiplier)
        self.speed.setCurrentIndex(2)
        controls.addWidget(self.speed)
        controls.addWidget(QLabel("层内路径"))
        self.path_slider = QSlider(Qt.Orientation.Horizontal)
        self.path_slider.valueChanged.connect(self.seek_path)
        controls.addWidget(self.path_slider, 1)
        self.segment_label = QLabel()
        controls.addWidget(self.segment_label)
        workspace.addWidget(self.timeline)
        self.layer_rail.hide()
        self.timeline.hide()

    def add_support_panel(self):
        dock = QDockWidget("模型工作区", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.tabs = QTabWidget(dock)
        self.tabs.setMinimumWidth(245)
        self.settings_panel = QWidget(self.tabs)
        self.tabs.addTab(self.settings_panel, "模型与切片")
        layout = QVBoxLayout(self.settings_panel)

        fields = QFormLayout()
        self.style_choice = QComboBox()
        for label, value in (("有机树状", "organic"), ("网格", "grid"), ("贴合", "snug")):
            self.style_choice.addItem(label, value)
        self.print_height = QDoubleSpinBox()
        self.print_height.setRange(0.01, 1000000)
        self.print_height.setValue(DEMO_HEIGHT_MM)
        self.print_height.setSuffix(" mm")
        self.print_height.setKeyboardTracking(False)
        self.print_height.setEnabled(False)
        self.up_axis = QComboBox()
        self.up_axis.addItem("Z 轴", 0)
        self.up_axis.addItem("Y 轴", 90)
        self.up_axis.setToolTip("选择源模型的竖直轴，切片时统一转为 Z 轴朝上。")
        self.up_axis.setEnabled(False)
        self.angle = QSpinBox()
        self.angle.setRange(0, 89)
        self.angle.setSpecialValueText("自动")
        self.angle.setValue(45)
        self.angle.setSuffix("°")
        self.angle.setToolTip("朝下表面与水平面的夹角小于此值时，可能生成支撑。0° 交由 PrusaSlicer 自动判断。")
        self.angle.setKeyboardTracking(False)
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0, 2)
        self.gap.setDecimals(2)
        self.gap.setSingleStep(0.05)
        self.gap.setValue(0.20)
        self.gap.setSuffix(" mm")
        self.gap.setKeyboardTracking(False)
        self.fill_density = QSpinBox()
        self.fill_density.setRange(0, 100)
        self.fill_density.setValue(10)
        self.fill_density.setSuffix(" %")
        self.fill_density.setKeyboardTracking(False)
        self.fill_pattern = QComboBox()
        for label, pattern in (("切片器默认", None), ("直线", "rectilinear"),
                               ("网格", "grid"), ("Gyroid", "gyroid")):
            self.fill_pattern.addItem(label, pattern)
        self.fill_pattern.setToolTip("100% 填充时使用直线。")
        fields.addRow("源模型竖直轴", self.up_axis)
        fields.addRow("模型高度", self.print_height)
        fields.addRow("填充密度", self.fill_density)
        fields.addRow("填充图案", self.fill_pattern)
        fields.addRow("支撑样式", self.style_choice)
        fields.addRow("悬垂阈值", self.angle)
        fields.addRow("接触间隙", self.gap)
        layout.addLayout(fields)

        self.generate_button = QPushButton("切片并预览")
        self.generate_button.setEnabled(False)
        self.auto_play_preview = QCheckBox("切片完成后自动播放")
        self.cancel_button = QPushButton("取消计算")
        self.cancel_button.setEnabled(False)
        self.dimensions = QLabel("模型尺寸  —")
        self.raw_dimensions = QLabel()
        self.raw_dimensions.setWordWrap(True)
        self.platform_status = QLabel("打印平台 220 × 220 mm")
        self.platform_status.setWordWrap(True)
        self.frame_bed_button = QPushButton("查看整个平台")
        self.frame_bed_button.setEnabled(False)
        self.frame_bed_button.clicked.connect(self.frame_build_plate)
        self.support_summary = QLabel(
            "打开模型后可切片" if self.prusa_available else "未找到 PrusaSlicer 或切片模块"
        )
        self.support_summary.setWordWrap(True)
        profile = QLabel("演示配置 · 仅从平台支撑\n0.4 mm 喷嘴 · 0.3 mm 层高")
        for control in (self.raw_dimensions, self.dimensions, self.platform_status,
                        self.frame_bed_button, self.auto_play_preview,
                        self.generate_button, self.cancel_button,
                        self.support_summary, profile):
            layout.addWidget(control)
        layout.addStretch()
        self.add_playback_tab()
        dock.setWidget(self.tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.style_choice.currentIndexChanged.connect(self.invalidate_support)
        self.up_axis.currentIndexChanged.connect(self.change_up_axis)
        self.print_height.valueChanged.connect(self.resize_model)
        self.angle.valueChanged.connect(self.invalidate_support)
        self.gap.valueChanged.connect(self.invalidate_support)
        self.fill_density.valueChanged.connect(self.invalidate_support)
        self.fill_pattern.currentIndexChanged.connect(self.invalidate_support)
        self.generate_button.clicked.connect(self.toggle_preview)
        self.cancel_button.clicked.connect(self.cancel_support)
        self.tabs.currentChanged.connect(self.on_workspace_changed)

    def add_playback_tab(self):
        self.playback_panel = QWidget(self.tabs)
        self.tabs.addTab(self.playback_panel, "打印预览")
        layout = QVBoxLayout(self.playback_panel)
        self.return_button = QPushButton("返回模型")
        self.return_button.clicked.connect(self.show_model)
        layout.addWidget(self.return_button)
        self.layer_mode = QComboBox()
        self.layer_mode.addItem("累计到当前层", False)
        self.layer_mode.addItem("仅当前层", True)
        self.layer_mode.currentIndexChanged.connect(self.change_layer_mode)
        mode_form = QFormLayout()
        mode_form.addRow("显示范围", self.layer_mode)
        layout.addLayout(mode_form)

        self.layer_label = QLabel()
        self.layer_label.hide()
        self.statusBar().addPermanentWidget(self.layer_label)
        self.path_toggles = {}
        for role, (label, _) in PATH_STYLES.items():
            toggle = QCheckBox(label)
            toggle.setChecked(True)
            toggle.toggled.connect(self.update_visibility)
            layout.addWidget(toggle)
            self.path_toggles[role] = toggle
        self.model_overlay = QCheckBox("叠加半透明原模型")
        self.model_overlay.toggled.connect(self.update_visibility)
        layout.addWidget(self.model_overlay)
        layout.addStretch()
        self.set_playback_available(False)

    def set_playback_available(self, available):
        self.tabs.setTabEnabled(self.tabs.indexOf(self.playback_panel), available)

    def in_preview(self):
        return bool(self.layers) and self.tabs.currentWidget() is self.playback_panel

    def show_model(self):
        self.tabs.setCurrentWidget(self.settings_panel)

    def toggle_preview(self):
        if self.in_preview():
            self.show_model()
        elif self.layers:
            self.tabs.setCurrentWidget(self.playback_panel)
        else:
            self.show_model()
            self.generate_support()

    def on_workspace_changed(self):
        if not self.in_preview():
            self.pause_print()
        self.update_visibility()

    def update_preview_action(self):
        ready = bool(self.layers)
        can_slice = self.prusa_available and self.platform_fits and self.future is None
        if self.future is not None:
            label = "正在切片…"
        elif ready:
            label = "查看打印预览"
        elif self.reslice_required:
            label = "重新切片"
        else:
            label = "切片并预览"
        self.generate_button.setText(label)
        self.generate_button.setEnabled(ready or can_slice)
        self.preview_action.setText("返回模型" if self.in_preview() else label)
        self.preview_action.setEnabled(ready or can_slice)

    def update_visibility(self):
        preview = self.in_preview()
        for role, actor in self.path_actors.items():
            actor.SetVisibility(preview and self.path_toggles[role].isChecked())
        self.actor.SetVisibility(not preview or self.model_overlay.isChecked())
        self.actor.GetProperty().SetOpacity(0.18 if preview else 1)
        self.layer_label.setVisible(preview)
        self.layer_rail.setVisible(preview)
        self.timeline.setVisible(preview)
        show_bed = self.current_path is not None and not preview
        self.bed_grid.SetVisibility(show_bed)
        for edge in self.bed_edges.values():
            edge.SetVisibility(show_bed)
        self.update_preview_action()
        self.viewport.update()

    def layer_limits(self, layer):
        start = self.layer_ends[layer - 2] if layer > 1 else 0
        return start, self.layer_ends[layer - 1]

    def show_progress(self, count, layer):
        self.active_layer = layer
        start, end = self.layer_limits(layer)
        first = start if self.layer_mode.currentData() else 0
        for track in self.tracks.values():
            track.show_range(first, count)
        with QSignalBlocker(self.layer_slider), QSignalBlocker(self.layer_number):
            self.layer_slider.setValue(layer)
            self.layer_number.setValue(layer)
        with QSignalBlocker(self.path_slider):
            self.path_slider.setRange(0, end - start)
            self.path_slider.setValue(count - start)
        height = self.layers[layer - 1][0]
        self.layer_label.setText(f"第 {layer}/{len(self.layers)} 层 · Z {height:g} mm")
        self.segment_label.setText(f"{count - start:,}/{end - start:,} 段")
        self.viewport.update()

    def seek_layer(self, layer):
        self.pause_print()
        if self.layers:
            self.play_position = self.layer_ends[layer - 1]
            self.show_progress(self.play_position, layer)

    def seek_path(self, steps):
        self.pause_print()
        if self.layers:
            start, _ = self.layer_limits(self.active_layer)
            self.play_position = start + steps
            self.show_progress(self.play_position, self.active_layer)

    def change_layer_mode(self):
        self.pause_print()
        if self.layers:
            self.show_progress(int(self.play_position), self.active_layer)

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.pause_print()
        elif self.in_preview():
            if self.layer_mode.currentData():
                start, self.play_limit = self.layer_limits(self.active_layer)
                if self.play_position >= self.play_limit:
                    self.play_position = start
            else:
                self.play_limit = self.layer_ends[-1]
                if self.play_position >= self.play_limit:
                    self.play_position = 0
                    self.active_layer = 1
            self.show_progress(int(self.play_position), self.active_layer)
            self.play_last = perf_counter()
            self.play_timer.start()
            self.play_button.setText("暂停")

    def pause_print(self):
        self.play_timer.stop()
        self.play_button.setText("播放")

    def advance_print(self):
        now = perf_counter()
        if self.layer_mode.currentData():
            start, end = self.layer_limits(self.active_layer)
            rate = (end - start) / 3
        else:
            rate = self.layer_ends[-1] / PLAYBACK_SECONDS
        self.play_position = min(
            self.play_limit, self.play_position + (now - self.play_last) * rate * self.speed.currentData()
        )
        self.play_last = now
        count = int(self.play_position)
        layer = max(self.active_layer, bisect_left(self.layer_ends, count) + 1)
        self.show_progress(count, layer)
        if self.play_position >= self.play_limit:
            self.pause_print()

    def invalidate_support(self):
        self.pause_print()
        if self.layers or self.future is not None:
            self.reslice_required = True
        self.layers = []
        self.layer_ends = []
        for actor in self.path_actors.values():
            actor.GetMapper().SetInputData(None)
        self.tracks = {}
        self.layer_label.clear()
        if self.tabs.currentWidget() is self.playback_panel:
            self.show_model()
        self.set_playback_available(False)
        self.update_visibility()
        if self.future is not None:
            self.cancel_support()
        elif self.current_path is not None and self.prusa_available:
            self.support_summary.setText(
                "设置已更新，请重新切片。" if self.reslice_required else "模型已就绪。"
            )

    def generate_support(self):
        if not self.prusa_available or not self.platform_fits or self.future is not None:
            return

        mesh = printable_mesh(self.mapper.GetInput(), self.actor)
        self.cancel_event = Event()
        self.autoplay = self.auto_play_preview.isChecked()
        self.future = self.pool.submit(
            slice_support, mesh, self.angle.value(), self.gap.value(),
            self.style_choice.currentData(), self.cancel_event,
            density=self.fill_density.value(), pattern=self.fill_pattern.currentData(),
        )
        self.update_preview_action()
        self.cancel_button.setEnabled(True)
        self.support_summary.setText("PrusaSlicer 正在切片…")
        self.poll_timer.start()

    def cancel_support(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.support_summary.setText("正在取消切片…")

    def poll_support(self):
        if not self.future.done():
            return
        future, canceled = self.future, self.cancel_event.is_set()
        self.poll_timer.stop()
        self.future = self.cancel_event = None
        self.cancel_button.setEnabled(False)
        self.update_preview_action()
        if canceled:
            self.support_summary.setText(
                "设置已更新，请重新切片。" if self.reslice_required else "切片已取消。"
            )
            return

        try:
            result = future.result()
        except Exception as error:
            self.support_summary.setText("切片失败")
            QMessageBox.warning(self, "PrusaSlicer 切片失败", str(error))
            return

        if not result["layers"]:
            self.support_summary.setText("切片完成，未找到可预览的挤出路径")
            return
        self.layers = result["layers"]
        self.layer_ends = [end for _, end in self.layers]
        self.tracks = result["tracks"]
        self.reslice_required = False
        for role, track in self.tracks.items():
            self.path_actors[role].GetMapper().SetInputData(track.data)
        with QSignalBlocker(self.layer_slider), QSignalBlocker(self.layer_number):
            self.layer_slider.setRange(1, len(self.layers))
            self.layer_number.setRange(1, len(self.layers))
        self.layer_total.setText(f"共 {len(self.layers)} 层")
        self.seek_layer(len(self.layers))
        self.set_playback_available(True)
        if self.tabs.currentWidget() is self.settings_panel:
            self.tabs.setCurrentWidget(self.playback_panel)
        self.update_visibility()
        self.support_summary.setText(
            f"挤出路径 {self.layer_ends[-1]:,} 段 · {result['seconds']:.1f} 秒\n"
            f"内部填充 {len(self.tracks['infill'].indices):,} 段\n"
            f"支撑 {result['support_count']:,} 段 · {result['path_length'] / 1000:.2f} m\n"
            f"支撑材料约 {result['material_volume']:.0f} mm³"
        )
        if self.autoplay and self.in_preview():
            self.toggle_playback()

    def resize_model(self):
        scale = self.print_height.value() / self.raw_height if self.raw_height > 0 else 1
        self.actor.SetScale(scale, scale, scale)
        self.actor.SetPosition(*(-coordinate * scale for coordinate in self.raw_origin))
        xmin, xmax, ymin, ymax, zmin, zmax = self.actor.GetBounds()
        self.dimensions.setText(
            f"打印尺寸 {xmax - xmin:.2f} × {ymax - ymin:.2f} × {zmax - zmin:.2f} mm"
        )
        self.update_platform_fit()
        self.invalidate_support()
        self.reset_view()

    def update_platform_fit(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.actor.GetBounds()
        distances = (("X−", -BED_HALF - xmin), ("X+", xmax - BED_HALF),
                     ("Y−", -BED_HALF - ymin), ("Y+", ymax - BED_HALF))
        overflow = {side: distance for side, distance in distances if distance > 1e-6}
        for side, edge in self.bed_edges.items():
            color = (1.0, 0.25, 0.20) if side in overflow else (0.67, 0.76, 0.87)
            edge.GetProperty().SetColor(*color)

        problems = []
        if overflow:
            directions = "、".join(f"{side} {distance:.2f} mm" for side, distance in overflow.items())
            problems.append(f"超出平台 · {directions}")
        if zmax - zmin < MIN_PRINT_HEIGHT - 1e-6:
            problems.append(f"模型高度至少需要 {MIN_PRINT_HEIGHT:g} mm")
        self.platform_fits = not problems
        self.platform_status.setText(
            "\n".join(problems) if problems else "模型在 220 × 220 mm 平台内"
        )
        self.platform_status.setStyleSheet("" if self.platform_fits else "color: #F36554")

    def place_model(self, scale=1, *, height=None):
        self.actor.SetOrientation(self.up_axis.currentData(), 0, 0)
        self.actor.SetScale(1, 1, 1)
        self.actor.SetPosition(0, 0, 0)
        xmin, xmax, ymin, ymax, zmin, zmax = self.actor.GetBounds()
        self.raw_origin = ((xmin + xmax) / 2, (ymin + ymax) / 2, zmin)
        self.raw_height = zmax - zmin
        self.raw_dimensions.setText(
            f"源尺寸 {xmax - xmin:.2f} × {ymax - ymin:.2f} × {self.raw_height:.2f}"
        )
        with QSignalBlocker(self.print_height):
            self.print_height.setValue(self.raw_height * scale if height is None else height)
        self.print_height.setEnabled(self.raw_height > 0)
        self.resize_model()

    def change_up_axis(self):
        self.place_model(self.actor.GetScale()[0])

    def closeEvent(self, event):
        self.pause_print()
        if self.cancel_event is not None:
            self.cancel_event.set()
        self.poll_timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.viewport.close()
        super().closeEvent(event)

    def open_model(self):
        directory = self.current_path.parent if self.current_path else Path(__file__).parent
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开 3D 模型", str(directory), "3D 模型 (*.obj *.stl)"
        )
        if filename:
            self.load_model(Path(filename))

    def load_model(self, path):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mesh = read_mesh(path)
        except (OSError, ValueError) as error:
            message = str(error)
        else:
            message = None
        finally:
            QApplication.restoreOverrideCursor()

        if message is not None:
            QMessageBox.warning(self, "打开模型失败", f"{path.name}\n{message}")
            return

        cat = path.name.casefold() == DEFAULT_MODEL.name.casefold()
        spider_man = path.stem.partition(" (")[0].replace("+", " ").casefold() == SPIDER_MAN_NAME
        y_up = cat or (spider_man and path.suffix.lower() == ".obj")
        self.display_mesh(mesh, path, y_up=y_up,
                          height=DEMO_HEIGHT_MM if cat or spider_man else None)

    def display_mesh(self, mesh, path, *, y_up=False, height=None):
        self.mapper.SetInputData(mesh)
        with QSignalBlocker(self.up_axis):
            self.up_axis.setCurrentIndex(1 if y_up else 0)
        self.up_axis.setEnabled(True)
        if self.current_path is None:
            self.renderer.AddActor(self.actor)
        self.current_path = path
        self.frame_bed_button.setEnabled(True)
        self.reset_action.setEnabled(True)
        self.place_model(height=height)
        self.reslice_required = False
        if self.prusa_available and self.future is None:
            self.support_summary.setText("模型已就绪。")
        self.show_model()
        self.update_preview_action()
        self.statusBar().showMessage(f"{path.name}    ·    左键拖动旋转，滚轮缩放")

    def reset_view(self):
        self.frame_view(self.actor.GetBounds())

    def frame_build_plate(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.actor.GetBounds()
        self.frame_view((min(xmin, -BED_HALF), max(xmax, BED_HALF),
                         min(ymin, -BED_HALF), max(ymax, BED_HALF), min(zmin, -0.05), zmax))

    def frame_view(self, bounds):
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0, 0, 0)
        camera.SetPosition(1, -2, 1.2)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera(bounds)
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()


def main():
    app = QApplication(sys.argv)
    window = ModelViewer()
    window.show()
    window.viewport.Initialize()
    if DEFAULT_MODEL.is_file():
        window.load_model(DEFAULT_MODEL)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
