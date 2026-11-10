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
    QFormLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox,
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
    "infill": ("蓝色填充", (0.28, 0.62, 1.0)),
    "support": ("绿色支撑", (0.12, 0.90, 0.35)),
}
PLAYBACK_SECONDS = 25


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


def slice_support(mesh, angle, gap, style, cancel):
    spec = spec_from_file_location("prusa_support_bridge", BRIDGE_SCRIPT)
    bridge = module_from_spec(spec)
    spec.loader.exec_module(bridge)
    # A separate directory for each job also isolates multiple viewer instances.
    with TemporaryDirectory(prefix="python3dprint-prusa-") as directory:
        job = bridge.prepare_job(mesh, directory, angle, gap, style)
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
        self.shown = 0
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(coordinates, dtype=np.float32).reshape(-1, 3),
                                    deep=True))
        self.lines = vtk.vtkCellArray()
        self.data = vtk.vtkPolyData()
        self.data.SetPoints(points)
        self.data.SetLines(self.lines)

    def show_before(self, index):
        count = int(self.indices.searchsorted(index))
        if count != self.shown:
            # Each extrusion segment uses two consecutive points. Reuse the
            # point buffer while moving the end of the visible connectivity.
            self.lines.SetData(2, numpy_to_vtk(self.connectivity[:2 * count]))
            self.data.Modified()
            self.shown = count


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
        self.cancel_event = None
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self.poll_support)
        self.layers = []
        self.layer_ends = []
        self.tracks = {}
        self.play_position = 0.0
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

        file_menu.addSeparator()
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        self.viewport = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.viewport)
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
        self.add_support_panel()
        self.statusBar().showMessage("尚未加载模型")

    def add_support_panel(self):
        dock = QDockWidget("PrusaSlicer 打印预览", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.tabs = QTabWidget(dock)
        self.tabs.setMinimumWidth(245)
        panel = QWidget(self.tabs)
        self.tabs.addTab(panel, "切片设置")
        layout = QVBoxLayout(panel)

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
        fields.addRow("支撑样式", self.style_choice)
        fields.addRow("源模型竖直轴", self.up_axis)
        fields.addRow("模型高度", self.print_height)
        fields.addRow("悬垂阈值", self.angle)
        fields.addRow("接触间隙", self.gap)
        layout.addLayout(fields)

        self.generate_button = QPushButton("切片并预览")
        self.generate_button.setEnabled(False)
        self.cancel_button = QPushButton("取消计算")
        self.cancel_button.setEnabled(False)
        self.dimensions = QLabel("模型尺寸  —")
        self.raw_dimensions = QLabel()
        self.raw_dimensions.setWordWrap(True)
        self.support_summary = QLabel(
            "打开模型后可切片" if self.prusa_available else "未找到 PrusaSlicer 或切片模块"
        )
        self.support_summary.setWordWrap(True)
        profile = QLabel("演示配置 · 仅从平台支撑\n220 × 220 mm 平台\n0.4 mm 喷嘴 · 0.3 mm 层高")
        for control in (self.generate_button, self.cancel_button, self.raw_dimensions, self.dimensions,
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
        self.generate_button.clicked.connect(self.generate_support)
        self.cancel_button.clicked.connect(self.cancel_support)

    def add_playback_tab(self):
        panel = QWidget(self.tabs)
        self.tabs.addTab(panel, "路径播放")
        layout = QVBoxLayout(panel)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)
        self.speed = QComboBox()
        for multiplier in (0.25, 0.5, 1, 2, 4):
            self.speed.addItem(f"{multiplier:g}×", multiplier)
        self.speed.setCurrentIndex(2)
        speed_form = QFormLayout()
        speed_form.addRow("播放速度", self.speed)
        layout.addLayout(speed_form)

        self.layer_label = QLabel()
        self.statusBar().addPermanentWidget(self.layer_label)
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.valueChanged.connect(self.seek_layer)
        self.segment_label = QLabel("挤出路径  —")
        for control in (QLabel("跳到指定层"), self.layer_slider, self.segment_label):
            layout.addWidget(control)
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
        self.tabs.setTabEnabled(1, False)

    def update_visibility(self):
        for role, actor in self.path_actors.items():
            actor.SetVisibility(bool(self.layers) and self.path_toggles[role].isChecked())
        self.actor.SetVisibility(not self.layers or self.model_overlay.isChecked())
        self.actor.GetProperty().SetOpacity(0.18 if self.layers else 1)
        self.viewport.update()

    def show_progress(self, count):
        for track in self.tracks.values():
            track.show_before(count)
        layer = bisect_left(self.layer_ends, count) + 1 if count else 0
        with QSignalBlocker(self.layer_slider):
            self.layer_slider.setValue(layer)
        height = self.layers[layer - 1][0] if layer else 0
        self.layer_label.setText(f"第 {layer}/{len(self.layers)} 层 · Z {height:g} mm")
        self.segment_label.setText(f"挤出路径 {count:,}/{self.layer_ends[-1]:,} 段")
        self.viewport.update()

    def seek_layer(self, layer):
        self.pause_print()
        if self.layers:
            self.play_position = self.layer_ends[layer - 1] if layer else 0
            self.show_progress(self.play_position)

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.pause_print()
        elif self.layers:
            if self.play_position >= self.layer_ends[-1]:
                self.seek_layer(0)
            self.play_last = perf_counter()
            self.play_timer.start()
            self.play_button.setText("暂停")

    def pause_print(self):
        self.play_timer.stop()
        self.play_button.setText("播放")

    def advance_print(self):
        now = perf_counter()
        total = self.layer_ends[-1]
        rate = total / PLAYBACK_SECONDS * self.speed.currentData()
        self.play_position = min(total, self.play_position + (now - self.play_last) * rate)
        self.play_last = now
        self.show_progress(int(self.play_position))
        if self.play_position >= total:
            self.pause_print()

    def invalidate_support(self):
        self.pause_print()
        self.layers = []
        self.layer_ends = []
        for actor in self.path_actors.values():
            actor.GetMapper().SetInputData(None)
        self.tracks = {}
        self.layer_label.clear()
        self.tabs.setCurrentIndex(0)
        self.tabs.setTabEnabled(1, False)
        self.update_visibility()
        if self.future is not None:
            self.cancel_support()
        elif self.current_path is not None and self.prusa_available:
            self.support_summary.setText("点击「切片并预览」开始切片")

    def generate_support(self):
        if not self.prusa_available or self.current_path is None or self.future is not None:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self.actor.GetBounds()
        if zmax - zmin < 0.3 or max(xmax - xmin, ymax - ymin) > 220:
            self.support_summary.setText("尺寸不适合当前配置。请调整高度，使 XY 尺寸在 220 mm 内且高度至少 0.3 mm。")
            return

        mesh = printable_mesh(self.mapper.GetInput(), self.actor)
        self.cancel_event = Event()
        self.future = self.pool.submit(
            slice_support, mesh, self.angle.value(), self.gap.value(),
            self.style_choice.currentData(), self.cancel_event,
        )
        self.generate_button.setEnabled(False)
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
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if canceled:
            self.support_summary.setText("切片已取消，可重新生成")
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
        for role, track in self.tracks.items():
            self.path_actors[role].GetMapper().SetInputData(track.data)
        with QSignalBlocker(self.layer_slider):
            self.layer_slider.setRange(0, len(self.layers))
        self.seek_layer(len(self.layers))
        self.update_visibility()
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        self.support_summary.setText(
            f"挤出路径 {self.layer_ends[-1]:,} 段 · {result['seconds']:.1f} 秒\n"
            f"支撑 {result['support_count']:,} 段 · {result['path_length'] / 1000:.2f} m\n"
            f"支撑材料约 {result['material_volume']:.0f} mm³"
        )

    def resize_model(self):
        scale = self.print_height.value() / self.raw_height if self.raw_height > 0 else 1
        self.actor.SetScale(scale, scale, scale)
        self.actor.SetPosition(*(-coordinate * scale for coordinate in self.raw_origin))
        xmin, xmax, ymin, ymax, zmin, zmax = self.actor.GetBounds()
        self.dimensions.setText(
            f"打印尺寸 {xmax - xmin:.2f} × {ymax - ymin:.2f} × {zmax - zmin:.2f} mm"
        )
        self.invalidate_support()
        self.reset_view()

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

        self.mapper.SetInputData(mesh)
        cat = path.name.casefold() == DEFAULT_MODEL.name.casefold()
        spider_man = path.stem.partition(" (")[0].replace("+", " ").casefold() == SPIDER_MAN_NAME
        y_up = cat or (spider_man and path.suffix.lower() == ".obj")
        with QSignalBlocker(self.up_axis):
            self.up_axis.setCurrentIndex(1 if y_up else 0)
        self.up_axis.setEnabled(True)
        if self.current_path is None:
            self.renderer.AddActor(self.actor)
        self.current_path = path
        self.generate_button.setEnabled(self.prusa_available and self.future is None)
        self.reset_action.setEnabled(True)
        self.place_model(height=DEMO_HEIGHT_MM if cat or spider_man else None)
        self.statusBar().showMessage(f"{path.name}    ·    左键拖动旋转，滚轮缩放")

    def reset_view(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0, 0, 0)
        camera.SetPosition(1, -2, 1.2)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera(self.actor.GetBounds())
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
