from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from math import ceil, floor
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cv2
from PyQt6.QtCore import QEvent, QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QLabel, QMessageBox, QPushButton, QRubberBand, QToolBar, QVBoxLayout,
    QWidget,
)
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from image_relief import make_relief, preview_rgb, read_image


PROJECT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("print_viewer", PROJECT / "1110" / "model_viewer.py")
viewer = module_from_spec(spec)
spec.loader.exec_module(viewer)

HEIGHT_ARRAY = "ModelHeight"
HEIGHT_PALETTES = {
    "地形渐变": ("#235D43", "#73A64F", "#B69855", "#91776C", "#FAF5EC"),
    "蓝橙渐变": ("#2356AE", "#53C1DB", "#FFF0AC", "#EB651F"),
}


class CropPreview(QLabel):
    crop_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(150)
        self.band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.anchor = None
        self.selection = None

    def image_rect(self):
        area = self.contentsRect()
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return QRect()
        size = pixmap.size()
        return QRect(area.x() + (area.width() - size.width()) // 2,
                     area.y() + (area.height() - size.height()) // 2,
                     size.width(), size.height())

    def clear_crop(self):
        self.anchor = self.selection = None
        self.band.hide()
        self.crop_changed.emit()

    def mousePressEvent(self, event):
        point = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(point):
            self.anchor = point
            self.band.setGeometry(QRect(point, point))
            self.band.show()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.anchor is not None:
            rectangle = QRect(self.anchor, event.position().toPoint()).normalized()
            self.band.setGeometry(rectangle.intersected(self.image_rect()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.anchor is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        area = self.image_rect()
        drag = QRect(self.anchor, event.position().toPoint()).normalized().intersected(area)
        self.anchor = None
        if drag.width() < 3 or drag.height() < 3:
            self.clear_crop()
            return
        self.selection = (
            (drag.x() - area.x()) / area.width(), (drag.y() - area.y()) / area.height(),
            (drag.x() + drag.width() - area.x()) / area.width(),
            (drag.y() + drag.height() - area.y()) / area.height(),
        )
        self.show_selection()
        self.crop_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_selection()

    def show_selection(self):
        if self.selection is None:
            return
        area = self.image_rect()
        left, top, right, bottom = self.selection
        self.band.setGeometry(
            QRect(QPoint(area.x() + round(left * area.width()), area.y() + round(top * area.height())),
                  QPoint(area.x() + round(right * area.width()) - 1,
                         area.y() + round(bottom * area.height()) - 1))
        )
        self.band.show()

    def crop(self, image):
        if self.selection is None:
            return image
        rows, cols = image.shape[:2]
        left, top, right, bottom = self.selection
        return image[floor(top * rows):ceil(bottom * rows), floor(left * cols):ceil(right * cols)]


class ImageReliefViewer(viewer.ModelViewer):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D 打印查看器 · 图片浮雕")
        self.image_path = None
        self.image = None
        self.relief_future = None
        self.skip_result = False
        self.relief_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Relief")
        self.relief_timer = QTimer(self)
        self.relief_timer.setInterval(100)
        self.relief_timer.timeout.connect(self.finish_relief)

        self.relief_panel = QWidget(self.tabs)
        layout = QVBoxLayout(self.relief_panel)
        self.source_label = QLabel("请选择 JPG 或 PNG 图片")
        self.source_label.setWordWrap(True)
        self.thumbnail = CropPreview()
        self.thumbnail.crop_changed.connect(self.update_crop_status)
        layout.addWidget(self.source_label)
        layout.addWidget(self.thumbnail)
        self.crop_status = QLabel("在图片上拖动以框选")
        self.clear_crop_button = QPushButton("使用整张图片")
        self.clear_crop_button.setEnabled(False)
        self.clear_crop_button.clicked.connect(self.thumbnail.clear_crop)
        layout.addWidget(self.crop_status)
        layout.addWidget(self.clear_crop_button)

        self.mode = QComboBox()
        self.mode.addItems(("照片 · 估计深度", "深度图 · 灰度转高度"))
        fields = QFormLayout()
        fields.addRow("来源类型", self.mode)
        self.width = self.add_size(fields, "浮雕宽度", 100)
        self.rise = self.add_size(fields, "起伏高度", 8)
        self.base = self.add_size(fields, "底板厚度", 2)
        self.invert = QCheckBox("反转凹凸")
        layout.addLayout(fields)
        layout.addWidget(self.invert)
        self.relief_button = QPushButton("生成浮雕")
        self.relief_button.setEnabled(False)
        self.relief_button.clicked.connect(self.generate_relief)
        layout.addWidget(self.relief_button)
        self.relief_status = QLabel()
        self.relief_status.setWordWrap(True)
        layout.addWidget(self.relief_status)
        layout.addStretch()

        toolbar = self.findChild(QToolBar)
        file_menu = self.menuBar().actions()[0].menu()
        self.image_action = QAction("导入图片", self)
        self.image_action.setShortcut("Ctrl+I")
        self.image_action.triggered.connect(self.import_image)
        toolbar.insertAction(self.reset_action, self.image_action)
        file_menu.insertAction(file_menu.actions()[-2], self.image_action)
        self.export_action = QAction("导出 STL", self)
        self.export_action.triggered.connect(self.export_stl)
        self.export_action.setEnabled(False)
        toolbar.insertAction(self.reset_action, self.export_action)
        file_menu.insertAction(file_menu.actions()[-2], self.export_action)
        toolbar.insertSeparator(self.reset_action)

        self.color_mode = QComboBox()
        self.color_mode.setToolTip("模型配色")
        self.color_mode.addItems(("单色", *HEIGHT_PALETTES))
        self.color_mode.currentTextChanged.connect(self.update_model_color)
        color_label = QLabel("模型配色")
        color_label.setObjectName("toolbarLabel")
        toolbar.addWidget(color_label)
        toolbar.addWidget(self.color_mode)
        self.color_action = toolbar.addAction("选颜色")
        self.color_action.triggered.connect(self.choose_color)
        toolbar.addSeparator()
        toolbar.addAction(self.preview_action)

        self.color_scale = vtk.vtkColorTransferFunction()
        self.height_legend = vtk.vtkScalarBarActor()
        self.height_legend.SetLookupTable(self.color_scale)
        self.height_legend.SetTitle("Z (mm)")
        self.height_legend.SetLabelFormat("%.3g")
        self.height_legend.SetNumberOfLabels(4)
        self.height_legend.SetMaximumNumberOfColors(256)
        self.height_legend.UnconstrainedFontSizeOn()
        self.height_legend.DrawBackgroundOn()
        self.height_legend.GetBackgroundProperty().SetColor(0.07, 0.08, 0.11)
        self.height_legend.GetBackgroundProperty().SetOpacity(0.86)
        for text, size in ((self.height_legend.GetTitleTextProperty(), 11),
                           (self.height_legend.GetLabelTextProperty(), 10)):
            text.SetFontFamilyToArial()
            text.SetFontSize(size)
            text.SetColor(0.86, 0.90, 0.95)
            text.BoldOff()
            text.ItalicOff()
            text.ShadowOff()
        self.height_legend.SetVisibility(False)
        self.renderer.AddViewProp(self.height_legend)
        self.place_height_legend(self.viewport.size())
        self.viewport.installEventFilter(self)
        self.tabs.insertTab(0, self.relief_panel, "图片浮雕")
        self.apply_theme()

    def apply_theme(self):
        for page in (self.settings_panel, self.playback_panel, self.relief_panel):
            page.setObjectName("workspacePage")
            page.layout().setContentsMargins(10, 10, 10, 10)
            page.layout().setSpacing(5)
        for widget in (self.layer_rail, self.timeline):
            widget.setObjectName("playbackControls")
        self.layer_rail.setMaximumWidth(78)
        for button in (self.generate_button, self.relief_button, self.play_button):
            button.setObjectName("primary")
        self.renderer.SetBackground(0.075, 0.086, 0.112)
        self.renderer.SetBackground2(0.17, 0.20, 0.25)
        self.bed_grid.GetProperty().SetColor(0.26, 0.31, 0.39)
        style = Path(__file__).with_name("viewer_dark.qss")
        self.setStyleSheet(style.read_text(encoding="utf-8"))

    def eventFilter(self, watched, event):
        if watched is self.viewport and event.type() == QEvent.Type.Resize:
            self.place_height_legend(event.size())
        return super().eventFilter(watched, event)

    def place_height_legend(self, size):
        width, height = max(1, size.width()), max(1, size.height())
        legend_width = min(86, width * 0.22)
        legend_height = min(172, height * 0.36)
        self.height_legend.SetPosition(1 - (legend_width + 12) / width, 12 / height)
        self.height_legend.SetWidth(legend_width / width)
        self.height_legend.SetHeight(legend_height / height)

    def update_crop_status(self):
        has_crop = self.thumbnail.selection is not None
        self.clear_crop_button.setEnabled(has_crop)
        self.crop_status.setText("在图片上拖动以框选")
        if has_crop:
            rows, cols = self.thumbnail.crop(self.image).shape[:2]
            self.crop_status.setText(f"已框选 {cols} × {rows} 像素")
        if self.relief_future is not None:
            self.skip_result = True
            self.relief_status.setText("选区已更新，请重新生成浮雕。")
        elif self.current_path == self.image_path and self.image_path is not None:
            self.relief_status.setText("要应用新选区，请重新生成浮雕。")

    def choose_color(self):
        color = QColorDialog.getColor(
            QColor.fromRgbF(*self.actor.GetProperty().GetColor()), self, "选择模型颜色"
        )
        if color.isValid():
            self.actor.GetProperty().SetColor(*color.getRgbF()[:3])
            self.color_mode.setCurrentText("单色")
            self.viewport.update()

    def update_model_color(self):
        palette = HEIGHT_PALETTES.get(self.color_mode.currentText())
        self.color_action.setEnabled(palette is None)
        mesh = self.mapper.GetInput()
        self.mapper.ScalarVisibilityOff()
        if palette is not None and mesh is not None:
            zmin, zmax = self.actor.GetBounds()[4:]
            if zmax > zmin:
                points = vtk_to_numpy(mesh.GetPoints().GetData())
                axis = 1 if self.up_axis.currentData() == 90 else 2
                matrix = self.actor.GetMatrix()
                heights = points[:, axis].astype(float) * matrix.GetElement(2, axis)
                heights += matrix.GetElement(2, 3)
                scalars = numpy_to_vtk(heights, deep=True)
                scalars.SetName(HEIGHT_ARRAY)
                mesh.GetPointData().AddArray(scalars)
                self.color_scale.RemoveAllPoints()
                for index, color in enumerate(palette):
                    level = zmin + (zmax - zmin) * index / (len(palette) - 1)
                    self.color_scale.AddRGBPoint(level, *QColor(color).getRgbF()[:3])
                self.mapper.SetLookupTable(self.color_scale)
                self.mapper.UseLookupTableScalarRangeOn()
                self.mapper.SetScalarModeToUsePointFieldData()
                self.mapper.SelectColorArray(HEIGHT_ARRAY)
                self.mapper.SetColorModeToMapScalars()
                self.mapper.InterpolateScalarsBeforeMappingOn()
                self.mapper.ScalarVisibilityOn()
        self.update_visibility()

    def update_visibility(self):
        super().update_visibility()
        self.height_legend.SetVisibility(
            self.mapper.GetScalarVisibility() and self.actor.GetVisibility()
        )

    def resize_model(self):
        super().resize_model()
        self.update_model_color()

    @staticmethod
    def add_size(form, label, value):
        control = QDoubleSpinBox()
        control.setRange(0.1, 300)
        control.setValue(value)
        control.setSuffix(" mm")
        form.addRow(label, control)
        return control

    def import_image(self):
        directory = self.image_path.parent if self.image_path else Path(__file__).parent
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入图片", str(directory), "图片 (*.jpg *.jpeg *.png)"
        )
        if filename:
            self.load_image(Path(filename))

    def load_image(self, path):
        try:
            image = read_image(path)
            height, width = image.shape[:2]
            ratio = min(220 / width, 150 / height, 1)
            reduced = cv2.resize(image, (max(1, round(width * ratio)),
                                         max(1, round(height * ratio))), interpolation=cv2.INTER_AREA)
            rgb = preview_rgb(reduced)
        except (OSError, ValueError, cv2.error) as error:
            QMessageBox.warning(self, "导入失败", str(error))
            return
        if self.relief_future is not None:
            self.skip_result = True
        self.image = image
        self.image_path = path
        preview = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                         QImage.Format.Format_RGB888)
        self.thumbnail.setPixmap(QPixmap.fromImage(preview))
        self.thumbnail.clear_crop()
        self.source_label.setText(f"{path.name}\n{width} × {height} 像素")
        self.mode.setCurrentIndex(int(path.stem.casefold().endswith("_depth")))
        self.relief_status.setText("图片已导入")
        self.relief_button.setEnabled(self.relief_future is None)
        self.tabs.setCurrentWidget(self.relief_panel)

    def generate_relief(self):
        if self.image is None or self.relief_future is not None:
            return
        self.pause_print()
        self.skip_result = False
        photo = self.mode.currentIndex() == 0
        self.relief_button.setEnabled(False)
        self.relief_status.setText("正在本地估计深度并生成浮雕…" if photo else "正在生成浮雕…")
        self.relief_future = self.relief_pool.submit(
            make_relief, self.thumbnail.crop(self.image), photo, self.width.value(), self.rise.value(),
            self.base.value(), self.invert.isChecked()
        )
        self.relief_timer.start()

    def finish_relief(self):
        if not self.relief_future.done():
            return
        future = self.relief_future
        self.relief_future = None
        self.relief_timer.stop()
        self.relief_button.setEnabled(True)
        if self.skip_result:
            return
        try:
            mesh = future.result()
        except Exception as error:
            self.relief_status.setText("生成失败。已有深度图可使用灰度转高度模式。")
            QMessageBox.warning(self, "生成浮雕失败", str(error))
            return
        self.display_mesh(mesh, self.image_path)
        self.relief_status.setText(f"浮雕已生成 · {mesh.GetNumberOfPolys():,} 个三角面")

    def display_mesh(self, mesh, path, *, y_up=False, height=None):
        if self.relief_future is not None:
            self.skip_result = True
            self.relief_status.setText("已切换模型，可重新生成浮雕。")
        super().display_mesh(mesh, path, y_up=y_up, height=height)
        self.export_action.setEnabled(True)

    def export_stl(self):
        suggested = self.current_path.with_name(f"{self.current_path.stem}_relief.stl")
        dialog = QFileDialog(self, "导出 STL", str(suggested), "STL (*.stl)")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("stl")
        if not dialog.exec():
            return
        path = Path(dialog.selectedFiles()[0])
        if path.suffix.lower() != ".stl":
            QMessageBox.warning(self, "导出失败", "请使用 .stl 文件扩展名。")
            return
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".stl", delete=False) as handle:
                temporary = Path(handle.name)
            mesh = viewer.printable_mesh(self.mapper.GetInput(), self.actor)
            writer = vtk.vtkSTLWriter()
            writer.SetFileTypeToBinary()
            writer.SetInputData(mesh)
            writer.SetFileName(str(temporary))
            if writer.Write() != 1 or writer.GetErrorCode() or temporary.stat().st_size <= 84:
                raise OSError("写入 STL 失败。")
            temporary.replace(path)
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))
            return
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.statusBar().showMessage(f"已保存 {path.name} · 可通过「打开 3D 模型」重新载入")

    def closeEvent(self, event):
        self.relief_timer.stop()
        self.relief_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = ImageReliefViewer()
    window.show()
    window.viewport.Initialize()
    if viewer.DEFAULT_MODEL.is_file():
        window.load_model(viewer.DEFAULT_MODEL)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
