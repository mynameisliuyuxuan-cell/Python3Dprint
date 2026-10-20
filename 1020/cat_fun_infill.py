from pathlib import Path
import math
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vedo import load
from shapely import make_valid
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union


# ===================== 参数 =====================

MODEL_PATH = Path(r"E:\cat\ORIGAMI_Chat_Free.obj")

MODEL_HEIGHT = 80.0
LAYER_HEIGHT = 1.2

LINE_WIDTH = 0.6
WALL_COUNT = 2
INFILL_SPACING = 3.0

SEGMENT_LENGTH = 1.5
SEGMENTS_PER_FRAME = 8
FRAME_MS = 30

ROTATION_SPEED = 2.0       # 每秒旋转角度，数值越小越慢

# 当前猫模型是 Y 轴朝上，转为 Z 轴朝上
ROTATE_X = 90
ROTATE_Y = 0
ROTATE_Z = 0

# GYROID_PERIOD = 12.0       # Gyroid 三维周期
# GYROID_GRID = 0.6          # Gyroid 采样间距

MODE_NAMES = ["1  HILBERT CURVE", "2  OCTAGRAM SPIRAL", "3  3D HONEYCOMB (KELVIN)"]

# ================================================


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]

    result = []
    for part in getattr(geometry, "geoms", []):
        result.extend(polygons(part))
    return result


def lines(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]

    result = []
    for part in getattr(geometry, "geoms", []):
        result.extend(lines(part))
    return result


def boundaries(region):
    result = []
    for poly in polygons(region):
        result.append(list(poly.exterior.coords))
        result.extend(list(ring.coords) for ring in poly.interiors)
    return result


def clip_paths(region, paths):
    result = []
    for path in paths:
        if len(path) < 2:
            continue
        clipped = region.intersection(LineString(path))
        for piece in lines(clipped):
            if piece.length > 1e-6:
                result.append(list(piece.coords))
    return result


def vtk_paths(data):
    """取出 VTK 中的线段/折线。"""
    cells = data.GetLines()
    cells.InitTraversal()
    ids = vtk.vtkIdList()
    result = []

    while cells.GetNextCell(ids):
        result.append([
            data.GetPoint(ids.GetId(i))[:2]
            for i in range(ids.GetNumberOfIds())
        ])
    return result


def prepare_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

    model = load(str(MODEL_PATH))
    if not hasattr(model, "dataset"):
        raise RuntimeError("模型未加载成单个网格")

    rotation = vtk.vtkTransform()
    rotation.RotateX(ROTATE_X)
    rotation.RotateY(ROTATE_Y)
    rotation.RotateZ(ROTATE_Z)

    rotated = vtk.vtkTransformFilter()
    rotated.SetInputData(model.dataset)
    rotated.SetTransform(rotation)
    rotated.Update()

    bounds = rotated.GetOutput().GetBounds()
    height = bounds[5] - bounds[4]

    if height <= 0:
        raise ValueError("模型高度无效")

    scale = MODEL_HEIGHT / height
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2

    # 明确设置变换矩阵：缩放、居中，并让脚底落在 Z=0。
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()

    for axis in range(3):
        matrix.SetElement(axis, axis, scale)

    matrix.SetElement(0, 3, -cx * scale)
    matrix.SetElement(1, 3, -cy * scale)
    matrix.SetElement(2, 3, -bounds[4] * scale)

    placement = vtk.vtkTransform()
    placement.SetMatrix(matrix)

    positioned = vtk.vtkTransformFilter()
    positioned.SetInputConnection(rotated.GetOutputPort())
    positioned.SetTransform(placement)

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(positioned.GetOutputPort())

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())
    clean.Update()

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(clean.GetOutput())
    return mesh


def cross_section(mesh, z):
    """沿水平 XY 平面截取模型，按奇偶规则保留孔洞。"""
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, z)
    plane.SetNormal(0, 0, 1)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(plane)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(cutter.GetOutputPort())

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(clean.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    region = Polygon()

    for path in vtk_paths(stripper.GetOutput()):
        if len(path) < 4:
            continue
        if np.linalg.norm(np.array(path[0]) - path[-1]) > 1e-4:
            continue

        for poly in polygons(make_valid(Polygon(path))):
            if poly.area > 1e-7:
                region = region.symmetric_difference(poly)

    return make_valid(region)











def hilbert_infill(region, z):
    """A space-filling Hilbert path, clipped to each cross section."""
    if region.is_empty:
        return []
    xmin, ymin, xmax, ymax = region.bounds
    span = max(xmax - xmin, ymax - ymin)
    order = max(1, min(7, math.ceil(math.log2(max(2, span / INFILL_SPACING)))))
    n = 2 ** order
    step = span / (n - 1)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    points = []
    for distance in range(n * n):
        x = y = 0
        t, size = distance, 1
        while size < n:
            rx = (t // 2) & 1
            ry = (t ^ rx) & 1
            if ry == 0:
                if rx == 1:
                    x, y = size - 1 - x, size - 1 - y
                x, y = y, x
            x += size * rx
            y += size * ry
            t //= 4
            size *= 2
        points.append((cx - span / 2 + x * step, cy - span / 2 + y * step))
    return clip_paths(region, [points])


def octagram_infill(region, z):
    """An outward-growing eight-point star spiral, clipped at walls."""
    result = []
    for poly in polygons(region):
        center = poly.representative_point()
        cx, cy = center.x, center.y
        xmin, ymin, xmax, ymax = poly.bounds
        radius = max(math.hypot(x - cx, y - cy)
                     for x in (xmin, xmax) for y in (ymin, ymax))
        # Sixteen vertices per turn: eight tips and eight inward corners.
        turns = math.ceil(radius / (0.68 * INFILL_SPACING)) + 2
        path = []
        for i in range(turns * 16 + 1):
            angle = i * math.pi / 8
            envelope = INFILL_SPACING * i / 16
            r = envelope * (1.0 if i % 2 == 0 else 0.68)
            path.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        result.extend(clip_paths(poly, [path]))
    return result


def honeycomb_3d_infill(region, z):
    """Horizontal sections of a BCC Voronoi honeycomb (truncated octahedra).

    Cells satisfy |dx|,|dy|,|dz| <= a and
    |dx|+|dy|+|dz| <= 1.5*a. Adjacent cells tile three-dimensional space.
    This is a Kelvin-style cellular demonstration, not Prusa's toolpath clone.
    """
    if region.is_empty:
        return []
    xmin, ymin, xmax, ymax = region.bounds
    a = INFILL_SPACING * 1.25
    pitch = 2 * a
    edges = []
    for parity in (0, 1):
        shift = parity * a
        for iz in range(math.floor((z - shift) / pitch) - 1,
                        math.floor((z - shift) / pitch) + 2):
            dz = abs(z - (iz * pitch + shift))
            if dz > a + 1e-8:
                continue
            diamond_radius = 1.5 * a - dz
            for ix in range(math.floor((xmin - shift) / pitch) - 1,
                            math.ceil((xmax - shift) / pitch) + 2):
                cx = ix * pitch + shift
                for iy in range(math.floor((ymin - shift) / pitch) - 1,
                                math.ceil((ymax - shift) / pitch) + 2):
                    cy = iy * pitch + shift
                    square = Polygon([(cx-a, cy-a), (cx+a, cy-a),
                                      (cx+a, cy+a), (cx-a, cy+a)])
                    d = diamond_radius
                    diamond = Polygon([(cx+d, cy), (cx, cy+d),
                                       (cx-d, cy), (cx, cy-d)])
                    cell = square.intersection(diamond)
                    if not cell.is_empty and cell.intersects(region):
                        edges.append(cell.boundary)
    if not edges:
        return []
    # Snap tiny floating-point differences before removing shared edges.
    from shapely import set_precision
    network = unary_union([set_precision(edge, 1e-6) for edge in edges])
    return [list(part.coords) for part in lines(region.intersection(network))
            if part.length > 1e-6]


def split_paths(paths, z, color):
    result = []

    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            a = np.asarray(a[:2], dtype=float)
            b = np.asarray(b[:2], dtype=float)

            distance = np.linalg.norm(b - a)
            if distance < 1e-7:
                continue

            count = max(1, math.ceil(distance / SEGMENT_LENGTH))

            for i in range(count):
                p = a + (b - a) * i / count
                q = a + (b - a) * (i + 1) / count
                result.append((
                    (float(p[0]), float(p[1]), z),
                    (float(q[0]), float(q[1]), z),
                    color,
                ))

    return result


def build_layers(mesh):
    modes = [[] for _ in MODE_NAMES]
    total = math.ceil(MODEL_HEIGHT / LAYER_HEIGHT)

    for layer in range(total):
        bottom = layer * LAYER_HEIGHT
        top = min(bottom + LAYER_HEIGHT, MODEL_HEIGHT)
        z = (bottom + top) / 2

        region = cross_section(mesh, z)
        wall_paths = []

        for wall in range(WALL_COUNT):
            offset = (wall + 0.5) * LINE_WIDTH
            wall_paths.extend(boundaries(region.buffer(-offset)))

        walls = split_paths(wall_paths, z, 0)

        inside = region.buffer(
            -(WALL_COUNT * LINE_WIDTH + LINE_WIDTH / 2)
        )

        patterns = [
            hilbert_infill(inside, z),
            octagram_infill(inside, z),
            honeycomb_3d_infill(inside, z),
        ]

        for mode, paths in enumerate(patterns):
            modes[mode].append(
                walls + split_paths(paths, z, 1)
            )

        print(
            f"\r正在计算三种趣味填充：{layer + 1}/{total}",
            end="",
            flush=True,
        )

    print()

    if not any(modes[0]):
        raise RuntimeError("没有有效切片，请检查模型封闭性")

    return modes


class PrintedPaths:
    def __init__(self, renderer, color):
        self.points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()

        self.data = vtk.vtkPolyData()
        self.data.SetPoints(self.points)
        self.data.SetLines(self.cells)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(self.data)
        tube.SetRadius(LINE_WIDTH / 2)
        tube.SetNumberOfSides(6)
        tube.CappingOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(*color)
        self.actor.GetProperty().SetAmbient(0.3)
        self.actor.GetProperty().SetDiffuse(0.7)
        renderer.AddActor(self.actor)

        # 保留管线引用。
        self.tube = tube
        self.mapper = mapper

    def add(self, a, b):
        first = self.points.InsertNextPoint(*a)
        second = self.points.InsertNextPoint(*b)

        self.cells.InsertNextCell(2)
        self.cells.InsertCellPoint(first)
        self.cells.InsertCellPoint(second)

    def update(self):
        self.points.Modified()
        self.cells.Modified()
        self.data.Modified()

    def clear(self):
        self.points.Reset()
        self.cells.Reset()
        self.update()


def make_label(renderer, text, x=12, y=12, size=16):
    label = vtk.vtkTextActor()
    label.SetInput(text)
    label.SetPosition(x, y)
    label.GetTextProperty().SetFontSize(size)
    label.GetTextProperty().SetColor(0.95, 0.95, 0.95)

    # 兼容你当前安装的 VTK。
    renderer.AddViewProp(label)
    return label


def main():
    if INFILL_SPACING <= 0 or LAYER_HEIGHT <= 0:
        raise ValueError("填充间距和层厚必须大于 0")

    mesh = prepare_model()
    layers = build_layers(mesh)

    window = vtk.vtkRenderWindow()
    window.SetWindowName("Cat - Hilbert / Octagram / 3D Honeycomb")
    window.SetSize(1500, 850)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera()
    )

    renderers = []
    actors = []
    labels = []
    ghosts = []

    for index, title in enumerate(MODE_NAMES):

        renderer = vtk.vtkRenderer()
        renderer.SetViewport(index / 3, 0.20, (index + 1) / 3, 1)
        renderer.SetBackground(0.035, 0.045, 0.065)
        window.AddRenderer(renderer)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)

        ghost = vtk.vtkActor()
        ghost.SetMapper(mapper)
        ghost.GetProperty().SetColor(0.65, 0.72, 0.8)
        ghost.GetProperty().SetOpacity(0.045)
        renderer.AddActor(ghost)

        red = PrintedPaths(renderer, (1.0, 0.14, 0.20))
        green = PrintedPaths(renderer, (0.12, 0.95, 0.40))

        label = make_label(renderer, title)

        camera = renderer.GetActiveCamera()
        camera.SetPosition(190, -160, 145)
        camera.SetFocalPoint(0, 0, MODEL_HEIGHT / 2)
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCamera()

        renderers.append(renderer)
        actors.append([red, green])
        labels.append(label)
        ghosts.append(ghost)

    # 底部显示操作说明。
    help_renderer = vtk.vtkRenderer()
    help_renderer.SetViewport(0, 0, 1, 0.20)
    help_renderer.SetBackground(0.025, 0.03, 0.045)
    window.AddRenderer(help_renderer)

    make_label(
        help_renderer,
        "Red: walls   Green: infill   |   Build direction: feet -> ears (+Z)\n"
        "Space: pause   R: restart   A: rotation   S: single layer / stack\n"
        "T: top view   V: angled view   G: ghost model   |   Mouse: rotate / zoom\n"
        "3D honeycomb: truncated-octahedron cells (Kelvin-style demonstration)",
        x=24,
        y=18,
        size=17,
    )

    state = {
        "layer": 0,
        "positions": [0] * len(MODE_NAMES),
        "paused": False,
        "rotate": True,
        "single": False,
        "last_time": time.perf_counter(),
    }

    def clear_print():
        for pair in actors:
            for actor in pair:
                actor.clear()

    def restart():
        state["layer"] = 0
        state["positions"] = [0] * len(MODE_NAMES)
        clear_print()

    def update_labels():
        current = min(state["layer"] + 1, len(layers[0]))
        status = "PAUSED" if state["paused"] else "PRINTING"

        if state["layer"] >= len(layers[0]):
            status = "FINISHED"

        display = "SINGLE" if state["single"] else "STACK"

        for label, title in zip(labels, MODE_NAMES):
            label.SetInput(
                f"{title}\n"
                f"Layer {current}/{len(layers[0])}  "
                f"{status}  {display}"
            )

    def set_view(top=False):
        for renderer in renderers:
            camera = renderer.GetActiveCamera()
            camera.SetFocalPoint(0, 0, MODEL_HEIGHT / 2)

            if top:
                camera.SetPosition(0, 0, MODEL_HEIGHT + 250)
                camera.SetViewUp(0, 1, 0)
            else:
                camera.SetPosition(190, -160, 145)
                camera.SetViewUp(0, 0, 1)

            renderer.ResetCamera()

    def on_key(obj, event):
        key = obj.GetKeySym().lower()

        if key == "space":
            state["paused"] = not state["paused"]
        elif key == "r":
            restart()
        elif key == "a":
            state["rotate"] = not state["rotate"]
        elif key == "s":
            state["single"] = not state["single"]
            restart()
        elif key == "g":
            for ghost in ghosts:
                ghost.SetVisibility(not ghost.GetVisibility())
        elif key == "t":
            state["rotate"] = False
            set_view(top=True)
        elif key == "v":
            set_view(top=False)

        update_labels()
        window.Render()

    def on_timer(obj, event):
        now = time.perf_counter()
        dt = min(now - state["last_time"], 0.15)
        state["last_time"] = now

        if state["rotate"]:
            for renderer in renderers:
                renderer.GetActiveCamera().Azimuth(
                    ROTATION_SPEED * dt
                )

        layer = state["layer"]

        if not state["paused"] and layer < len(layers[0]):
            for mode in range(len(MODE_NAMES)):
                segments = layers[mode][layer]
                start = state["positions"][mode]
                end = min(
                    start + SEGMENTS_PER_FRAME,
                    len(segments),
                )

                changed = set()

                for a, b, color in segments[start:end]:
                    actors[mode][color].add(a, b)
                    changed.add(color)

                for color in changed:
                    actors[mode][color].update()

                state["positions"][mode] = end

            # 所有填充方式完成本层后，一起上升到下一层。
            completed = all(
                state["positions"][mode]
                >= len(layers[mode][layer])
                for mode in range(len(MODE_NAMES))
            )

            if completed:
                state["layer"] += 1
                state["positions"] = [0] * len(MODE_NAMES)

                if (
                    state["single"]
                    and state["layer"] < len(layers[0])
                ):
                    clear_print()

        update_labels()

        for renderer in renderers:
            renderer.ResetCameraClippingRange()

        window.Render()

    interactor.AddObserver("KeyPressEvent", on_key)
    interactor.AddObserver("TimerEvent", on_timer)

    update_labels()
    interactor.Initialize()
    window.Render()
    interactor.CreateRepeatingTimer(FRAME_MS)
    interactor.Start()


if __name__ == "__main__":
    main()