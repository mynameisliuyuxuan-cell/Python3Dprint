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

GYROID_PERIOD = 12.0       # Gyroid 三维周期
GYROID_GRID = 0.6          # Gyroid 采样间距

MODE_NAMES = [
    "1  HORIZONTAL",
    "2  HEXAGON",
    "3  GYROID",
    "4  DIAGONAL",
    "5  SPIRAL",
]

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


def straight_infill(region, angle=0):
    """按指定角度排列平行线，并逐行反向。"""
    if region.is_empty:
        return []

    xmin, ymin, xmax, ymax = region.bounds
    theta = math.radians(angle)

    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])

    corners = np.array([
        [xmin, ymin], [xmin, ymax],
        [xmax, ymin], [xmax, ymax],
    ])

    offsets = corners @ normal
    distances = corners @ direction
    lo, hi = distances.min() - 1, distances.max() + 1

    paths = []
    row = 0
    offset = (
        math.floor(offsets.min() / INFILL_SPACING)
        * INFILL_SPACING
    )

    while offset <= offsets.max():
        a = normal * offset + direction * lo
        b = normal * offset + direction * hi

        pieces = sorted(
            lines(region.intersection(LineString([a, b]))),
            key=lambda item: np.dot(item.coords[0], direction),
        )

        if row % 2:
            pieces.reverse()

        for piece in pieces:
            points = sorted(
                piece.coords,
                key=lambda point: np.dot(point, direction),
            )
            if row % 2:
                points.reverse()
            paths.append(points)

        offset += INFILL_SPACING
        row += 1

    return paths


def hexagon_infill(region):
    """平铺六边形，共用边只保留一份。"""
    if region.is_empty:
        return []

    xmin, ymin, xmax, ymax = region.bounds
    radius = INFILL_SPACING
    dy = math.sqrt(3) * radius
    edges = {}

    first_col = math.floor(xmin / (1.5 * radius)) - 2
    last_col = math.ceil(xmax / (1.5 * radius)) + 2

    for col in range(first_col, last_col + 1):
        cx = col * 1.5 * radius
        shift = (col % 2) * dy / 2

        first_row = math.floor((ymin - shift) / dy) - 2
        last_row = math.ceil((ymax - shift) / dy) + 2

        for row in range(first_row, last_row + 1):
            cy = row * dy + shift

            vertices = [
                (
                    round(cx + radius * math.cos(i * math.pi / 3), 7),
                    round(cy + radius * math.sin(i * math.pi / 3), 7),
                )
                for i in range(6)
            ]

            for i in range(6):
                a = vertices[i]
                b = vertices[(i + 1) % 6]
                key = tuple(sorted((a, b)))
                edges[key] = LineString([a, b])

    network = unary_union(list(edges.values()))
    clipped = region.intersection(network)

    return [
        list(piece.coords)
        for piece in lines(clipped)
        if piece.length > 1e-6
    ]


def gyroid_infill(region, z):
    """
    三维 Gyroid 隐式曲面的水平截线。
    图案随 Z 改变，不是把同一张二维波纹逐层复制。
    """
    if region.is_empty:
        return []

    xmin, ymin, xmax, ymax = region.bounds
    step = GYROID_GRID

    nx = max(2, math.ceil((xmax - xmin) / step) + 1)
    ny = max(2, math.ceil((ymax - ymin) / step) + 1)

    x = xmin + np.arange(nx) * step
    y = ymin + np.arange(ny) * step
    xx, yy = np.meshgrid(x, y)

    k = 2 * math.pi / GYROID_PERIOD
    X, Y, Z = xx * k, yy * k, z * k

    values = (
        np.sin(X) * np.cos(Y)
        + np.sin(Y) * np.cos(Z)
        + np.sin(Z) * np.cos(X)
    )

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetOrigin(xmin, ymin, z)
    image.SetSpacing(step, step, step)

    scalars = numpy_to_vtk(
        np.ascontiguousarray(values.ravel()),
        deep=True,
    )
    scalars.SetName("gyroid")
    image.GetPointData().SetScalars(scalars)

    contour = vtk.vtkContourFilter()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(contour.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    return clip_paths(region, vtk_paths(stripper.GetOutput()))


def spiral_infill(region):
    """各独立截面内部的阿基米德螺旋，超出轮廓部分裁掉。"""
    result = []

    for poly in polygons(region):
        center = poly.representative_point()
        cx, cy = center.x, center.y
        xmin, ymin, xmax, ymax = poly.bounds

        radius = max(
            math.hypot(x - cx, y - cy)
            for x in (xmin, xmax)
            for y in (ymin, ymax)
        )

        # 每转一圈，半径增加一个填充间距。
        b = INFILL_SPACING / (2 * math.pi)
        maximum = radius / b
        theta = 0.0
        points = []

        while theta <= maximum:
            r = b * theta
            points.append((
                cx + r * math.cos(theta),
                cy + r * math.sin(theta),
            ))

            # 外圈使用更小角度步长，保持曲线平滑。
            theta += min(
                0.15,
                0.5 / max(math.hypot(r, b), 0.1),
            )

        result.extend(clip_paths(poly, [points]))

    return result


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
            straight_infill(inside, 0),
            hexagon_infill(inside),
            gyroid_infill(inside, z),
            straight_infill(inside, 45),
            spiral_infill(inside),
        ]

        for mode, paths in enumerate(patterns):
            modes[mode].append(
                walls + split_paths(paths, z, 1)
            )

        print(
            f"\r正在计算五种填充：{layer + 1}/{total}",
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
    window.SetWindowName("Cat - Five Infill Patterns")
    window.SetSize(1500, 960)

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
        col = index % 3
        row = index // 3

        renderer = vtk.vtkRenderer()
        renderer.SetViewport(
            col / 3,
            0.5 if row == 0 else 0,
            (col + 1) / 3,
            1 if row == 0 else 0.5,
        )
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

    # 第六格放操作说明。
    help_renderer = vtk.vtkRenderer()
    help_renderer.SetViewport(2 / 3, 0, 1, 0.5)
    help_renderer.SetBackground(0.025, 0.03, 0.045)
    window.AddRenderer(help_renderer)

    make_label(
        help_renderer,
        "FIVE INFILL PATTERNS\n\n"
        "Red: outer walls\n"
        "Green: internal infill\n\n"
        "Space: pause / continue\n"
        "R: restart\n"
        "A: rotation on / off\n"
        "S: single layer / stack\n"
        "G: ghost model on / off\n"
        "T: top view\n"
        "V: angled view\n\n"
        "Mouse drag: rotate\n"
        "Mouse wheel: zoom",
        x=24,
        y=40,
        size=19,
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