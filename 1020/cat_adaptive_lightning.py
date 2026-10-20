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

MODE_NAMES = ["1  ADAPTIVE CUBIC", "2  LIGHTNING (ROOF-DRIVEN)"]

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


















def adaptive_cells(mesh):
    """Surface-distance-driven octree in a corner-up cubic coordinate frame."""
    from vtk.util.numpy_support import vtk_to_numpy
    basis = np.array([[1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(3)],
                      [-1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(3)],
                      [0, -2 / math.sqrt(6), 1 / math.sqrt(3)]])
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    rotated = points @ basis
    center = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
    minimum = INFILL_SPACING
    size = minimum * 2 ** math.ceil(math.log2(np.ptp(rotated, axis=0).max() / minimum))
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(normals.GetOutput())
    cells = []
    corners = np.array([[x, y, z] for x in (-1, 1)
                        for y in (-1, 1) for z in (-1, 1)])

    def visit(c, edge):
        p = basis @ c
        d = distance.EvaluateFunction(p)
        radius = math.sqrt(3) * edge / 2
        if d > radius:
            return
        # Small cells near the surface, larger cells deep inside.
        refine = edge > minimum * 1.01 and (
            edge > minimum * 4.01 or abs(d) < radius + minimum * 0.15
        )
        if refine:
            for direction in corners:
                visit(c + direction * edge / 4, edge / 2)
        else:
            cells.append((c + corners * edge / 2) @ basis.T)

    visit(center, size)
    print(f'Adaptive cubic: {len(cells)} octree cells')
    return cells


def adaptive_section(region, z, cells):
    """Slice the rotated octree cells, then clip to the real model interior."""
    from shapely.geometry import MultiPoint
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if bin(a ^ b).count('1') == 1]
    paths = []
    for vertices in cells:
        if z < vertices[:, 2].min() or z > vertices[:, 2].max():
            continue
        cuts = []
        for a, b in edges:
            p, q = vertices[a], vertices[b]
            if abs(p[2] - z) < 1e-8:
                cuts.append(tuple(p[:2]))
            if (p[2] - z) * (q[2] - z) < 0:
                t = (z - p[2]) / (q[2] - p[2])
                cuts.append(tuple((p + t * (q - p))[:2]))
        if len(cuts) >= 3:
            hull = MultiPoint(cuts).convex_hull
            if hull.geom_type == 'Polygon':
                paths.append(hull.boundary)
    if not paths or region.is_empty:
        return []
    from shapely import set_precision
    network = unary_union([set_precision(path, 1e-6) for path in paths])
    return [list(item.coords) for item in lines(region.intersection(network))
            if item.length > 1e-6]


def roof_seeds(region):
    """Sample support targets on upward-facing areas, not random points."""
    from shapely.geometry import Point
    seeds = []
    spacing = INFILL_SPACING * 2
    for poly in polygons(region):
        if poly.area < LINE_WIDTH ** 2:
            continue
        xmin, ymin, xmax, ymax = poly.bounds
        local = []
        for x in np.arange(math.ceil(xmin / spacing) * spacing, xmax, spacing):
            for y in np.arange(math.ceil(ymin / spacing) * spacing, ymax, spacing):
                if poly.covers(Point(x, y)):
                    local.append(np.array([x, y], dtype=float))
        if not local and poly.area > spacing * LINE_WIDTH:
            p = poly.representative_point()
            local.append(np.array([p.x, p.y]))
        seeds.extend(local)
    return seeds



def lightning_layers(regions, zs):
    """Ceiling-driven, merge-first branching demonstration.

    Descend toward neighboring support nodes; isolated trunks remain vertical.
    Terminate only when the descending trunk actually meets the inner shell.
    This is a geometry visualization rather than an extrusion planner.
    """
    from shapely.geometry import Point
    result = [[] for _ in zs]
    active = []
    step_limit = LAYER_HEIGHT
    merges = roots = 0
    for k in range(len(zs) - 1, -1, -1):
        current = regions[k]
        z = zs[k]
        upper = regions[k + 1] if k + 1 < len(zs) else Polygon()
        corridor = current.intersection(upper)
        waiting = []
        # Anchor to the shell when a vertical trunk exits the interior.
        for xy, upper_z in active:
            point = Point(xy)
            if current.is_empty or not current.covers(point):
                up_clearance = point.distance(upper.boundary) if not upper.is_empty else 0
                low_clearance = point.distance(current) if not current.is_empty else LAYER_HEIGHT
                fraction = up_clearance / max(up_clearance + low_clearance, 1e-8)
                root_z = upper_z - fraction * (upper_z - z)
                root_z = min(upper_z - 1e-5, max(z, root_z))
                result[k].append(((float(xy[0]), float(xy[1]), root_z),
                                  (float(xy[0]), float(xy[1]), upper_z), 1))
                roots += 1
            else:
                waiting.append((xy, upper_z))
        used = set()
        next_active = []
        for i, (xy, upper_z) in enumerate(waiting):
            if i in used:
                continue
            candidates = sorted(
                (np.linalg.norm(other - xy), j)
                for j, (other, _) in enumerate(waiting)
                if j != i and j not in used
                and corridor.covers(LineString([xy, other]))
            )
            group = [(xy, upper_z)]
            target = xy.copy()
            if candidates:
                distance, j = candidates[0]
                other, other_z = waiting[j]
                if distance <= 2 * step_limit:
                    target = (xy + other) / 2
                    group.append((other, other_z))
                    used.add(j)
                    merges += 1
                else:
                    target = xy + (other - xy) * (step_limit / distance)
            used.add(i)
            for parent, parent_z in group:
                result[k].append(((float(target[0]), float(target[1]), z),
                                  (float(parent[0]), float(parent[1]), parent_z), 1))
            next_active.append((target, z))
        if not current.is_empty:
            roof = current.difference(upper)
            for seed in roof_seeds(roof):
                if not any(np.linalg.norm(seed - node) < INFILL_SPACING
                           for node, _ in next_active):
                    next_active.append((seed, z))
        if k == 0:
            for xy, _ in next_active:
                result[k].append(((float(xy[0]), float(xy[1]), 0.0),
                                  (float(xy[0]), float(xy[1]), z), 1))
                roots += 1
        active = next_active
    print(f'Lightning: {merges} branch joins, {roots} shell/base roots')
    return result



def build_layers(mesh):
    total = math.ceil(MODEL_HEIGHT / LAYER_HEIGHT)
    zs, regions, walls = [], [], []
    for layer in range(total):
        z = (layer * LAYER_HEIGHT + min((layer + 1) * LAYER_HEIGHT, MODEL_HEIGHT)) / 2
        section = cross_section(mesh, z)
        paths = []
        for wall in range(WALL_COUNT):
            paths.extend(boundaries(section.buffer(-(wall + 0.5) * LINE_WIDTH)))
        walls.append(split_paths(paths, z, 0))
        regions.append(section.buffer(-(WALL_COUNT * LINE_WIDTH + LINE_WIDTH / 2)))
        zs.append(z)
    cells = adaptive_cells(mesh)
    lightning = lightning_layers(regions, zs)
    modes = [[], []]
    for layer, (region, z) in enumerate(zip(regions, zs)):
        cubic = split_paths(adaptive_section(region, z, cells), z, 1)
        modes[0].append(walls[layer] + cubic)
        modes[1].append(walls[layer] + lightning[layer])
        print(f'\rAdaptive / Lightning: {layer + 1}/{total}', end='', flush=True)
    print()
    if not any(walls):
        raise RuntimeError('No closed model cross sections')
    return modes


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
    window.SetWindowName("Cat - Adaptive Cubic / Lightning")
    window.SetSize(1400, 850)

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
        renderer.SetViewport(index / 2, 0.20, (index + 1) / 2, 1)
        renderer.SetBackground(0.035, 0.045, 0.065)
        window.AddRenderer(renderer)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)

        ghost = vtk.vtkActor()
        ghost.SetMapper(mapper)
        ghost.GetProperty().SetColor(0.65, 0.72, 0.8)
        ghost.GetProperty().SetOpacity(0.045)
        renderer.AddActor(ghost)
        ghost.SetVisibility(False)

        red = PrintedPaths(renderer, (1.0, 0.14, 0.20))
        green = PrintedPaths(renderer, (0.12, 0.95, 0.40))
        red.actor.GetProperty().SetOpacity(0.20)
        if index == 1:
            green.tube.SetRadius(LINE_WIDTH * 0.65)

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
        "T: top view   V: angled view   G: ghost   W: wall opacity   F: finished model\n"
        "Adaptive: surface-driven octree. Lightning: ceiling-driven branches (merge-first demo). Not G-code.",
        x=24,
        y=18,
        size=17,
    )

    state = {
        "layer": 0,
        "positions": [0] * len(MODE_NAMES),
        "paused": False,
        "walls_faint": True,
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
        elif key == "w":
            state["walls_faint"] = not state["walls_faint"]
            for pair in actors:
                pair[0].actor.GetProperty().SetOpacity(0.20 if state["walls_faint"] else 1.0)
        elif key == "f":
            clear_print()
            for mode, pair in enumerate(actors):
                for layer in layers[mode]:
                    for a, b, color in layer:
                        pair[color].add(a, b)
                for actor in pair:
                    actor.update()
            state["layer"] = len(layers[0])
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