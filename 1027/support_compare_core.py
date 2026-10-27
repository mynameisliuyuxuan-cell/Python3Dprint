"""Shared, pinned geometry core for the three-strategy comparison.
No GUI launch or external slicing engine is required.
"""
from pathlib import Path


import argparse


import math


import time


import numpy as np


import vtk


from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


from vedo import load


MODEL_PATH = Path(__file__).resolve().parent / 'ORIGAMI_Chat_Free.obj'


MODEL_HEIGHT = 80.0           # 统一模型高度；演示按毫米理解


OVERHANG_ANGLE = 45.0        # 朝下表面与水平面的夹角阈值，小于此值生成支撑


SUPPORT_SPACING = 3.0        # 支撑采样间距；越小越密


SUPPORT_RADIUS = 0.55        # 支撑柱半径


TIP_HEIGHT = 2.0            # 顶端锥形部分的高度


CONTACT_GAP = 0.25          # 与模型底面的竖直间隙


MIN_SUPPORT_HEIGHT = 1.0


ROTATION_SPEED = 2.0        # 每秒旋转角度


def prepare_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'找不到猫模型：{MODEL_PATH}')
    model = load(str(MODEL_PATH))
    if not hasattr(model, 'dataset'):
        raise RuntimeError('OBJ 没有加载成单个网格')
    # 原始猫为 Y-up：绕 X 旋转 +90 度后，原 Y 成为新 Z。
    original = vtk_to_numpy(model.dataset.GetPoints().GetData())
    points = original[:, [0, 2, 1]].astype(np.float64).copy()
    points[:, 1] *= -1
    low, high = points.min(axis=0), points.max(axis=0)
    height = high[2] - low[2]
    if height <= 0:
        raise ValueError('模型高度无效')
    points -= np.array([(low[0] + high[0]) / 2,
                        (low[1] + high[1]) / 2, low[2]])
    points *= MODEL_HEIGHT / height
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(model.dataset)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    mesh.SetPoints(vtk_points)
    # OBJ normals describe the original Y-up geometry. Recompute after rotation.
    mesh.GetPointData().SetNormals(None)
    mesh.GetCellData().SetNormals(None)

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(mesh)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(normals.GetOutput())
    return result


def set_chinese_font(text_property):
    """使用系统黑体 TTF，确保 VTK/FreeType 能显示中文字形。"""
    font = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not font.exists():
        raise FileNotFoundError(f"未找到中文字体：{font}")
    text_property.SetFontFamily(vtk.VTK_FONT_FILE)
    text_property.SetFontFile(str(font))
    text_property.SetBold(False)
    text_property.SetItalic(False)


def actor_for(data, color, opacity=1):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetAmbient(0.25)
    return actor


def highlight_faces(mesh, threshold, enabled=True):
    """Orange shows angle-selected downward faces above the platform.

    Not all highlighted faces have an unobstructed vertical path to the bed.
    """
    normals = vtk_to_numpy(mesh.GetCellData().GetNormals())
    mask = normals[:, 2] < -math.cos(math.radians(threshold))
    for i in np.flatnonzero(mask):
        if mesh.GetCell(int(i)).GetBounds()[5] <= MIN_SUPPORT_HEIGHT + CONTACT_GAP:
            mask[i] = False
    colors = np.tile(np.array([235, 25, 36], dtype=np.uint8), (len(normals), 1))
    if enabled:
        colors[mask] = (255, 165, 30)
    scalars = numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName('OverhangColors')
    mesh.GetCellData().SetScalars(scalars)
    mesh.Modified()
    return int(mask.sum())


from dataclasses import dataclass


DEFAULTS = dict(angle=45., spacing=5., merge=14., branch_angle=35., diameter=.8, gap=.6)


@dataclass
class Node:
    p: np.ndarray
    leaves: frozenset


class TreeBuilder:
    """Greedy downward support graph, checked against the full model.

    Distances are sampled with a half-step safety margin; center-line ray
    intersections are also rejected. This does not replace print validation.
    """
    def __init__(self, mesh, settings):
        self.mesh, self.cfg = mesh, settings.copy()
        self.distance = vtk.vtkImplicitPolyDataDistance()
        self.distance.SetInput(mesh)
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(mesh)
        self.locator.BuildLocator()
        self.nodes, self.edges = [], []
        self.joins = 0

    def node(self, p, leaves):
        self.nodes.append(Node(np.array(p, dtype=float), frozenset(leaves)))
        return len(self.nodes) - 1

    def radius(self, count):
        return min(1.6, self.cfg['diameter'] / 2 * math.sqrt(count))

    def diagnose_edge(self, a, b, radius, end_radius=None):
        delta = b - a
        length = float(np.linalg.norm(delta))
        def failure(code, reason, **extra):
            return dict(code=code, reason=reason, a=list(map(float, a)),
                        b=list(map(float, b)), **extra)
        if length < 1e-6 or b[2] >= a[2] - 1e-6:
            return failure('direction', '支撑段没有向下延伸或长度不足')
        angle = math.degrees(math.atan2(np.linalg.norm(delta[:2]), a[2]-b[2]))
        if angle > self.cfg['branch_angle'] + 1e-6:
            return failure('angle', f'分支倾角 {angle:.1f}° 超过上限 {self.cfg["branch_angle"]:g}°')
        t, sub, cell = vtk.reference(0.), vtk.reference(0), vtk.reference(0)
        hit = [0.] * 3
        if self.locator.IntersectWithLine(a, b, 1e-7, t, hit, [0.]*3, sub, cell):
            return failure('intersection', '支撑中心线与模型相交', witness=hit)
        count = max(1, math.ceil(length / .4))
        end_radius = radius if end_radius is None else end_radius
        margin = (length + abs(end_radius-radius)) / count / 2
        for fraction in np.linspace(0, 1, count + 1):
            point = a + fraction * delta
            r = radius + (end_radius-radius) * fraction
            clearance = float(self.distance.EvaluateFunction(point))
            required = r + .08 + margin
            if clearance < required:
                return failure('clearance', f'净空不足：检测 {clearance:.2f} 毫米，要求至少 {required:.2f} 毫米',
                               witness=list(map(float, point)), clearance=clearance, required=required)
        return None

    def clear(self, a, b, radius, end_radius=None):
        return self.diagnose_edge(a, b, radius, end_radius) is None

    def build(self, contacts, mode='tree', fallback=True):
        self.failures, self.children = {}, {}
        self.splits, self.recovered = 0, set()
        active = []
        for i, point in enumerate(contacts):
            top = np.array(point) - [0, 0, self.cfg['gap']]
            head = top - [0, 0, 2.5]
            if head[2] <= .5:
                reason = dict(code='tip_height', reason='接触点过低，放不下当前尖端和间隙')
            else:
                reason = self.diagnose_edge(top, head, .06, self.radius(1))
            if reason:
                self.failures[i] = dict(stage='接触尖端', attempts=[reason], **reason)
                continue
            a, b = self.node(top, [i]), self.node(head, [i])
            self.edges.append((a, b, .06, self.radius(1)))
            active.append(b)

        if mode == 'tree':
            while len(active) > 1:
                pairs = []
                for i in range(len(active)):
                    for j in range(i+1, len(active)):
                        a, b = self.nodes[active[i]], self.nodes[active[j]]
                        lateral = np.linalg.norm(a.p[:2]-b.p[:2])
                        if lateral <= self.cfg['merge']:
                            pairs.append((lateral + .15*abs(a.p[2]-b.p[2]), i, j))
                merged = False
                for _, i, j in sorted(pairs):
                    ai, bi = active[i], active[j]
                    a, b = self.nodes[ai], self.nodes[bi]
                    midpoint = (a.p[:2]+b.p[:2])/2
                    leaves = a.leaves | b.leaves
                    r = self.radius(len(leaves))
                    lateral = max(np.linalg.norm(a.p[:2]-midpoint), np.linalg.norm(b.p[:2]-midpoint))
                    drop = max(2.5, lateral/math.tan(math.radians(self.cfg['branch_angle']))+.2)
                    parent = np.array([*midpoint, min(a.p[2], b.p[2])-drop])
                    if parent[2] <= r+.5 or self.distance.EvaluateFunction(parent) < r+.3:
                        continue
                    ra, rb = self.radius(len(a.leaves)), self.radius(len(b.leaves))
                    if not (self.clear(a.p,parent,ra,r) and self.clear(b.p,parent,rb,r)):
                        continue
                    pi = self.node(parent, leaves)
                    self.children[pi] = [ai, bi]
                    self.edges.extend([(ai,pi,ra,r), (bi,pi,rb,r)])
                    active = [v for k,v in enumerate(active) if k not in (i,j)] + [pi]
                    merged = True
                    break
                if not merged:
                    break

        solved, roots, removed = set(), [], set()
        bounds = self.mesh.GetBounds()

        def attach(index, recovering=False):
            node = self.nodes[index]
            r = self.radius(len(node.leaves))
            candidates = [np.array([*node.p[:2], 0.])]
            if mode == 'tree':
                reach = min(node.p[2]*math.tan(math.radians(self.cfg['branch_angle']))*.8, self.cfg['merge'])
                for reach_radius in (reach/2, reach):
                    for theta in np.linspace(0, 2*math.pi, 12, endpoint=False):
                        candidates.append(np.array([node.p[0]+reach_radius*math.cos(theta),
                                                    node.p[1]+reach_radius*math.sin(theta), 0.]))
            attempts = []
            for p in candidates:
                if not (bounds[0]-8+r <= p[0] <= bounds[1]+8-r and bounds[2]-8+r <= p[1] <= bounds[3]+8-r):
                    attempts.append(dict(code='bed_bounds', reason='落点超出平台边界'))
                    continue
                reason = self.diagnose_edge(node.p, p, r)
                if reason:
                    attempts.append(reason)
                    continue
                root = self.node(p, node.leaves)
                self.edges.append((index,root,r,r))
                roots.append((p,r))
                solved.update(node.leaves)
                if recovering:
                    self.recovered.update(node.leaves)
                return
            if fallback and index in self.children:
                # Undo just this join. Retain each child's valid upper subtree
                # and try its own platform route before splitting it further.
                self.splits += 1
                for child in self.children[index]:
                    removed.add((child,index))
                    attach(child, True)
                return
            reason = next((item for item in attempts if item['code'] != 'bed_bounds'), attempts[0])
            for leaf in node.leaves:
                self.failures[leaf] = dict(stage='连接平台', attempts=attempts,
                    tried=len(candidates), **reason)

        for index in active:
            attach(index)
        self.solved, self.roots = solved, roots
        self.edges = [edge for edge in self.edges
                      if (edge[0],edge[1]) not in removed
                      and self.nodes[edge[0]].leaves <= solved]
        # Count only joins remaining in the rendered, platform-connected graph.
        self.joins = len({b for _,b,_,_ in self.edges if b in self.children})
        assert set(self.failures) == set(range(len(contacts))) - solved
        return self


    def geometry(self):
        append = vtk.vtkAppendPolyData()
        volume = 0.
        for ai, bi, ra, rb in self.edges:
            a, b = self.nodes[ai].p, self.nodes[bi].p
            points = vtk.vtkPoints()
            points.InsertNextPoint(*a)
            points.InsertNextPoint(*b)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(2)
            cells.InsertCellPoint(0)
            cells.InsertCellPoint(1)
            data = vtk.vtkPolyData()
            data.SetPoints(points)
            data.SetLines(cells)
            radii = numpy_to_vtk(np.array([ra, rb], dtype=float), deep=True)
            data.GetPointData().SetScalars(radii)
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(data)
            tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            tube.SetNumberOfSides(12)
            tube.CappingOn()
            tube.Update()
            append.AddInputData(tube.GetOutput())
            length = np.linalg.norm(b - a)
            volume += math.pi * length * (ra*ra + ra*rb + rb*rb) / 3
        for p, r in self.roots:
            base = vtk.vtkConeSource()
            base.SetDirection(0, 0, 1)
            base.SetCenter(p[0], p[1], .5)
            base.SetHeight(1)
            # Only use a foot if its larger radius clears the model.
            foot_r = min(2., r * 1.8)
            if self.distance.EvaluateFunction([p[0], p[1], .5]) <= foot_r + .6:
                continue
            base.SetRadius(foot_r)
            base.SetResolution(16)
            base.Update()
            append.AddInputData(base.GetOutput())
            volume += math.pi * foot_r**2 / 3
        output = vtk.vtkPolyData()
        if append.GetNumberOfInputConnections(0):
            append.Update()
            output.DeepCopy(append.GetOutput())
        return output, volume


def automatic_contacts(mesh, angle, spacing):
    """Sample the real downward triangles, including regions blocked below."""
    normals = vtk_to_numpy(mesh.GetCellData().GetNormals())
    contacts = []
    for cell_id in np.flatnonzero(normals[:, 2] < -math.cos(math.radians(angle))):
        cell = mesh.GetCell(int(cell_id))
        vertices = np.array([mesh.GetPoint(cell.GetPointId(j)) for j in range(3)])
        if vertices[:, 2].max() <= 2:
            continue
        span = max(np.linalg.norm(vertices[i] - vertices[j]) for i in range(3) for j in range(i))
        n = max(1, math.ceil(span / spacing))
        for i in range(n):
            for j in range(n - i):
                u, v = (i + 1/3) / n, (j + 1/3) / n
                p = vertices[0] + u * (vertices[1]-vertices[0]) + v * (vertices[2]-vertices[0])
                if p[2] > 3.5 and all(np.linalg.norm(p-q) >= spacing * .7 for q in contacts):
                    contacts.append(p)
    return contacts


def glyph_contacts(contacts, solved=None):
    data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for p in contacts:
        points.InsertNextPoint(*p)
    data.SetPoints(points)
    colors = np.array([(255, 225, 30) if solved is None or i in solved else (190, 65, 255)
                       for i in range(len(contacts))], dtype=np.uint8).reshape(-1, 3)
    data.GetPointData().SetScalars(numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    return data
