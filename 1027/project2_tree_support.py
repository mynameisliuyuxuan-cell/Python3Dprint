"""Simple build-plate supports using only numpy, VTK and the existing vedo.

Model: red. Supports: green. Model starts Y-up and is placed Z-up.
Run normally for the interactive viewer, or --preview FILE for a PNG.
"""
from pathlib import Path
import argparse
import math
import time

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vedo import load


# ======================= 可调参数 =======================
MODEL_PATH = Path(__file__).resolve().parent / 'ORIGAMI_Chat_Free.obj'
MODEL_HEIGHT = 80.0           # 统一模型高度；演示按毫米理解
OVERHANG_ANGLE = 45.0        # 朝下表面与水平面的夹角阈值，小于此值生成支撑
SUPPORT_SPACING = 3.0        # 支撑采样间距；越小越密
SUPPORT_RADIUS = 0.55        # 支撑柱半径
TIP_HEIGHT = 2.0            # 顶端锥形部分的高度
CONTACT_GAP = 0.25          # 与模型底面的竖直间隙
MIN_SUPPORT_HEIGHT = 1.0
ROTATION_SPEED = 2.0        # 每秒旋转角度
# ========================================================


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


def glyph_contacts(contacts, solved=None, selected=None):
    data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for p in contacts:
        points.InsertNextPoint(*p)
    data.SetPoints(points)
    colors = np.array([(255, 225, 30) if solved is None or i in solved else (190, 65, 255)
                       for i in range(len(contacts))], dtype=np.uint8).reshape(-1, 3)
    if selected is not None and 0 <= selected < len(colors):
        colors[selected] = (50, 220, 255)
    data.GetPointData().SetScalars(numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    return data


def main(preview=None):
    mesh = prepare_model()
    cfg = DEFAULTS.copy()
    contacts = automatic_contacts(mesh, cfg['angle'], cfg['spacing'])
    state = dict(mode='tree', rotate=True, edit=False, highlight=True, transparent=False,
                 dirty=True, seed_dirty=False, last=time.perf_counter(), solved=None, selected=None)
    window = vtk.vtkRenderWindow()
    window.SetWindowName('猫模型 · 树状支撑实验（1027）')
    window.SetSize(1500, 1050)
    scene, panel = vtk.vtkRenderer(), vtk.vtkRenderer()
    scene.SetViewport(.29, 0, 1, 1)
    panel.SetViewport(0, 0, .29, 1)
    scene.SetBackground(.035, .045, .065)
    panel.SetBackground(.065, .08, .11)
    window.AddRenderer(scene)
    window.AddRenderer(panel)
    model_actor = actor_for(mesh, (.92, .10, .14))
    model_actor.GetMapper().SetScalarModeToUseCellData()
    model_actor.GetMapper().SetColorModeToDirectScalars()
    model_actor.GetMapper().ScalarVisibilityOn()
    support_actor = actor_for(vtk.vtkPolyData(), (.1, .95, .3))
    scene.AddActor(model_actor)
    scene.AddActor(support_actor)
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(.75)
    sphere.SetThetaResolution(12)
    sphere.SetPhiResolution(8)
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetSourceConnection(sphere.GetOutputPort())
    mapper.ScalingOff()
    mapper.SetColorModeToDirectScalars()
    dots = vtk.vtkActor()
    dots.SetMapper(mapper)
    scene.AddActor(dots)
    bounds = mesh.GetBounds()
    bed = vtk.vtkPlaneSource()
    bed.SetOrigin(bounds[0]-8, bounds[2]-8, -.1)
    bed.SetPoint1(bounds[1]+8, bounds[2]-8, -.1)
    bed.SetPoint2(bounds[0]-8, bounds[3]+8, -.1)
    bed.Update()
    scene.AddActor(actor_for(bed.GetOutput(), (.18, .21, .25)))

    def text(renderer, content, x, y, size=17):
        actor = vtk.vtkTextActor()
        actor.SetInput(content)
        actor.SetPosition(x, y)
        set_chinese_font(actor.GetTextProperty())
        actor.GetTextProperty().SetFontSize(size)
        actor.GetTextProperty().SetColor(.95, .95, .95)
        renderer.AddViewProp(actor)
        return actor

    text(panel, '树状支撑实验\n修改参数后按 G 重新生成\n角度或间距改变后按 A 重采样\n注意：A 会清除手动接触点', 18, 945, 18)
    text(panel, '红：模型　橙：悬垂候选面\n黄：接触点　绿：支撑　紫：未解决　青：选中', 16, 16, 14)
    info = text(scene, '', 16, 16, 17)
    inspector = text(scene, '点击黄色/紫色接触点查看诊断；N：下一个未解决点。', 16, 970, 17)
    diagnostic_line = vtk.vtkLineSource()
    diagnostic_mapper = vtk.vtkPolyDataMapper()
    diagnostic_mapper.SetInputConnection(diagnostic_line.GetOutputPort())
    diagnostic_actor = vtk.vtkActor()
    diagnostic_actor.SetMapper(diagnostic_mapper)
    diagnostic_actor.GetProperty().SetColor(.75, .25, 1)
    diagnostic_actor.GetProperty().SetLineWidth(3)
    diagnostic_actor.SetVisibility(False)
    scene.AddActor(diagnostic_actor)
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(style)

    def update_dots():
        mapper.SetInputData(glyph_contacts(contacts, state['solved'], state['selected']))
        highlight_faces(mesh, cfg['angle'], state['highlight'])

    def message(line):
        mode = '树状合并' if state['mode'] == 'tree' else '独立直柱'
        edit = '接触点编辑' if state['edit'] else '旋转观察'
        info.SetInput(f'{mode}；{edit}；接触点 {len(contacts)} 个\n{line}\n'
                      'G：生成　A：重新采样（清除手动点）　C：直柱/树状　E：编辑开关\n'
                      '点击接触点查看原因；N：下个失败点；编辑模式点击接触点删除\n'
                      'H：高亮　M：透明　S：支撑　空格：转动　R：视角　D：默认\n'
                      '避障采用采样距离与射线检查；教学原型，尚未验证实机打印。')

    def mark_dirty(seed=False):
        state.update(dirty=True, solved=None, selected=None)
        diagnostic_actor.SetVisibility(False)
        inspector.SetInput('接触点或参数已修改，请重新生成后查看诊断。')
        state['seed_dirty'] = state['seed_dirty'] or seed
        support_actor.SetVisibility(False)
        update_dots()
        message('角度/间距已变更，请按 A 重采样。' if state['seed_dirty'] else '已修改，请按 G 生成支撑。')

    def generate():
        if state['seed_dirty']:
            message('请先按 A 重采样；此操作会清除手动接触点。')
            return
        state['rotate'] = False
        message('正在生成与检查支撑，请稍候……')
        window.Render()
        start = time.perf_counter()
        graph = TreeBuilder(mesh, cfg).build(contacts, state['mode'])
        data, volume = graph.geometry()
        support_actor.GetMapper().SetInputData(data)
        support_actor.SetVisibility(True)
        state.update(solved=graph.solved, dirty=False, graph=graph)
        update_dots()
        count = len(graph.solved)
        diagnostic_actor.SetVisibility(False)
        state['selected'] = None
        update_dots()
        inspector.SetInput('点击黄色/紫色点查看原因；N：下一个未解决点。\n青色：当前选中；紫线：失败候选，属于诊断示意。')
        message(f'成功 {count}；未解决 {len(contacts)-count}；平台根部 {len(graph.roots)}；合并 {graph.joins}\n'
                f'拆分回退 {graph.splits} 次，恢复 {len(graph.recovered)} 个点；体积累加约 {volume:.0f} 立方毫米（含交叠）\n'
                f'耗时 {time.perf_counter()-start:.1f} 秒；紫色点可点击查看失败阶段与原因。')
        print(f'{state["mode"]}: contacts={len(contacts)}, solved={count}, roots={len(graph.roots)}, joins={graph.joins}')

    def reset_camera():
        cam = scene.GetActiveCamera()
        cam.SetPosition(190, -150, 110)
        cam.SetFocalPoint(0, 0, 32)
        cam.SetViewUp(0, 0, 1)
        scene.ResetCamera()

    sliders = {}
    specs = [('angle', '悬垂角度（度）', 5, 85, .83, 0),
             ('spacing', '接触点间距（毫米）', 3, 10, .70, 1),
             ('merge', '分支合并距离（毫米）', 3, 25, .57, 1),
             ('branch_angle', '分支偏离竖直的最大角度', 10, 55, .44, 0),
             ('diameter', '单枝基础直径（毫米）', .4, 1.6, .31, 2),
             ('gap', '顶部接触间隙（毫米）', .3, 1.5, .18, 2)]
    for key, title, low, high, y, decimals in specs:
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(low)
        rep.SetMaximumValue(high)
        rep.SetValue(cfg[key])
        rep.SetTitleText(title)
        rep.SetLabelFormat('{:.' + str(decimals) + 'f}')
        for prop in (rep.GetTitleProperty(), rep.GetLabelProperty()):
            set_chinese_font(prop)
            prop.SetColor(.95, .95, .95)
        rep.SetTitleHeight(.018)
        rep.SetLabelHeight(.018)
        rep.GetSliderProperty().SetColor(.1, .95, .3)
        for coordinate, x in ((rep.GetPoint1Coordinate(), .035), (rep.GetPoint2Coordinate(), .255)):
            coordinate.SetCoordinateSystemToNormalizedDisplay()
            coordinate.SetValue(x, y)
        slider = vtk.vtkSliderWidget()
        slider.SetInteractor(interactor)
        slider.SetCurrentRenderer(panel)
        slider.SetRepresentation(rep)
        slider.SetAnimationModeToJump()
        def changed(obj, event, name=key, precision=decimals):
            value = round(obj.GetRepresentation().GetValue(), precision)
            if value != cfg[name]:
                cfg[name] = value
                state['rotate'] = False
                mark_dirty(name in ('angle', 'spacing'))
                window.Render()
        slider.AddObserver('InteractionEvent', changed)
        slider.AddObserver('EndInteractionEvent', changed)
        sliders[key] = slider, rep

    def resample():
        nonlocal contacts
        contacts = automatic_contacts(mesh, cfg['angle'], cfg['spacing'])
        state['seed_dirty'] = False
        mark_dirty()


    def inspect_contact(index):
        state['selected'] = index
        state['rotate'] = False
        diagnostic_actor.SetVisibility(False)
        graph = state.get('graph')
        p = contacts[index]
        header = f'接触点 #{index+1}；位置 ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})'
        if state['dirty'] or graph is None:
            detail = '当前结果待更新，请先按 G 生成。'
        elif index in graph.solved:
            detail = '已成功连接平台。' + (' 此点通过拆分合并分支恢复。' if index in graph.recovered else '')
        else:
            failure = graph.failures[index]
            detail = f"失败阶段：{failure['stage']}；{failure['reason']}"
            if failure['stage'] == '连接平台':
                detail += f"\n尝试 {failure['tried']} 个落点均失败，上方显示其中一个候选的原因。"
            if 'a' in failure and 'b' in failure:
                diagnostic_line.SetPoint1(*failure['a'])
                diagnostic_line.SetPoint2(*failure['b'])
                diagnostic_actor.SetVisibility(True)
            suggestion = ('可尝试增加顶部间隙或减小基础直径。' if failure['stage'] == '接触尖端'
                          else '可尝试减小直径、调整最大分支倾角；当前只搜索有限候选落点。')
            detail += '\n' + suggestion
        inspector.SetInput(header + '\n' + detail)
        update_dots()

    def contact_at_screen(x, y):
        # Pick the small contact spheres in screen space, including points
        # revealed by model transparency. Background clicks still rotate.
        candidates = []
        for i, p in enumerate(contacts):
            scene.SetWorldPoint(*p, 1)
            scene.WorldToDisplay()
            px, py, depth = scene.GetDisplayPoint()
            distance = (px-x)**2 + (py-y)**2
            if 0 <= depth <= 1 and distance <= 12**2:
                candidates.append((distance, depth, i))
        return min(candidates)[2] if candidates else None

    def keypress(obj, event):
        key = obj.GetKeySym().lower()
        if key == 'g': generate()
        elif key == 'a': resample()
        elif key == 'c':
            state['mode'] = 'straight' if state['mode'] == 'tree' else 'tree'
            mark_dirty()
        elif key == 'e':
            state['edit'] = not state['edit']
            state['rotate'] = False
            message('编辑开关已切换；按 G 重新生成，按 E 返回观察模式。')
        elif key == 'n':
            graph = state.get('graph')
            failures = sorted(graph.failures) if graph is not None and not state['dirty'] else []
            if failures:
                old = state['selected'] if state['selected'] is not None else -1
                index = next((i for i in failures if i > old), failures[0])
                inspect_contact(index)
            else:
                inspector.SetInput('没有未解决点，或尚未重新生成。')
        elif key == 'h':
            state['highlight'] = not state['highlight']
            update_dots()
        elif key == 'm':
            state['transparent'] = not state['transparent']
            model_actor.GetProperty().SetOpacity(.25 if state['transparent'] else 1)
        elif key == 's': support_actor.SetVisibility(not support_actor.GetVisibility())
        elif key == 'space': state['rotate'] = not state['rotate']
        elif key == 'r': reset_camera()
        elif key == 'd':
            cfg.update(DEFAULTS)
            for name, (_, rep) in sliders.items(): rep.SetValue(cfg[name])
            resample()
            generate()
        window.Render()

    picker = vtk.vtkCellPicker()
    picker.PickFromListOn()
    picker.AddPickList(model_actor)
    picker.SetTolerance(.005)

    def click(obj, event):
        x, y = interactor.GetEventPosition()
        if x < window.GetSize()[0] * .29:
            return
        selected = contact_at_screen(x, y)
        if selected is not None:
            if state['edit']:
                del contacts[selected]
                mark_dirty()
            else:
                inspect_contact(selected)
            window.Render()
            return
        if not state['edit']:
            style.OnLeftButtonDown()
            return
        if not picker.Pick(x, y, 0, scene):
            return
        if picker.GetActor() == model_actor:
            p = np.array(picker.GetPickPosition())
            cell = picker.GetCellId()
            normal = mesh.GetCellData().GetNormals().GetTuple(cell)
            if normal[2] < -math.cos(math.radians(cfg['angle'])) and p[2] > 3.5:
                if all(np.linalg.norm(p-q) > .8 for q in contacts):
                    contacts.append(p)
                    mark_dirty()
            else:
                message('请点击朝下的橙色候选面；可先旋转或开启透明。')
        window.Render()

    def release(obj, event):
        if not state['edit']: style.OnLeftButtonUp()
    style.AddObserver('LeftButtonPressEvent', click)
    style.AddObserver('LeftButtonReleaseEvent', release)
    # Prevent VTK's default character shortcuts (e.g. S=surface mode)
    # from conflicting with this experiment's shortcuts.
    style.AddObserver('CharEvent', lambda *_: None)
    interactor.AddObserver('KeyPressEvent', keypress)

    def timer(obj, event):
        now = time.perf_counter()
        dt = min(now-state['last'], .15)
        state['last'] = now
        if state['rotate']:
            scene.GetActiveCamera().Azimuth(ROTATION_SPEED * dt)
            scene.ResetCameraClippingRange()
            window.Render()
    interactor.AddObserver('TimerEvent', timer)
    reset_camera()
    update_dots()
    if preview: window.SetOffScreenRendering(1)
    interactor.Initialize()
    for slider, _ in sliders.values(): slider.EnabledOn()
    generate()
    if preview:
        # Show joints and colored failure markers without the shell obscuring them.
        model_actor.GetProperty().SetOpacity(.45)
        window.Render()
        if contacts:
            graph = state['graph']
            index = min(graph.failures) if graph.failures else 0
            scene.SetWorldPoint(*contacts[index], 1)
            scene.WorldToDisplay()
            px, py, _ = scene.GetDisplayPoint()
            assert contact_at_screen(px, py) == index
            # Exercise the actual click observer used by the interactive UI.
            interactor.SetEventPosition(round(px), round(py))
            style.InvokeEvent('LeftButtonPressEvent')
            assert state['selected'] == index
            window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(Path(preview).resolve()))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        for slider, _ in sliders.values(): slider.EnabledOff()
        window.Finalize()
        return
    state['rotate'] = True
    window.Render()
    interactor.CreateRepeatingTimer(30)
    interactor.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='猫模型树状支撑交互实验')
    parser.add_argument('--preview', help='离屏保存 PNG 预览')
    main(parser.parse_args().preview)
