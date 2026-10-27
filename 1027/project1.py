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


def scan_model(mesh, spacing=SUPPORT_SPACING):
    """Cache the closest upward ray hits once, before slider interactions."""
    if spacing <= 0:
        raise ValueError('支撑间距必须大于零')
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.GetBounds()
    samples = []
    for x in np.arange(xmin + spacing / 2, xmax, spacing):
        for y in np.arange(ymin + spacing / 2, ymax, spacing):
            t, sub, cell = vtk.reference(0.0), vtk.reference(0), vtk.reference(0)
            point, local = [0.0] * 3, [0.0] * 3
            hit = locator.IntersectWithLine(
                (x, y, zmin - 1), (x, y, zmax + 1), 1e-6,
                t, point, local, sub, cell,
            )
            if hit:
                samples.append((float(x), float(y), float(point[2]), int(cell)))
    return samples


def generate_supports(mesh, threshold=OVERHANG_ANGLE, samples=None,
                      radius=SUPPORT_RADIUS, gap=CONTACT_GAP):
    """Grid rays select the first surface above the bed.

    Only downward overhang faces receive supports. Selecting the first hit
    prevents a support from continuing through a foot or other model surface.
    Overhangs blocked by lower model geometry are deliberately not handled by
    this build-plate-only demonstrator.
    """
    if radius <= 0:
        raise ValueError('支撑间距、半径必须大于零')
    if not 0 < threshold < 90 or gap < 0:
        raise ValueError('悬垂角度须在 0～90° 之间，接触间隙不能为负')
    normals = vtk_to_numpy(mesh.GetCellData().GetNormals())
    needs_support = normals[:, 2] < -math.cos(math.radians(threshold))
    if samples is None:
        samples = scan_model(mesh)
    xy_points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    append = vtk.vtkAppendPolyData()
    columns = []
    for x, y, roof_z, cell in samples:
            top = roof_z - gap
            if not needs_support[cell] or top < MIN_SUPPORT_HEIGHT:
                continue
            tip_height = min(TIP_HEIGHT, top * 0.4)
            shaft_top = top - tip_height
            a = xy_points.InsertNextPoint(float(x), float(y), 0)
            b = xy_points.InsertNextPoint(float(x), float(y), shaft_top)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(a)
            lines.InsertCellPoint(b)
            # Cone apex points upward; base meets the vertical tube.
            cone = vtk.vtkConeSource()
            cone.SetDirection(0, 0, 1)
            cone.SetCenter(float(x), float(y), shaft_top + tip_height / 2)
            cone.SetRadius(radius)
            cone.SetHeight(tip_height)
            cone.SetResolution(12)
            cone.CappingOn()
            cone.Update()
            append.AddInputData(cone.GetOutput())
            columns.append((float(x), float(y), float(top), float(roof_z), int(cell)))
    if columns:
        data = vtk.vtkPolyData()
        data.SetPoints(xy_points)
        data.SetLines(lines)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(data)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(12)
        tube.CappingOn()
        tube.Update()
        append.AddInputData(tube.GetOutput())
        append.Update()
        output = vtk.vtkPolyData()
        output.DeepCopy(append.GetOutput())
    else:
        output = vtk.vtkPolyData()
    print(f'检测到朝下悬垂三角面：{int(needs_support.sum())}')
    print(f'生成平台直柱支撑：{len(columns)} 根')
    return output, columns



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


def main(preview=None):
    mesh = prepare_model()
    settings = dict(angle=OVERHANG_ANGLE, spacing=SUPPORT_SPACING,
                    diameter=2 * SUPPORT_RADIUS, gap=CONTACT_GAP)
    state = dict(rotate=True, transparent=False, highlight=True,
                 last=time.perf_counter())
    samples = scan_model(mesh, settings['spacing'])
    window = vtk.vtkRenderWindow()
    window.SetWindowName('猫模型 · 交互式支撑实验')
    window.SetSize(1400, 950)
    scene, panel = vtk.vtkRenderer(), vtk.vtkRenderer()
    scene.SetViewport(0.29, 0, 1, 1)
    panel.SetViewport(0, 0, 0.29, 1)
    scene.SetBackground(0.035, 0.045, 0.065)
    panel.SetBackground(0.065, 0.08, 0.11)
    window.AddRenderer(scene)
    window.AddRenderer(panel)
    model_actor = actor_for(mesh, (0.92, 0.10, 0.14))
    model_actor.GetMapper().SetScalarModeToUseCellData()
    model_actor.GetMapper().SetColorModeToDirectScalars()
    model_actor.GetMapper().ScalarVisibilityOn()
    support_actor = actor_for(vtk.vtkPolyData(), (0.10, 0.95, 0.30))
    scene.AddActor(model_actor)
    scene.AddActor(support_actor)
    bounds = mesh.GetBounds()
    bed = vtk.vtkPlaneSource()
    bed.SetOrigin(bounds[0] - 8, bounds[2] - 8, -0.1)
    bed.SetPoint1(bounds[1] + 8, bounds[2] - 8, -0.1)
    bed.SetPoint2(bounds[0] - 8, bounds[3] + 8, -0.1)
    bed.Update()
    scene.AddActor(actor_for(bed.GetOutput(), (0.18, 0.21, 0.25)))

    def text(renderer, content, x, y, size=17):
        label = vtk.vtkTextActor()
        label.SetInput(content)
        label.SetPosition(x, y)
        label.GetTextProperty().SetFontSize(size)
        set_chinese_font(label.GetTextProperty())
        label.GetTextProperty().SetColor(0.95, 0.95, 0.95)
        renderer.AddViewProp(label)
        return label

    text(panel, '支撑实验室\n拖动滑块，实时比较效果', 20, 860, 22)
    info = text(scene, '', 16, 18, 17)
    text(panel, '红色：模型\n橙色：符合角度条件的悬垂面\n绿色：支撑\n\n'
         'H：悬垂面高亮开关\nM：模型透明度\nS：支撑显示开关\n'
         '空格：自动旋转开关\nR：恢复视角\nD：恢复默认参数', 20, 18, 14)

    def rebuild(rescan=False):
        nonlocal samples
        if rescan:
            samples = scan_model(mesh, settings['spacing'])
        data, columns = generate_supports(
            mesh, settings['angle'], samples,
            radius=settings['diameter'] / 2, gap=settings['gap'])
        support_actor.GetMapper().SetInputData(data)
        selected = highlight_faces(mesh, settings['angle'], state['highlight'])
        r = settings['diameter'] / 2
        volume = sum(math.pi * r * r * (top - min(TIP_HEIGHT, top * .4)
                     + min(TIP_HEIGHT, top * .4) / 3) for _, _, top, _, _ in columns)
        info.SetInput(
            f'支撑：{len(columns)} 根；悬垂三角面：{selected} 个\n'
            f"悬垂面与水平面夹角 < {settings['angle']:g}°；支撑间距：{settings['spacing']:g} 毫米\n"
            f"支撑柱直径：{settings['diameter']:g} 毫米；接触间隙：{settings['gap']:g} 毫米\n"
            f'支撑几何体积约 {volume:.0f} 立方毫米（非实际耗材量）\n'
            '橙色：仅按角度识别；绿色：可从平台直接支撑。')
        return len(columns), selected, volume

    def reset_camera():
        camera = scene.GetActiveCamera()
        camera.SetPosition(190, -150, 110)
        camera.SetFocalPoint(0, 0, MODEL_HEIGHT * .4)
        camera.SetViewUp(0, 0, 1)
        scene.ResetCamera()

    reset_camera()
    rebuild()
    if preview:
        window.SetOffScreenRendering(1)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    sliders = {}

    def on_start(obj, event):
        state['resume_rotation'] = state['rotate']
        state['rotate'] = False

    def make_callback(key, decimals):
        def changed(obj, event):
            value = round(obj.GetRepresentation().GetValue(), decimals)
            if value != settings[key]:
                settings[key] = value
                rebuild(rescan=(key == 'spacing'))
                window.Render()
        return changed

    specs = [
        ('angle', '悬垂面角度阈值（度）', 5, 85, .79, 0),
        ('spacing', '支撑间距（毫米）', 1.5, 8, .60, 1),
        ('diameter', '支撑柱直径（毫米）', .4, 2.4, .41, 2),
        ('gap', '接触间隙（毫米）', 0, 1.0, .22, 2),
    ]
    for key, title, minimum, maximum, y, decimals in specs:
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(minimum)
        rep.SetMaximumValue(maximum)
        rep.SetValue(settings[key])
        rep.SetTitleText(title)
        set_chinese_font(rep.GetTitleProperty())
        set_chinese_font(rep.GetLabelProperty())
        rep.SetLabelFormat('{:.' + str(decimals) + 'f}')
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(.035, y)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(.255, y)
        rep.SetSliderLength(.025)
        rep.SetSliderWidth(.035)
        rep.SetTitleHeight(.022)
        rep.SetLabelHeight(.020)
        rep.GetTitleProperty().SetColor(.95, .95, .95)
        rep.GetLabelProperty().SetColor(.95, .95, .95)
        rep.GetSliderProperty().SetColor(.1, .95, .3)
        slider = vtk.vtkSliderWidget()
        slider.SetInteractor(interactor)
        slider.SetCurrentRenderer(panel)
        slider.SetRepresentation(rep)
        slider.SetAnimationModeToJump()
        callback = make_callback(key, decimals)
        slider.AddObserver('StartInteractionEvent', on_start)
        slider.AddObserver('InteractionEvent', callback)

        def end(obj, event, cb=callback):
            cb(obj, event)
            state['rotate'] = state.get('resume_rotation', False)
        slider.AddObserver('EndInteractionEvent', end)
        sliders[key] = (slider, rep)

    def on_key(obj, event):
        key = obj.GetKeySym().lower()
        if key == 'h':
            state['highlight'] = not state['highlight']
            highlight_faces(mesh, settings['angle'], state['highlight'])
        elif key == 'm':
            state['transparent'] = not state['transparent']
            model_actor.GetProperty().SetOpacity(.25 if state['transparent'] else 1)
        elif key == 's':
            support_actor.SetVisibility(not support_actor.GetVisibility())
        elif key == 'space':
            state['rotate'] = not state['rotate']
        elif key == 'r':
            reset_camera()
        elif key == 'd':
            settings.update(angle=OVERHANG_ANGLE, spacing=SUPPORT_SPACING,
                            diameter=2 * SUPPORT_RADIUS, gap=CONTACT_GAP)
            for name, (_, rep) in sliders.items():
                rep.SetValue(settings[name])
            rebuild(rescan=True)
        window.Render()

    def on_timer(obj, event):
        now = time.perf_counter()
        dt = min(now - state['last'], .15)
        state['last'] = now
        if state['rotate']:
            scene.GetActiveCamera().Azimuth(ROTATION_SPEED * dt)
            scene.ResetCameraClippingRange()
            window.Render()

    interactor.AddObserver('KeyPressEvent', on_key)
    interactor.AddObserver('TimerEvent', on_timer)
    interactor.Initialize()
    for slider, _ in sliders.values():
        slider.EnabledOn()
    window.Render()
    if preview:
        # Exercise each real slider callback and restore defaults afterwards.
        for name, value in [('angle', 65), ('spacing', 5),
                            ('diameter', 1.8), ('gap', .6)]:
            slider, rep = sliders[name]
            old = settings[name]
            rep.SetValue(value)
            slider.InvokeEvent('InteractionEvent')
            rep.SetValue(old)
            slider.InvokeEvent('InteractionEvent')
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(Path(preview).resolve()))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        for slider, _ in sliders.values():
            slider.EnabledOff()
        window.Finalize()
        print(f'预览已保存：{preview}')
        return
    interactor.CreateRepeatingTimer(30)
    interactor.Start()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='猫模型简单平台支撑演示')
    parser.add_argument('--preview', help='保存离屏预览 PNG，不启动交互窗口')
    main(parser.parse_args().preview)
