"""三种支撑策略使用同一组接触点进行对照，复用 project2 的几何算法。"""
from pathlib import Path
import argparse
import csv
from datetime import datetime
import time

import vtk
import support_compare_core as engine


STRATEGIES = [('独立直柱', 'straight', False),
              ('树状合并', 'tree', False),
              ('树状＋拆分重试', 'tree', True)]
RETRY_CASE = dict(engine.DEFAULTS, angle=55., spacing=4., merge=25., diameter=1.4, gap=.6)


def run_comparison(mesh, settings):
    """接触点只采样一次；三次运行不更改模型、参数和接触点。"""
    contacts = engine.automatic_contacts(mesh, settings['angle'], settings['spacing'])
    results = []
    for name, mode, fallback in STRATEGIES:
        started = time.perf_counter()
        graph = engine.TreeBuilder(mesh, settings).build(contacts, mode, fallback=fallback)
        data, volume = graph.geometry()
        elapsed = time.perf_counter() - started
        results.append(dict(name=name, graph=graph, data=data, volume=volume,
                            seconds=elapsed, contacts=len(contacts)))
        print(f'{name}: {len(graph.solved)}/{len(contacts)}, roots={len(graph.roots)}, '
              f'volume={volume:.1f}, recovered={len(graph.recovered)}')
    return contacts, results


def export_results(settings, results):
    directory = Path(__file__).resolve().parent / 'comparison_results'
    directory.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = directory / f'support_compare_{stamp}.csv'
    fields = ['方案', '接触点总数', '成功连接', '未解决', '平台根部', '有效合并',
              '拆分次数', '恢复点数', '体积累加_立方毫米_含交叠', '生成结构耗时_秒',
              '悬垂角度', '接触点间距', '合并距离', '分支倾角', '基础直径', '接触间隙']
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for result in results:
            graph = result['graph']
            writer.writerow([result['name'], result['contacts'], len(graph.solved),
                             len(graph.failures), len(graph.roots), graph.joins,
                             graph.splits, len(graph.recovered), round(result['volume'], 3),
                             round(result['seconds'], 4),
                             *[settings[k] for k in ('angle', 'spacing', 'merge', 'branch_angle', 'diameter', 'gap')]])
    return path


def main(preview=None, case='default'):
    settings = (RETRY_CASE if case == 'retry' else engine.DEFAULTS).copy()
    state = dict(rotate=True, transparent=True, highlight=True, dirty=True,
                 last=time.perf_counter(), results=[], contacts=[])
    mesh = engine.prepare_model()
    window = vtk.vtkRenderWindow()
    window.SetWindowName('猫模型 · 三种支撑策略对照实验')
    window.SetSize(1620, 1080)
    window.SetNumberOfLayers(2)
    background = vtk.vtkRenderer()
    background.SetBackground(.065, .08, .11)
    background.InteractiveOff()
    window.AddRenderer(background)
    controls = vtk.vtkRenderer()
    controls.SetViewport(0, 0, 1, 1)
    controls.SetLayer(1)
    controls.InteractiveOff()
    controls.SetBackground(.065, .08, .11)
    window.AddRenderer(controls)

    def text(renderer, content, x, y, size=17):
        label = vtk.vtkTextActor()
        label.SetInput(content)
        label.SetPosition(x, y)
        engine.set_chinese_font(label.GetTextProperty())
        label.GetTextProperty().SetFontSize(size)
        label.GetTextProperty().SetColor(.95, .95, .95)
        renderer.AddViewProp(label)
        return label

    text(controls, '同一组接触点 · 三种支撑策略对照', 18, 1040, 22)
    status = text(controls, '', 18, 735, 16)
    scenes, model_actors, support_actors, dot_mappers, labels = [], [], [], [], []
    camera = vtk.vtkCamera()
    for i, (name, _, _) in enumerate(STRATEGIES):
        scene = vtk.vtkRenderer()
        scene.SetViewport(i / 3, .17, (i + 1) / 3, .67)
        scene.SetBackground(.035 + .008*i, .045, .065)
        scene.SetActiveCamera(camera)
        window.AddRenderer(scene)
        model_actor = engine.actor_for(mesh, (.92, .10, .14), .45)
        model_actor.GetMapper().SetScalarModeToUseCellData()
        model_actor.GetMapper().SetColorModeToDirectScalars()
        model_actor.GetMapper().ScalarVisibilityOn()
        support_actor = engine.actor_for(vtk.vtkPolyData(), (.1, .95, .3))
        scene.AddActor(model_actor)
        scene.AddActor(support_actor)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(.65)
        sphere.SetThetaResolution(10)
        sphere.SetPhiResolution(8)
        dot_mapper = vtk.vtkGlyph3DMapper()
        dot_mapper.SetSourceConnection(sphere.GetOutputPort())
        dot_mapper.ScalingOff()
        dot_mapper.SetColorModeToDirectScalars()
        dots = vtk.vtkActor()
        dots.SetMapper(dot_mapper)
        scene.AddActor(dots)
        bounds = mesh.GetBounds()
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(bounds[0]-8, bounds[2]-8, -.1)
        plane.SetPoint1(bounds[1]+8, bounds[2]-8, -.1)
        plane.SetPoint2(bounds[0]-8, bounds[3]+8, -.1)
        plane.Update()
        scene.AddActor(engine.actor_for(plane.GetOutput(), (.18, .21, .25)))
        title = text(scene, name, 12, 0, 21)
        title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        title.SetPosition(.03, .93)
        metrics = vtk.vtkRenderer()
        metrics.SetViewport(i / 3, 0, (i + 1) / 3, .17)
        metrics.SetBackground(.055, .065, .09)
        window.AddRenderer(metrics)
        label = text(metrics, '待生成', 16, 20, 17)
        scenes.append(scene)
        model_actors.append(model_actor)
        support_actors.append(support_actor)
        dot_mappers.append(dot_mapper)
        labels.append(label)

    def reset_camera():
        camera.SetPosition(190, -150, 110)
        camera.SetFocalPoint(0, 0, 32)
        camera.SetViewUp(0, 0, 1)
        scenes[0].ResetCamera(mesh.GetBounds())

    def mark_dirty():
        state['dirty'] = True
        state['rotate'] = False
        engine.highlight_faces(mesh, settings['angle'], state['highlight'])
        for actor in support_actors:
            actor.SetVisibility(False)
        for mapper in dot_mappers:
            mapper.SetInputData(vtk.vtkPolyData())
        for label in labels:
            label.SetInput('参数已变更，请按 G 同时重新生成。')
        status.SetInput('G：重算三组　F：拆分重试示例　D：默认参数　M：透明度　H：悬垂面　空格：旋转　X：导出 CSV')

    def generate():
        state['rotate'] = False
        status.SetInput('正在用同一组接触点生成三种方案，请稍候……')
        window.Render()
        contacts, results = run_comparison(mesh, settings)
        state.update(contacts=contacts, results=results, dirty=False)
        engine.highlight_faces(mesh, settings['angle'], state['highlight'])
        for result, actor, mapper, label in zip(results, support_actors, dot_mappers, labels):
            graph = result['graph']
            actor.GetMapper().SetInputData(result['data'])
            actor.SetVisibility(True)
            mapper.SetInputData(engine.glyph_contacts(contacts, graph.solved))
            total = len(contacts)
            rate = 100 * len(graph.solved) / total if total else 0
            label.SetInput(
                f'成功 {len(graph.solved)}/{total}（{rate:.1f}%）；未解决 {len(graph.failures)}\n'
                f'平台根部 {len(graph.roots)}；有效合并 {graph.joins}\n'
                f'拆分 {graph.splits} 次；恢复 {len(graph.recovered)} 个点\n'
                f'体积累加约 {result["volume"]:.0f} 立方毫米（含交叠）\n'
                f'结构生成耗时 {result["seconds"]:.3f} 秒（不含共享采样）')
        status.SetInput(
            f'共同接触点 {len(contacts)} 个。先看成功率，再比较体积；并非精确耗材或强度评估。\n'
            'G：重算　F：拆分示例　D：默认　M：透明　H：高亮　空格：旋转　R：视角　X：保存 CSV')
        window.Render()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.AddObserver('CharEvent', lambda *_: None)
    interactor.SetInteractorStyle(style)
    sliders = {}
    specs = [('angle', '悬垂角度（度）', 5, 85, 0, 0),
             ('spacing', '接触点间距（毫米）', 3, 10, 1, 1),
             ('merge', '分支合并距离（毫米）', 3, 25, 2, 1),
             ('branch_angle', '最大分支倾角（相对竖直）', 10, 55, 3, 0),
             ('diameter', '单枝基础直径（毫米）', .4, 1.6, 4, 2),
             ('gap', '接触间隙（毫米）', .3, 1.5, 5, 2)]
    for key, title, low, high, position, decimals in specs:
        col, row = position % 3, position // 3
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(low)
        rep.SetMaximumValue(high)
        rep.SetValue(settings[key])
        rep.SetTitleText(title)
        rep.SetLabelFormat('{:.' + str(decimals) + 'f}')
        for prop in (rep.GetTitleProperty(), rep.GetLabelProperty()):
            engine.set_chinese_font(prop)
            prop.SetColor(.95, .95, .95)
        rep.SetTitleHeight(.019)
        rep.SetLabelHeight(.018)
        rep.GetSliderProperty().SetColor(.1, .95, .3)
        for coord, x in ((rep.GetPoint1Coordinate(), col/3+.045),
                         (rep.GetPoint2Coordinate(), (col+1)/3-.045)):
            coord.SetCoordinateSystemToNormalizedViewport()
            coord.SetValue(x, .90 if row == 0 else .80)
        slider = vtk.vtkSliderWidget()
        slider.SetInteractor(interactor)
        slider.SetDefaultRenderer(controls)
        slider.SetCurrentRenderer(controls)
        slider.SetRepresentation(rep)
        slider.SetAnimationModeToJump()
        def changed(obj, event, name=key, precision=decimals):
            value = round(obj.GetRepresentation().GetValue(), precision)
            if value != settings[name]:
                settings[name] = value
                mark_dirty()
                window.Render()
        slider.AddObserver('InteractionEvent', changed)
        slider.AddObserver('EndInteractionEvent', changed)
        sliders[key] = slider, rep

    def load_case(values):
        settings.update(values)
        for key, (_, rep) in sliders.items():
            rep.SetValue(settings[key])
        generate()

    def keypress(obj, event):
        key = obj.GetKeySym().lower()
        if key == 'g': generate()
        elif key == 'f': load_case(RETRY_CASE)
        elif key == 'd': load_case(engine.DEFAULTS)
        elif key == 'm':
            state['transparent'] = not state['transparent']
            for actor in model_actors:
                actor.GetProperty().SetOpacity(.45 if state['transparent'] else 1)
        elif key == 'h':
            state['highlight'] = not state['highlight']
            engine.highlight_faces(mesh, settings['angle'], state['highlight'])
        elif key == 'space': state['rotate'] = not state['rotate']
        elif key == 'r': reset_camera()
        elif key == 'x':
            if state['dirty']:
                status.SetInput('参数已改变，请先按 G 生成后再保存，避免记录不一致。')
            else:
                path = export_results(settings, state['results'])
                status.SetInput(f'对比结果已保存：comparison_results/{path.name}\nG：重算　F：拆分示例　D：默认　M：透明　空格：旋转')
        window.Render()

    def timer(obj, event):
        now = time.perf_counter()
        dt = min(now-state['last'], .15)
        state['last'] = now
        if state['rotate']:
            camera.Azimuth(engine.ROTATION_SPEED * dt)
            for renderer in scenes:
                renderer.ResetCameraClippingRange()
            window.Render()
    interactor.AddObserver('KeyPressEvent', keypress)
    interactor.AddObserver('TimerEvent', timer)
    reset_camera()
    if preview:
        window.SetOffScreenRendering(1)
    interactor.Initialize()
    for slider, _ in sliders.values():
        slider.EnabledOn()
    generate()
    if preview:
        # Exercise a real slider callback, then restore the original snapshot.
        slider, rep = sliders['angle']
        value = settings['angle']
        rep.SetValue(min(85, value+1))
        slider.InvokeEvent('InteractionEvent')
        assert state['dirty']
        rep.SetValue(value)
        slider.InvokeEvent('InteractionEvent')
        generate()
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
        return
    state['rotate'] = True
    interactor.CreateRepeatingTimer(30)
    interactor.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='同一接触点集合的三策略支撑对照')
    parser.add_argument('--preview', help='离屏生成 PNG 预览')
    parser.add_argument('--case', choices=['default','retry'], default='default')
    args = parser.parse_args()
    main(args.preview, args.case)
