"""Pose study with local tree supports and actual PrusaSlicer support paths.

Requires support_compare_core.py and prusa_support_bridge.py in this folder.
All generated G-code is for a virtual demo printer, not real hardware.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
from datetime import datetime
import threading
import time

import numpy as np
import vtk
import support_compare_core as core
import prusa_support_bridge as bridge

CACHE = Path(__file__).resolve().parent / 'prusa_compare_cache'


def study_poses(base, tilt, angle, gap):
    """Same-size cat in three orientations; no rescaling after rotation."""
    cases=[('直立',0.,0.),('绕 X 倾斜',float(tilt),0.),('绕 Y 倾斜',0.,float(tilt))]
    cfg=dict(core.DEFAULTS,angle=float(angle),gap=float(gap))
    records=[]
    for name,x,y in cases:
        mesh=bridge.oriented_model(base,x,y)
        contacts=core.automatic_contacts(mesh,cfg['angle'],cfg['spacing'])
        started=time.perf_counter()
        graph=core.TreeBuilder(mesh,cfg).build(contacts,'tree',fallback=True)
        data,volume=graph.geometry()
        records.append(dict(name=name,x=x,y=y,mesh=mesh,contacts=contacts,
                            graph=graph,data=data,volume=volume,
                            own_seconds=time.perf_counter()-started))
    return records


def main(preview=None, tilt=20., angle=45., gap=.6):
    base=core.prepare_model()
    cfg=dict(tilt=tilt,angle=angle,gap=gap)
    state=dict(records=[],futures=[],seen=set(),busy=False,dirty=True,
               style='organic',rotate=False,transparent=True,last=time.perf_counter())
    cancel=threading.Event()
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='PrusaPose')
    window=vtk.vtkRenderWindow()
    window.SetWindowName('猫模型 · 姿态与 PrusaSlicer 支撑对照')
    window.SetSize(1740,1180)
    window.SetNumberOfLayers(2)
    background=vtk.vtkRenderer()
    background.SetBackground(.055,.07,.10)
    background.InteractiveOff()
    window.AddRenderer(background)
    overlay=vtk.vtkRenderer()
    overlay.SetLayer(1)
    overlay.InteractiveOff()
    window.AddRenderer(overlay)

    def text(renderer,content,x,y,size=16):
        actor=vtk.vtkTextActor()
        actor.SetInput(content)
        actor.SetPosition(x,y)
        core.set_chinese_font(actor.GetTextProperty())
        actor.GetTextProperty().SetFontSize(size)
        actor.GetTextProperty().SetColor(.95,.95,.95)
        renderer.AddViewProp(actor)
        return actor

    text(overlay,'同尺寸猫模型 · 三种姿态 · 自制支撑 / PrusaSlicer 实际支撑路径',18,1144,22)
    status=text(overlay,'准备计算……',18,996,16)
    footer=text(overlay,
        '上排：自制树状，按各姿态独立采样；下排：Prusa 实际挤出路径，绿色为支撑及接触层。\n'
        'G：重算全部　P：重算/加载 Prusa　K：普通/有机支撑　M：模型透明　空格：旋转　R：视角　X：保存 CSV\n'
        '自制实体体积与 Prusa 挤出量口径不同，不直接等同；虚拟打印机输出仅供研究，勿发送实机。',18,12,16)
    camera=vtk.vtkCamera()
    views=[]
    for row in range(2):
        for col in range(3):
            renderer=vtk.vtkRenderer()
            renderer.SetViewport(col/3,.47 if row==0 else .10,(col+1)/3,.84 if row==0 else .47)
            renderer.SetBackground(.035+.009*col,.045,.065)
            renderer.SetActiveCamera(camera)
            window.AddRenderer(renderer)
            model=core.actor_for(vtk.vtkPolyData(),(.92,.10,.14),.4)
            model.GetMapper().SetScalarModeToUseCellData()
            model.GetMapper().SetColorModeToDirectScalars()
            model.GetMapper().ScalarVisibilityOn()
            supports=core.actor_for(vtk.vtkPolyData(),(.10,.95,.30))
            supports.GetProperty().SetLineWidth(1.6)
            renderer.AddActor(model)
            renderer.AddActor(supports)
            bed=core.actor_for(vtk.vtkPolyData(),(.18,.21,.25))
            renderer.AddActor(bed)
            title=text(renderer,'',12,0,18)
            title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            title.SetPosition(.025,.93)
            metrics=text(renderer,'',12,10,15)
            views.append(dict(renderer=renderer,model=model,supports=supports,
                              bed=bed,title=title,metrics=metrics))

    def reset_camera():
        if not state['records']:
            return
        bounds=np.array([r['mesh'].GetBounds() for r in state['records']])
        union=[bounds[:,0].min(),bounds[:,1].max(),bounds[:,2].min(),bounds[:,3].max(),
               bounds[:,4].min(),bounds[:,5].max()]
        camera.SetPosition(190,-150,120)
        camera.SetFocalPoint(0,0,(union[4]+union[5])/2)
        camera.SetViewUp(0,0,1)
        camera.ParallelProjectionOn()
        views[0]['renderer'].ResetCamera(union)
        camera.Zoom(.88)

    def apply_prusa(index,result):
        record=state['records'][index]
        record['prusa']=result
        view=views[3+index]
        view['supports'].GetMapper().SetInputData(bridge.path_polydata(result['segments']))
        mode='有机树状' if state['style']=='organic' else '普通网格'
        cached='缓存加载' if result['cached'] else '切片＋读取'
        view['title'].SetInput(f'{record["name"]}｜Prusa {mode}')
        view['metrics'].SetInput(
            f'支撑路径 {result["path_length"]/1000:.2f} 米；路径段 {len(result["segments"])}\n'
            f'支撑挤出估算 {result["material_volume"]:.0f} 立方毫米；{cached} {result["seconds"]:.1f} 秒\n'
            '0.4 喷嘴 / 0.3 层高 / 平台支撑；含支撑接触层')
        print(record['name'],'Prusa segments',len(result['segments']),
              'support material mm3',round(result['material_volume'],1),flush=True)

    def finished_summary():
        valid=[r for r in state['records'] if 'prusa' in r]
        if len(valid)==3:
            best=min(valid,key=lambda r:r['prusa']['material_volume'])
            status.SetInput(f'Prusa 参考：当前参数下支撑挤出估算最少的是“{best["name"]}”。'
                            '角度/间隙一致，接触点与密度策略各自决定。')
        else:
            status.SetInput('部分 Prusa 计算失败，窗口保留错误信息；按 P 重试。')

    def start_prusa():
        if state['busy']:
            return
        if state['dirty']:
            status.SetInput('姿态/阈值已改变，请按 G 重新计算全部。')
            return
        state['busy']=True
        state['seen']=set()
        state['futures']=[]
        try:
            for i,record in enumerate(state['records']):
                record.pop('prusa',None)
                view=views[i+3]
                view['supports'].GetMapper().SetInputData(vtk.vtkPolyData())
                view['metrics'].SetInput('Prusa 正在排队/计算……（可继续旋转观察）')
                job=bridge.prepare_job(record['mesh'],CACHE,cfg['angle'],cfg['gap'],state['style'])
                state['futures'].append(pool.submit(bridge.run_job,job,cancel))
            status.SetInput('后台计算 PrusaSlicer：0/3；首次需生成，之后可复用缓存。')
        except Exception as exc:
            state['busy']=False
            status.SetInput(f'启动 Prusa 失败：{exc}')

    def poll():
        if not state['busy']:
            return
        for i,future in enumerate(state['futures']):
            if i in state['seen'] or not future.done():
                continue
            state['seen'].add(i)
            try:
                apply_prusa(i,future.result())
            except Exception as exc:
                state['records'][i]['prusa_error']=str(exc)
                views[i+3]['metrics'].SetInput('Prusa 计算失败：\n'+str(exc)[:150])
                print('Prusa error',i,str(exc),flush=True)
        count=len(state['seen'])
        if count==len(state['futures']):
            state['busy']=False
            finished_summary()
        else:
            status.SetInput(f'后台计算 PrusaSlicer：{count}/3；计算中暂不更改参数。')
        window.Render()

    def generate():
        if state['busy']:
            status.SetInput('Prusa 正在计算，请完成后再修改姿态。')
            return
        state['rotate']=False
        status.SetInput('正在计算三种姿态的自制树状支撑……')
        window.Render()
        state['records']=study_poses(base,cfg['tilt'],cfg['angle'],cfg['gap'])
        state['dirty']=False
        for i,record in enumerate(state['records']):
            mesh=record['mesh']
            core.highlight_faces(mesh,cfg['angle'],True)
            bounds=mesh.GetBounds()
            bed=vtk.vtkPlaneSource()
            bed.SetOrigin(bounds[0]-8,bounds[2]-8,-.1)
            bed.SetPoint1(bounds[1]+8,bounds[2]-8,-.1)
            bed.SetPoint2(bounds[0]-8,bounds[3]+8,-.1)
            bed.Update()
            for row in (0,1):
                view=views[row*3+i]
                view['model'].GetMapper().SetInputData(mesh)
                view['bed'].GetMapper().SetInputData(bed.GetOutput())
                view['supports'].GetMapper().SetInputData(vtk.vtkPolyData())
                view['title'].SetInput(f'{record["name"]}（X {record["x"]:g}°，Y {record["y"]:g}°）｜'+
                                       ('自制树状' if row==0 else 'Prusa'))
            view=views[i]
            view['supports'].GetMapper().SetInputData(record['data'])
            graph=record['graph']
            view['metrics'].SetInput(
                f'接通 {len(graph.solved)}/{len(record["contacts"])}；未解决 {len(graph.failures)}；根部 {len(graph.roots)}\n'
                f'管段体积累加约 {record["volume"]:.0f} 立方毫米（含交叠）\n'
                f'模型高度 {bounds[5]:.1f} 毫米；生成 {record["own_seconds"]:.2f} 秒（整体尺寸未缩放）')
            print(record['name'],'own',len(graph.solved),'/',len(record['contacts']),
                  'volume',round(record['volume'],1),flush=True)
        reset_camera()
        start_prusa()

    interactor=vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    style=vtk.vtkInteractorStyleTrackballCamera()
    style.AddObserver('CharEvent',lambda *_:None)
    interactor.SetInteractorStyle(style)
    sliders={}
    for key,title,low,high,index,decimals in [
        ('tilt','倾斜角度（X / Y，度）',0,40,0,0),
        ('angle','悬垂角度阈值（度）',5,85,1,0),
        ('gap','接触间隙（毫米）',.3,1.2,2,2)]:
        rep=vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(low)
        rep.SetMaximumValue(high)
        rep.SetValue(cfg[key])
        rep.SetTitleText(title)
        rep.SetLabelFormat('{:.'+str(decimals)+'f}')
        for prop in (rep.GetTitleProperty(),rep.GetLabelProperty()):
            core.set_chinese_font(prop)
            prop.SetColor(.95,.95,.95)
        rep.SetTitleHeight(.018)
        rep.SetLabelHeight(.018)
        rep.GetSliderProperty().SetColor(.1,.95,.3)
        for coordinate,x in ((rep.GetPoint1Coordinate(),index/3+.06),
                             (rep.GetPoint2Coordinate(),(index+1)/3-.06)):
            coordinate.SetCoordinateSystemToNormalizedViewport()
            coordinate.SetValue(x,.91)
        slider=vtk.vtkSliderWidget()
        slider.SetInteractor(interactor)
        slider.SetDefaultRenderer(overlay)
        slider.SetCurrentRenderer(overlay)
        slider.SetRepresentation(rep)
        slider.SetAnimationModeToJump()
        def changed(obj,event,name=key,precision=decimals):
            if state['busy']:
                obj.GetRepresentation().SetValue(cfg[name])
                return
            value=round(obj.GetRepresentation().GetValue(),precision)
            if value!=cfg[name]:
                cfg[name]=value
                state.update(dirty=True,rotate=False)
                for view in views:
                    view['supports'].GetMapper().SetInputData(vtk.vtkPolyData())
                    view['metrics'].SetInput('参数已修改，请按 G 重新生成。')
                status.SetInput('参数已修改。按 G：重新旋转、采样、生成并调用 Prusa。')
                window.Render()
        slider.AddObserver('InteractionEvent',changed)
        slider.AddObserver('EndInteractionEvent',changed)
        sliders[key]=(slider,rep)

    def export_csv():
        if state['busy'] or state['dirty']:
            status.SetInput('请等待当前计算完成，且参数未修改后再导出。')
            return
        directory=Path(__file__).resolve().parent/'comparison_results'
        directory.mkdir(exist_ok=True)
        path=directory/f'pose_prusa_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
        with path.open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.writer(handle)
            writer.writerow(['姿态','X倾角','Y倾角','悬垂阈值','接触间隙','自制接触点',
                             '自制接通','自制根部','自制管段体积累加_含交叠',
                             'Prusa类型','Prusa路径长度mm','Prusa支撑挤出估算mm3','Prusa段数','Prusa错误'])
            for r in state['records']:
                p=r.get('prusa',{})
                writer.writerow([r['name'],r['x'],r['y'],cfg['angle'],cfg['gap'],len(r['contacts']),
                                 len(r['graph'].solved),len(r['graph'].roots),r['volume'],state['style'],
                                 p.get('path_length',''),p.get('material_volume',''),len(p.get('segments',[])),
                                 r.get('prusa_error','')])
        status.SetInput(f'已保存：comparison_results/{path.name}')

    def keypress(obj,event):
        key=obj.GetKeySym().lower()
        if key=='g': generate()
        elif key=='p': start_prusa()
        elif key=='k':
            if not state['busy']:
                state['style']='grid' if state['style']=='organic' else 'organic'
                start_prusa()
        elif key=='m':
            state['transparent']=not state['transparent']
            for view in views:
                view['model'].GetProperty().SetOpacity(.4 if state['transparent'] else 1)
        elif key=='space': state['rotate']=not state['rotate']
        elif key=='r': reset_camera()
        elif key=='x': export_csv()
        window.Render()

    def timer(obj,event):
        poll()
        now=time.perf_counter()
        dt=min(now-state['last'],.15)
        state['last']=now
        if state['rotate']:
            camera.Azimuth(2*dt)
            for view in views:
                view['renderer'].ResetCameraClippingRange()
            window.Render()
    interactor.AddObserver('KeyPressEvent',keypress)
    interactor.AddObserver('TimerEvent',timer)
    if preview:
        window.SetOffScreenRendering(1)
    interactor.Initialize()
    for slider,_ in sliders.values(): slider.EnabledOn()
    try:
        generate()
        if preview:
            while state['busy']:
                poll()
                time.sleep(.05)
            if not all('prusa' in r for r in state['records']):
                raise RuntimeError('部分 Prusa 对照失败，请查看日志')
            window.Render()
            capture=vtk.vtkWindowToImageFilter()
            capture.SetInput(window)
            capture.Update()
            writer=vtk.vtkPNGWriter()
            writer.SetFileName(str(Path(preview).resolve()))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            for slider,_ in sliders.values(): slider.EnabledOff()
            window.Finalize()
        else:
            interactor.CreateRepeatingTimer(80)
            interactor.Start()
    finally:
        cancel.set()
        pool.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='姿态变化与 PrusaSlicer 本地支撑对照')
    parser.add_argument('--preview',help='实际计算 Prusa 并保存 PNG 预览')
    parser.add_argument('--tilt',type=float,default=20.)
    args=parser.parse_args()
    main(args.preview,args.tilt)
