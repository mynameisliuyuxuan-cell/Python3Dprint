"""Local PrusaSlicer bridge: explicit demo profile, cached files, support G-code.

The generated G-code uses a virtual demo printer and must not be sent to hardware.
No model or configuration is uploaded to a service.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
import time
import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

PRUSA_EXE = Path(r'C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe')
CENTER = (110., 110.)
FILAMENT_DIAMETER = 1.75


def oriented_model(mesh, x_angle=0., y_angle=0.):
    """Rotate the already normalized cat; never rescale for individual poses."""
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    center = (points.min(axis=0)+points.max(axis=0))/2
    x, y = math.radians(x_angle), math.radians(y_angle)
    rx = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    points = (points-center) @ (ry @ rx).T
    low, high = points.min(axis=0), points.max(axis=0)
    points -= [(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]]
    data = vtk.vtkPolyData()
    data.DeepCopy(mesh)
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    data.SetPoints(vp)
    data.GetPointData().SetNormals(None)
    data.GetCellData().SetNormals(None)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(data)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def demo_config(angle, gap, style='organic'):
    if style not in ('organic','grid','snug'):
        raise ValueError('Unknown Prusa support style')
    # Full model STL has no embedded scripts. Explicitly clear all executable
    # post-processing and printer-specific start/end code in the local profile.
    settings = dict(layer_height=.3, first_layer_height=.3,
                    nozzle_diameter=.4, filament_diameter=FILAMENT_DIAMETER,
                    extrusion_multiplier=1, bed_shape='0x0,220x0,220x220,0x220',
                    gcode_flavor='marlin2', binary_gcode=0, gcode_comments=1,
                    use_relative_e_distances=0, use_volumetric_e=0,
                    gcode_label_objects='disabled',
                    support_material=1, support_material_auto=1,
                    support_material_buildplate_only=1,
                    support_material_style=style,
                    support_material_threshold=float(angle),
                    support_material_contact_distance=float(gap),
                    support_material_bottom_contact_distance=float(gap),
                    support_material_interface_layers=2,
                    support_material_spacing=2.5,
                    support_material_xy_spacing=.4,
                    support_tree_angle=35, support_tree_angle_slow=25,
                    support_tree_branch_distance=3,
                    support_tree_tip_diameter=.8,
                    fill_density='10%', perimeters=2,
                    top_solid_layers=3, bottom_solid_layers=3,
                    skirts=0, brim_width=0, threads=2,
                    start_gcode='', end_gcode='', before_layer_gcode='',
                    layer_gcode='', toolchange_gcode='', post_process='')
    return '# Virtual demo printer. DO NOT PRINT THIS G-CODE.\n' + '\n'.join(
        f'{key} = {value}' for key,value in settings.items()) + '\n'


def prepare_job(mesh, cache_root, angle, gap, style='organic'):
    if not PRUSA_EXE.is_file():
        raise FileNotFoundError(f'未找到 PrusaSlicer：{PRUSA_EXE}')
    stl_writer = vtk.vtkSTLWriter()
    # Use geometry bytes for a stable cache identifier, independent of file names.
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    config = demo_config(angle,gap,style)
    signature = hashlib.sha256(points.tobytes()+config.encode('utf-8')+
                               str(PRUSA_EXE.stat().st_mtime_ns).encode()+b'bridge-v1').hexdigest()[:20]
    directory = Path(cache_root)/signature
    directory.mkdir(parents=True,exist_ok=True)
    model_path = directory/'model.stl'
    stl_writer.SetFileName(str(model_path))
    stl_writer.SetInputData(mesh)
    stl_writer.SetFileTypeToBinary()
    if stl_writer.Write() != 1:
        raise RuntimeError('导出姿态 STL 失败')
    config_path = directory/'demo.ini'
    config_path.write_text(config,encoding='utf-8')
    (directory/'README.txt').write_text(
        '本目录仅用于支撑算法研究。虚拟打印机：220x220 平台、0.4 喷嘴、0.3 层高。\n'
        'G-code 不匹配实际硬件，请勿发送打印机。没有上传任何模型。\n',encoding='utf-8')
    return dict(directory=directory, model=model_path, config=config_path,
                gcode=directory/'demo.gcode', style=style)


def parse_support_gcode(path, center=CENTER):
    """Read only positively extruding support/interface G0/G1 moves.

    Handle G90/G91, M82/M83 and G92. Arcs with support extrusion are rejected
    rather than silently approximated or discarded.
    """
    position = dict(X=0.,Y=0.,Z=0.,E=0.)
    xyz_absolute = e_absolute = True
    role = ''
    segments, length, extrusion = [], 0., 0.
    roles = {}
    number = re.compile(r'([XYZE])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
    with Path(path).open(encoding='utf-8-sig',errors='strict') as handle:
        for raw in handle:
            command, _, comment = raw.partition(';')
            if comment.strip().startswith('TYPE:'):
                role = comment.strip()[5:].strip()
            command = command.strip()
            if not command:
                continue
            op = command.split()[0].upper()
            values = {key.upper():float(value) for key,value in number.findall(command)}
            if op == 'G90': xyz_absolute=True; continue
            if op == 'G91': xyz_absolute=False; continue
            if op == 'M82': e_absolute=True; continue
            if op == 'M83': e_absolute=False; continue
            if op == 'G92': position.update(values); continue
            if op not in ('G0','G00','G1','G01','G2','G02','G3','G03'):
                continue
            old=position.copy()
            for axis in 'XYZ':
                if axis in values:
                    position[axis]=values[axis] if xyz_absolute else old[axis]+values[axis]
            if 'E' in values:
                position['E']=values['E'] if e_absolute else old['E']+values['E']
            de=position['E']-old['E']
            support=role.casefold().startswith('support material')
            distance=math.dist([old[a] for a in 'XYZ'],[position[a] for a in 'XYZ'])
            if support and de>1e-7 and distance>1e-7:
                if op in ('G2','G02','G3','G03'):
                    raise ValueError('支撑路径包含圆弧，当前解析器不支持；请关闭圆弧拟合')
                a=(old['X']-center[0],old['Y']-center[1],old['Z'])
                b=(position['X']-center[0],position['Y']-center[1],position['Z'])
                segments.append((a,b))
                length+=distance
                extrusion+=de
                roles[role]=roles.get(role,0)+1
    return dict(segments=segments, path_length=length,
                filament_length=extrusion,
                material_volume=extrusion*math.pi*(FILAMENT_DIAMETER/2)**2,
                roles=roles)


def run_job(job, cancel=None, timeout=240):
    started=time.perf_counter()
    complete=job['directory']/'complete.json'
    cached=job['gcode'].is_file() and complete.is_file()
    if not cached:
        args=[str(PRUSA_EXE),'--datadir',str(job['directory']/'runtime'),
              '--load',str(job['config']),
              '--center',f'{CENTER[0]:g},{CENTER[1]:g}',
              '--export-gcode','--output',str(job['gcode']),str(job['model'])]
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while True:
            if cancel is not None and cancel.is_set():
                process.terminate()
                process.communicate()
                raise RuntimeError('已取消 Prusa 对照计算')
            try:
                output,_=process.communicate(timeout=.3)
                break
            except subprocess.TimeoutExpired:
                if time.perf_counter()-started > timeout:
                    process.kill()
                    process.communicate()
                    raise TimeoutError('PrusaSlicer 计算超时，已终止本次启动的进程')
        (job['directory']/'slicer.log').write_bytes(output)
        if process.returncode:
            raise RuntimeError('PrusaSlicer 失败：'+output.decode('utf-8',errors='replace')[-1800:])
        if not job['gcode'].is_file():
            raise RuntimeError('PrusaSlicer 未产生 G-code')
    result=parse_support_gcode(job['gcode'])
    result.update(seconds=time.perf_counter()-started,cached=cached,
                  gcode_path=str(job['gcode']),style=job['style'])
    if not cached:
        complete.write_text(json.dumps({key:value for key,value in result.items()
                                       if key!='segments'},ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def path_polydata(segments):
    points=vtk.vtkPoints()
    cells=vtk.vtkCellArray()
    for a,b in segments:
        start=points.InsertNextPoint(*a)
        end=points.InsertNextPoint(*b)
        cells.InsertNextCell(2)
        cells.InsertCellPoint(start)
        cells.InsertCellPoint(end)
    data=vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetLines(cells)
    return data
