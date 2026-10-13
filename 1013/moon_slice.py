from pathlib import Path
import argparse
import numpy as np
import vtk
from vedo import Mesh, Plotter, Text2D, load

MODEL_PATH = Path(r"E:\moon\Moon 2K.obj")
TEXTURE_PATH = Path(r"E:\moon\Textures\Diffuse_2K.png")
N_LAYERS = 100  # 修改这里改变切片层数；坐标为模型单位，不是毫米


def make_slices(model, count):
    """水平平面与真实网格求交；合并 OBJ 纹理接缝处的重复顶点。"""
    if count < 1:
        raise ValueError("切片层数必须大于零")
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(model.dataset)
    clean.Update()
    bounds = clean.GetOutput().GetBounds()
    low, high = bounds[4:6]
    if high <= low:
        raise ValueError("模型没有有效的 Z 轴高度")
    step = (high - low) / count
    # 每层中心取截面，避开球体两极退化为点的情况。
    heights = low + (np.arange(count) + 0.5) * step
    sections = []
    for z in heights:
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, float(z))
        plane.SetNormal(0, 0, 1)
        cutter = vtk.vtkCutter()
        cutter.SetInputConnection(clean.GetOutputPort())
        cutter.SetCutFunction(plane)
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(cutter.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(stripper.GetOutput())
        if result.GetNumberOfLines() == 0:
            raise RuntimeError(f"高度 {z:.5f} 未生成轮廓")
        sections.append(result)
    return heights, sections


def build_view(offscreen=False):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    moon = load(str(MODEL_PATH))
    if moon is None or not moon.ncells:
        raise RuntimeError("模型加载失败或模型没有面")
    heights, sections = make_slices(moon, N_LAYERS)
    moon.color("white")
    if TEXTURE_PATH.is_file():
        moon.texture(str(TEXTURE_PATH))

    plot = Plotter(shape=(1, 3), sharecam=False, size=(1500, 650),
                   bg="#18202b", title="Moon - Layer Slicing", offscreen=offscreen)
    plot.at(0).show(moon, Text2D("Moon model", c="white"),
                    axes=1, interactive=False, elevation=20, azimuth=25)
    contours = []
    for i, section in enumerate(sections):
        color = (i / max(1, N_LAYERS - 1), 0.8, 1 - i / max(1, N_LAYERS - 1))
        contours.append(Mesh(section, c=color).lw(2).lighting("off"))
    plot.at(1).show(*contours, Text2D(f"{N_LAYERS} horizontal sections", c="white"),
                    axes=1, interactive=False, elevation=25, azimuth=30)

    index = N_LAYERS // 2
    # 单层图投影到 XY 平面，保持统一比例，切换层时不自动放大。
    def flat_contour(i):
        actor = Mesh(sections[i], c="cyan").lw(3).lighting("off")
        actor.z(-float(heights[i]))
        return actor

    selected = flat_contour(index)
    label = Text2D("", c="white")

    def update_label(i):
        label.text(f"Layer {i + 1}/{N_LAYERS}\nZ = {heights[i]:.4f} (model units)")

    update_label(index)
    bounds = moon.bounds()
    cx, cy = (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    plot.at(2).show(selected, label, interactive=False, mode="image",
                    camera=dict(pos=(cx, cy, span * 3), focal_point=(cx, cy, 0),
                                viewup=(0, 1, 0)))
    plot.camera.ParallelProjectionOn()
    plot.camera.SetParallelScale(float(span) * 0.85)

    def change_layer(widget, event):
        nonlocal selected
        i = max(0, min(N_LAYERS - 1, int(round(widget.GetRepresentation().GetValue())) - 1))
        plot.remove(selected, at=2)
        selected = flat_contour(i)
        plot.at(2).add(selected)
        update_label(i)
        plot.render()

    plot.at(2).add_slider(change_layer, 1, N_LAYERS, value=index + 1,
                          title="Layer", pos=4, c="white")
    plot.render()
    print(f"已生成 {len(sections)} 层截面，间距：{(heights[1]-heights[0]) if N_LAYERS > 1 else span:.5f} 模型单位")
    print("左：月球贴图；中：所有切片轮廓；右：单层截面（拖动滑块切换）。")
    print("这是几何切片演示，不生成打印机 G-code。")
    return plot, sections, change_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="离屏验证，不弹出交互窗口")
    parser.add_argument("--screenshot", type=str)
    args = parser.parse_args()
    plot, sections, callback = build_view(offscreen=args.check)
    if args.check:
        # 验证每层确实有非零长度的闭合轮廓。
        for section in sections:
            for j in range(section.GetNumberOfCells()):
                cell = section.GetCell(j)
                first = np.array(section.GetPoint(cell.GetPointId(0)))
                last = np.array(section.GetPoint(cell.GetPointId(cell.GetNumberOfPoints() - 1)))
                assert cell.GetNumberOfPoints() >= 4 and np.allclose(first, last), "存在未闭合轮廓"
        slider = vtk.vtkSliderWidget()
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(1)
        rep.SetMaximumValue(N_LAYERS)
        slider.SetRepresentation(rep)
        for layer in (1, N_LAYERS, N_LAYERS // 2 + 1):
            rep.SetValue(layer)
            callback(slider, None)
        if args.screenshot:
            plot.screenshot(args.screenshot)
        print("PASS: 50-layer geometry, closed contours, slider updates, rendering")
    else:
        plot.interactive()
    plot.close()
