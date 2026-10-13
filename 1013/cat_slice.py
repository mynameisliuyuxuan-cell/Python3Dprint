from pathlib import Path
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vedo import Plotter, Text2D, load


# =========================
# 1. 你的原有设置
# =========================

MODEL_PATH = Path(r"E:\cat\ORIGAMI_Chat_Free.obj")

RESOLUTION = 48
INTERVAL_MS = 4
REPLAY_DELAY = 3.0
CUBE_RATIO = 1.0

# 新增：镜头动画设置
FRAME_MS = 16             # 请求约 60 帧/秒，实际帧率取决于电脑
ROTATION_SPEED = 8.0      # 每秒旋转角度，约 45 秒一圈


# =========================
# 2. 生成并排序体素
# =========================

def voxelize(model):
    bounds = np.array(model.bounds()).reshape(3, 2)
    lengths = bounds[:, 1] - bounds[:, 0]

    spacing = float(lengths.max() / RESOLUTION)

    if spacing <= 0:
        raise ValueError("模型尺寸无效")

    dimensions = np.maximum(
        1, np.ceil(lengths / spacing).astype(int)
    )

    origin = bounds[:, 0] + spacing / 2

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(model.dataset)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputConnection(clean.GetOutputPort())
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(spacing, spacing, spacing)
    stencil.SetOutputWholeExtent(
        0, int(dimensions[0]) - 1,
        0, int(dimensions[1]) - 1,
        0, int(dimensions[2]) - 1,
    )

    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar()
    image.Update()

    values = vtk_to_numpy(
        image.GetOutput().GetPointData().GetScalars()
    )

    mask = values.reshape(
        int(dimensions[2]),
        int(dimensions[1]),
        int(dimensions[0]),
    )

    iz, iy, ix = np.nonzero(mask)

    if len(ix) == 0:
        raise RuntimeError("没有生成体素，请检查模型是否封闭")

    centers = origin + np.column_stack((ix, iy, iz)) * spacing

    # 从下到上；同一层内按蛇形顺序逐个放置
    snake_x = np.where(iy % 2 == 0, ix, -ix)
    order = np.lexsort((snake_x, iy, iz))

    return (
        centers[order],
        iz[order],
        spacing,
        int(dimensions[2]),
    )


# =========================
# 3. 批量渲染小方块
# =========================

def create_cube_renderer(centers, z_indices, spacing, height_layers):
    # 每个方块的颜色由高度决定
    ratio = z_indices / max(1, height_layers - 1)

    colors = np.column_stack((
        0.15 + 0.80 * ratio,
        np.full(len(centers), 0.75),
        0.95 - 0.75 * ratio,
    ))

    colors = np.ascontiguousarray(
        np.clip(colors * 255, 0, 255).astype(np.uint8)
    )

    centers = np.ascontiguousarray(centers, dtype=np.float32)

    cube = vtk.vtkCubeSource()
    side = spacing * CUBE_RATIO
    cube.SetXLength(side)
    cube.SetYLength(side)
    cube.SetZLength(side)

    points = vtk.vtkPoints()

    data = vtk.vtkPolyData()
    data.SetPoints(points)

    # 一个渲染对象显示所有方块，避免创建大量独立演员
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(data)
    mapper.SetSourceConnection(cube.GetOutputPort())
    mapper.ScalingOff()
    mapper.OrientOff()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("CubeColors")
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(0.35)
    actor.GetProperty().SetDiffuse(0.65)
    actor.GetProperty().SetSpecular(0.15)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetInterpolationToFlat()
    actor.SetVisibility(False)

    def reveal(count):
        if count == 0:
            actor.SetVisibility(False)
            return

        points.SetData(
            numpy_to_vtk(centers[:count], deep=True)
        )

        color_array = numpy_to_vtk(
            colors[:count],
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        color_array.SetName("CubeColors")

        data.GetPointData().SetScalars(color_array)
        points.Modified()
        data.Modified()
        actor.SetVisibility(True)

    return actor, reveal


# =========================
# 4. 主程序
# =========================

def main():
    if RESOLUTION < 4:
        raise ValueError("RESOLUTION 至少为 4")

    if INTERVAL_MS <= 0 or FRAME_MS <= 0:
        raise ValueError("时间间隔必须大于零")

    if REPLAY_DELAY < 0:
        raise ValueError("重播等待时间不能为负数")

    if not 0 < CUBE_RATIO <= 1:
        raise ValueError("CUBE_RATIO 应在 0 到 1 之间")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")

    cat = load(str(MODEL_PATH))

    if cat is None or not cat.ncells:
        raise RuntimeError("模型加载失败")

    # 当前猫模型是 Y 轴朝上，转为 Z 轴朝上
    cat.rotate_x(90)
    cat.color("lightgray").lighting("plastic")

    centers, z_indices, spacing, height_layers = voxelize(cat)
    total_cubes = len(centers)

    cube_actor, reveal = create_cube_renderer(
        centers, z_indices, spacing, height_layers
    )

    print(f"方块总数：{total_cubes}")
    print(f"高度层数：{height_layers}")
    print(f"方块边长：{spacing * CUBE_RATIO:.4f} 模型单位")
    print(f"理论搭建时间：{total_cubes * INTERVAL_MS / 1000:.2f} 秒")
    print("空格：暂停 / 继续（同时暂停镜头）")
    print("R：重新开始搭建")
    print("关闭模型窗口退出。")


    # =========================
    # 5. 创建左右视图
    # =========================

    plot = Plotter(
        shape=(1, 2),
        sharecam=True,
        size=(1400, 800),
        bg="#18202b",
        title="Cat - Rotating Voxel Animation",
    )

    bounds = np.array(cat.bounds()).reshape(3, 2)
    center = bounds.mean(axis=1)
    distance = float(np.max(bounds[:, 1] - bounds[:, 0])) * 2

    camera_settings = dict(
        pos=center + np.array([
            distance,
            distance * 0.8,
            distance * 0.6,
        ]),
        focal_point=center,
        viewup=(0, 0, 1),
    )

    plot.at(0).show(
        cat,
        Text2D("Original model", c="white"),
        axes=1,
        camera=camera_settings,
        interactive=False,
    )

    status = Text2D("", c="white")

    # 不可见模型用于确定右侧显示范围
    bounds_model = cat.clone().alpha(0)

    plot.at(1).show(
        bounds_model,
        status,
        axes=1,
        interactive=False,
        resetcam=False,
    )

    plot.renderers[1].AddActor(cube_actor)

    camera = plot.renderers[1].GetActiveCamera()

    # 两个视图共享镜头，旋转时可以同步比较
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0, 0, 1)

    state = {
        "count": 0,
        "paused": False,
        "build_elapsed": 0.0,
        "hold_elapsed": 0.0,
        "last_time": time.monotonic(),
    }

    def update_text():
        count = state["count"]

        if state["paused"]:
            mode = "PAUSED"
        elif count == total_cubes:
            mode = "COMPLETE - rotating showcase"
        else:
            mode = "BUILDING"

        current_layer = (
            int(z_indices[count - 1]) + 1
            if count else 0
        )

        status.text(
            f"Cubes: {count} / {total_cubes}\n"
            f"Height layer: {current_layer} / {height_layers}\n"
            f"Progress: {count / total_cubes * 100:.1f}%\n"
            f"{mode}\n"
            "SPACE: pause | R: restart"
        )

    def restart():
        state["count"] = 0
        state["build_elapsed"] = 0.0
        state["hold_elapsed"] = 0.0
        state["last_time"] = time.monotonic()
        reveal(0)


    # =========================
    # 6. 同时进行旋转和搭建
    # =========================

    def on_timer(caller, event):
        now = time.monotonic()
        elapsed = now - state["last_time"]
        state["last_time"] = now

        if state["paused"]:
            return

        # 避免拖动窗口、短暂卡顿后镜头突然跳转
        elapsed = min(elapsed, 0.1)

        # 沿固定竖直轴环绕模型中心
        camera.Azimuth(ROTATION_SPEED * elapsed)
        camera.OrthogonalizeViewUp()

        if state["count"] < total_cubes:
            state["build_elapsed"] += elapsed

            target_count = min(
                total_cubes,
                int(state["build_elapsed"] * 1000 / INTERVAL_MS),
            )

            if target_count > state["count"]:
                # 按已排序的方块序列累积显示
                state["count"] = target_count
                reveal(target_count)

                if target_count == total_cubes:
                    state["hold_elapsed"] = 0.0

        else:
            # 完成后仍然旋转，展示完整方块猫
            state["hold_elapsed"] += elapsed

            if state["hold_elapsed"] >= REPLAY_DELAY:
                restart()

        for renderer in plot.renderers:
            renderer.ResetCameraClippingRange()

        update_text()
        plot.render()

    def on_key(caller, event):
        key = caller.GetKeySym().lower()

        if key == "space":
            state["paused"] = not state["paused"]
            state["last_time"] = time.monotonic()

        elif key == "r":
            restart()

        else:
            return

        update_text()
        plot.render()

    update_text()
    plot.render()

    plot.interactor.AddObserver("TimerEvent", on_timer)
    plot.interactor.AddObserver("KeyPressEvent", on_key)

    timer_id = plot.timer_callback(
        "create",
        dt=FRAME_MS,
    )

    # 初始化完成后再开始计时
    state["last_time"] = time.monotonic()

    try:
        plot.interactive()
    finally:
        plot.timer_callback("destroy", timer_id=timer_id)
        plot.close()


if __name__ == "__main__":
    main()