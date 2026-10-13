from pathlib import Path

import cv2
import numpy as np
import trimesh
import vtk


# ==========================================
# 自定义 VTK 鼠标操作
# 左键拖动：平移
# 右键拖动：旋转
# 滚轮：缩放
# ==========================================

class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self):
        super().__init__()

        self.AddObserver(
            "LeftButtonPressEvent",
            self.left_button_press
        )

        self.AddObserver(
            "LeftButtonReleaseEvent",
            self.left_button_release
        )

        self.AddObserver(
            "RightButtonPressEvent",
            self.right_button_press
        )

        self.AddObserver(
            "RightButtonReleaseEvent",
            self.right_button_release
        )

    def left_button_press(self, obj, event):
        self.StartPan()

    def left_button_release(self, obj, event):
        self.EndPan()

    def right_button_press(self, obj, event):
        self.StartRotate()

    def right_button_release(self, obj, event):
        self.EndRotate()


# ==========================================
# 1. 路径设置
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

IMAGE_PATH = DATA_DIR / "0929test2.png"
OUTPUT_PATH = DATA_DIR / "output_solid.stl"


# ==========================================
# 2. 参数设置
# ==========================================

MODEL_WIDTH = 100.0       # 模型宽度，mm
RELIEF_HEIGHT = 8.0       # 浮雕最大高度，mm
BASE_THICKNESS = 2.0      # 底板厚度，mm

TARGET_WIDTH = 200        # 图片采样宽度

# False：白色高，黑色低
# True ：黑色高，白色低
INVERT = False


# ==========================================
# 3. 读取图片
# ==========================================

img = cv2.imread(
    str(IMAGE_PATH),
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    raise FileNotFoundError(
        f"找不到图片：{IMAGE_PATH}"
    )

print("图片读取成功")
print("图片路径：", IMAGE_PATH)
print("原始尺寸：", img.shape)


# ==========================================
# 4. 按比例缩放
# ==========================================

original_height, original_width = img.shape

aspect_ratio = original_height / original_width

target_height = int(
    TARGET_WIDTH * aspect_ratio
)

img = cv2.resize(
    img,
    (TARGET_WIDTH, target_height),
    interpolation=cv2.INTER_AREA
)

print("处理后尺寸：", img.shape)


# ==========================================
# 5. 图片灰度转高度
# ==========================================

height_map = img.astype(
    np.float32
) / 255.0

if INVERT:
    height_map = 1.0 - height_map

height_map *= RELIEF_HEIGHT


# ==========================================
# 6. 计算模型尺寸
# ==========================================

rows, cols = height_map.shape

MODEL_HEIGHT = (
    MODEL_WIDTH * rows / cols
)

dx = MODEL_WIDTH / (cols - 1)
dy = MODEL_HEIGHT / (rows - 1)

print("模型宽度：", MODEL_WIDTH, "mm")
print("模型高度：", MODEL_HEIGHT, "mm")
print("底板厚度：", BASE_THICKNESS, "mm")
print("浮雕最大高度：", RELIEF_HEIGHT, "mm")


# ==========================================
# 7. 顶面顶点
# ==========================================

top_vertices = []

for y in range(rows):

    for x in range(cols):

        px = x * dx
        py = y * dy

        # 底板顶部 + 浮雕高度
        pz = BASE_THICKNESS + height_map[y, x]

        top_vertices.append([
            px,
            py,
            pz
        ])

top_vertices = np.array(
    top_vertices,
    dtype=np.float32
)


# ==========================================
# 8. 底面顶点
# ==========================================

bottom_vertices = []

for y in range(rows):

    for x in range(cols):

        px = x * dx
        py = y * dy
        pz = 0.0

        bottom_vertices.append([
            px,
            py,
            pz
        ])

bottom_vertices = np.array(
    bottom_vertices,
    dtype=np.float32
)


# ==========================================
# 9. 合并全部顶点
# ==========================================

vertices = np.vstack([
    top_vertices,
    bottom_vertices
])

top_offset = 0
bottom_offset = rows * cols


# ==========================================
# 10. 生成顶面
# ==========================================

faces = []

for y in range(rows - 1):

    for x in range(cols - 1):

        p0 = top_offset + y * cols + x
        p1 = top_offset + y * cols + x + 1

        p2 = top_offset + (y + 1) * cols + x
        p3 = top_offset + (y + 1) * cols + x + 1

        faces.append([
            p0,
            p1,
            p2
        ])

        faces.append([
            p1,
            p3,
            p2
        ])


# ==========================================
# 11. 生成底面
# ==========================================

for y in range(rows - 1):

    for x in range(cols - 1):

        p0 = bottom_offset + y * cols + x
        p1 = bottom_offset + y * cols + x + 1

        p2 = bottom_offset + (y + 1) * cols + x
        p3 = bottom_offset + (y + 1) * cols + x + 1

        # 底面法线方向反过来
        faces.append([
            p0,
            p2,
            p1
        ])

        faces.append([
            p1,
            p2,
            p3
        ])


# ==========================================
# 12. 左侧封边
# ==========================================

for y in range(rows - 1):

    top_a = top_offset + y * cols
    top_b = top_offset + (y + 1) * cols

    bottom_a = bottom_offset + y * cols
    bottom_b = bottom_offset + (y + 1) * cols

    faces.append([
        top_a,
        bottom_a,
        top_b
    ])

    faces.append([
        top_b,
        bottom_a,
        bottom_b
    ])


# ==========================================
# 13. 右侧封边
# ==========================================

for y in range(rows - 1):

    top_a = (
        top_offset +
        y * cols +
        cols - 1
    )

    top_b = (
        top_offset +
        (y + 1) * cols +
        cols - 1
    )

    bottom_a = (
        bottom_offset +
        y * cols +
        cols - 1
    )

    bottom_b = (
        bottom_offset +
        (y + 1) * cols +
        cols - 1
    )

    faces.append([
        top_a,
        top_b,
        bottom_a
    ])

    faces.append([
        top_b,
        bottom_b,
        bottom_a
    ])


# ==========================================
# 14. 上侧封边
# ==========================================

for x in range(cols - 1):

    top_a = top_offset + x
    top_b = top_offset + x + 1

    bottom_a = bottom_offset + x
    bottom_b = bottom_offset + x + 1

    faces.append([
        top_a,
        top_b,
        bottom_a
    ])

    faces.append([
        top_b,
        bottom_b,
        bottom_a
    ])


# ==========================================
# 15. 下侧封边
# ==========================================

last_row = rows - 1

for x in range(cols - 1):

    top_a = (
        top_offset +
        last_row * cols +
        x
    )

    top_b = (
        top_offset +
        last_row * cols +
        x + 1
    )

    bottom_a = (
        bottom_offset +
        last_row * cols +
        x
    )

    bottom_b = (
        bottom_offset +
        last_row * cols +
        x + 1
    )

    faces.append([
        top_a,
        bottom_a,
        top_b
    ])

    faces.append([
        top_b,
        bottom_a,
        bottom_b
    ])


faces = np.array(
    faces,
    dtype=np.int64
)


# ==========================================
# 16. 创建封闭网格
# ==========================================

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=True
)


# ==========================================
# 17. 检查是否封闭
# ==========================================

print()
print("是否封闭：", mesh.is_watertight)
print("是否体积模型：", mesh.is_volume)

if mesh.is_watertight:
    print("模型已封闭，可作为实体 STL 使用")
else:
    print("警告：模型仍不是完全封闭网格")


# ==========================================
# 18. 导出 STL
# ==========================================

mesh.export(
    str(OUTPUT_PATH)
)

print()
print("转换完成！")
print("输出文件：")
print(OUTPUT_PATH)

print()
print("顶点数量：", len(mesh.vertices))
print("三角面数量：", len(mesh.faces))


# ==========================================
# 19. VTK 读取 STL
# ==========================================

reader = vtk.vtkSTLReader()
reader.SetFileName(
    str(OUTPUT_PATH)
)
reader.Update()


# ==========================================
# 20. 映射器
# ==========================================

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(
    reader.GetOutputPort()
)


# ==========================================
# 21. Actor
# ==========================================

actor = vtk.vtkActor()
actor.SetMapper(mapper)

actor.GetProperty().SetColor(
    0.82,
    0.82,
    0.82
)


# ==========================================
# 22. Renderer
# ==========================================

renderer = vtk.vtkRenderer()

renderer.AddActor(actor)

renderer.SetBackground(
    0.15,
    0.15,
    0.15
)


# ==========================================
# 23. Render Window
# ==========================================

render_window = vtk.vtkRenderWindow()

render_window.AddRenderer(
    renderer
)

render_window.SetSize(
    1000,
    800
)

render_window.SetWindowName(
    "Printable Relief Preview"
)


# ==========================================
# 24. Interactor
# ==========================================

interactor = vtk.vtkRenderWindowInteractor()

interactor.SetRenderWindow(
    render_window
)


# ==========================================
# 25. 自定义鼠标
# ==========================================

style = MyInteractorStyle()

interactor.SetInteractorStyle(
    style
)


# ==========================================
# 26. 坐标轴
# ==========================================

axes = vtk.vtkAxesActor()

orientation_marker = vtk.vtkOrientationMarkerWidget()

orientation_marker.SetOrientationMarker(
    axes
)

orientation_marker.SetInteractor(
    interactor
)

orientation_marker.SetViewport(
    0.0,
    0.0,
    0.2,
    0.2
)

orientation_marker.EnabledOn()

orientation_marker.InteractiveOff()


# ==========================================
# 27. 自动相机
# ==========================================

renderer.ResetCamera()


# ==========================================
# 28. 显示
# ==========================================

render_window.Render()

interactor.Initialize()

interactor.Start()