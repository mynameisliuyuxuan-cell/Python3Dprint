import vtk


# 自定义鼠标操作
class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()

        self.AddObserver("LeftButtonPressEvent", self.left_press)
        self.AddObserver("LeftButtonReleaseEvent", self.left_release)

        self.AddObserver("RightButtonPressEvent", self.right_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_release)

    # 左键：平移
    def left_press(self, obj, event):
        self.StartPan()

    def left_release(self, obj, event):
        self.EndPan()

    # 右键：旋转
    def right_press(self, obj, event):
        self.StartRotate()

    def right_release(self, obj, event):
        self.EndRotate()


# =========================
# 读取你的 STL 模型
# =========================
reader = vtk.vtkSTLReader()
reader.SetFileName("program.stl")
reader.Update()

print("模型点数：", reader.GetOutput().GetNumberOfPoints())
print("模型面数：", reader.GetOutput().GetNumberOfCells())
print("模型范围：", reader.GetOutput().GetBounds())


# 映射器
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())


# 模型 Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 模型颜色
actor.GetProperty().SetColor(0.8, 0.8, 0.8)


# 渲染器
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.2, 0.2, 0.2)


# 渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1000, 800)
render_window.SetWindowName("STL Model")


# 交互器
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 使用自定义操作
style = MyInteractorStyle()
interactor.SetInteractorStyle(style)


# 左下角坐标轴
axes = vtk.vtkAxesActor()

orientation_marker = vtk.vtkOrientationMarkerWidget()
orientation_marker.SetOrientationMarker(axes)
orientation_marker.SetInteractor(interactor)
orientation_marker.SetViewport(0.0, 0.0, 0.2, 0.2)
orientation_marker.EnabledOn()
orientation_marker.InteractiveOff()


# 自动让相机对准模型
renderer.ResetCamera()


# 开始显示
render_window.Render()
interactor.Initialize()
interactor.Start()