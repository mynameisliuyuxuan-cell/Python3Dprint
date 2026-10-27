# 姿态与 PrusaSlicer 支撑对照

入口：`project4_pose_prusa.py`，直接用项目现有的系统 Python 3.14 运行。

同目录需保留：`support_compare_core.py`、`prusa_support_bridge.py` 和猫模型 OBJ。
PrusaSlicer 路径：`C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe`。

## 对照内容

- 三列：直立、绕 X 倾斜、绕 Y 倾斜，默认倾角 20°。
- 上排：自制树状支撑及拆分回退，分别在当前姿态采样接触点。
- 下排：实际调用本机 PrusaSlicer，解析生成 G-code 中的支撑与支撑接触层挤出路径。
- 原猫先转成 Z-up、高度归一为 80；之后各姿态只旋转和平移到 Z=0，不再重新缩放。
- 所有视窗共用相机，便于保持同一观察方向和显示尺度。

## 操作

- 顶部滑块：倾角、悬垂阈值、接触间隙；修改后按 G 重算全部。
- G：重新生成自制支撑，再后台调用 Prusa。
- P：重新计算或加载当前姿态的 Prusa 缓存。
- K：在 Prusa 普通网格 / 有机树状支撑之间切换。
- M：模型透明度；空格：自动旋转；R：重置观察视角。
- X：将当前已完成结果导出至 `comparison_results/` 的 CSV。
- 关闭窗口会取消排队任务，并终止由本次计算启动、尚未完成的 Prusa 子进程。

## 指标口径

- 上排显示接通/未解决、根部数和自制管段/底座体积累加；累加值包含交叠。
- 下排显示支撑挤出路径长度、段数，以及由支撑 E 增量乘以 1.75 毫米丝材截面积估算的材料体积。
- 自制结构体积和 Prusa 挤出材料体积不属于同一口径，不应直接做节省百分比。
- 两侧对齐输入角度和接触间隙；接触点、支撑密度、桥接与层高规则仍各自决定。
- Prusa 本地演示配置使用 220×220 平台、0.4 毫米喷嘴、0.3 毫米层高，仅平台支撑。
- 不代表实际打印机配置；生成 G-code 仅用于研究，不能发送到真实打印机。

## 缓存与验证

- `prusa_compare_cache/` 保存每种几何姿态与配置对应的 STL、INI、G-code、日志和统计；已加入本仓库本地排除规则，不随 Git 同步。
- 计算完全在本机进行，不上传模型。
- 默认 20° 倾斜/45° 阈值/0.6 毫米间隙，实际检查过三种姿态和有机支撑渲染。
- 已检查姿态变换不缩放、模型落地、G-code 支撑角色过滤、相对/绝对挤出、回抽/空驶过滤及普通网格支撑调用。

离屏预览：

```powershell
& "C:\Users\刘雨轩\AppData\Local\Programs\Python\Python314\python.exe" "C:\Users\刘雨轩\PycharmProjects\Python3Dprint\1027\project4_pose_prusa.py" --preview "E:\pose_prusa_preview.png"
```

官方参考：
- https://github.com/prusa3d/PrusaSlicer/wiki/Command-Line-Interface
- https://help.prusa3d.com/article/support-material_1698
