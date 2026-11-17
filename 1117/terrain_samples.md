# 地形样本与进阶计划

高程数据来自 USGS 3DEP。每张灰度图为 512 × 512 像素，单通道 16 位。每幅图的黑色对应该图最低高程，白色对应最高高程。可用 `最低高程 + 灰度值 / 65535 × (最高高程 − 最低高程)` 近似还原原始栅格高程。

| 文件 | 地形 | 灰度 0 对应海拔（m） | 灰度 65535 对应海拔（m） |
| --- | --- | ---: | ---: |
| [grand_canyon_depth.png](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox=-12493086,4301051,-12465857,4328280&bboxSR=3857&imageSR=3857&size=512,512&format=tiff&pixelType=F32&compression=LZ77&f=image) | 科罗拉多大峡谷 | 710.421936 | 2531.513184 |
| [crater_lake_depth.png](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox=-13602660,5295180,-13584900,5312940&bboxSR=3857&imageSR=3857&size=512,512&format=tiff&pixelType=F32&compression=LZ77&f=image) | 火山口湖及周边 | 1679.459595 | 2482.104980 |
| [death_valley_depth.png](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox=-13021448,4318817,-12991691,4348574&bboxSR=3857&imageSR=3857&size=513,513&format=tiff&pixelType=F32&compression=LZ77&f=image) | 死亡谷盆地及山缘 | −85.062836 | 1740.246826 |
| [mississippi_meanders_depth.png](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox=-10156308,3933746,-10122763,3967291&bboxSR=3857&imageSR=3857&size=512,512&format=tiff&pixelType=F32&compression=LZ77&f=image) | 密西西比河曲流及低平洪泛平原 | 29.590000 | 68.352280 |

表中链接指向原始 GeoTIFF。死亡谷的源栅格为 513 × 513，经面积重采样得到 512 × 512 后再做灰度映射，表中列出映射时使用的高程范围。

1. **海拔读取与垂直比例**。支持填写高程范围与实际地面宽度，保留现有指定起伏高度的模式，再增加等比缩放与垂直夸张倍数。用死亡谷检查负海拔显示，用大峡谷和密西西比河样本对比等比缩放与固定 8 mm 起伏。
2. **剖面与等高线**。在灰度图上选两个点，显示沿线距离与海拔，把同一条线标到三维地形上。用一条横跨大峡谷的剖面检查两侧谷缘与河谷的位置，再加可调等高距的等高线。
3. **裁剪与打印验证**。选定区域后重新构网，保留指定底板厚度。用大峡谷裁一块区域，验证 STL 尺寸和封闭性，再用已有 PrusaSlicer 预览检查底板与逐层路径。
