from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def read_image(path):
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    if image is None:
        raise ValueError("无法解码该图片。")
    return image


def gray_image(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preview_rgb(image):
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=1)
def depth_model():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    cache = Path("E:/huggingface/hub")
    options = {"local_files_only": True}
    if cache.is_dir():
        options["cache_dir"] = str(cache)
    name = "Intel/dpt-large"
    processor = AutoImageProcessor.from_pretrained(name, **options)
    model = AutoModelForDepthEstimation.from_pretrained(name, **options).eval()
    return processor, model


def estimate_depth(image):
    import torch

    processor, model = depth_model()
    inputs = processor(images=preview_rgb(image), return_tensors="pt")
    with torch.inference_mode():
        return model(**inputs).predicted_depth[0].cpu().numpy()


def make_relief(image, photo, width, rise, base, invert):
    depth = estimate_depth(image) if photo else gray_image(image)
    original_rows, original_cols = image.shape[:2]
    aspect = original_rows / original_cols
    cols = max(2, round(min(200, 200 / aspect)))
    rows = max(2, round(min(200, 200 * aspect)))
    sampled = cv2.resize(depth.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    if not np.isfinite(sampled).all():
        raise ValueError("深度图包含无效数值。")

    minimum, maximum = sampled.min(), sampled.max()
    if maximum > minimum:
        sampled = (sampled - minimum) / (maximum - minimum)
        if invert:
            sampled = 1 - sampled
    else:
        sampled.fill(0)

    x, y = np.meshgrid(np.linspace(0, width, cols), np.linspace(0, width * aspect, rows))
    top = np.column_stack((x.ravel(), y.ravel(), base + rise * sampled[::-1].ravel()))
    bottom = top.copy()
    bottom[:, 2] = 0
    vertices = np.concatenate((top, bottom))

    grid = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    a, b, c, d = (part.ravel() for part in
                  (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]))
    top_faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    offset = len(top)
    bottom_faces = top_faces[:, ::-1] + offset
    rim = np.concatenate((grid[0, :-1], grid[:-1, -1], grid[-1, :0:-1], grid[:0:-1, 0]))
    following = np.roll(rim, -1)
    side_faces = np.column_stack((rim + offset, following + offset, following))
    side_faces_2 = np.column_stack((rim + offset, following, rim))
    triangles = np.concatenate((top_faces, bottom_faces, side_faces, side_faces_2))

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(3, numpy_to_vtkIdTypeArray(triangles.ravel(), deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh
