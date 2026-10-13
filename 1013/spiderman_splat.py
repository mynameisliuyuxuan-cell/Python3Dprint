"""OBJ textured surface samples rendered with Gaussian splats (not trained 3DGS)."""
from pathlib import Path
import argparse
import time
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

MODEL_PATH = Path(r'E:\Spider-Man+Brand+New+Day.obj')
TEXTURE_DIR = Path(r'E:\Textures\Textures')
POINT_COUNT = 180000
BUILD_SECONDS = 16.0
HOLD_SECONDS = 4.0
ROTATION_SPEED = 9.0
SPLAT_SIZE = 0.0012  # Relative to model height; larger means softer/blobbier.
FRAME_MS = 33
SOLID_RESOLUTION = 120


def sample_model():
    vertices, uvs, faces, uvfaces, groups = [], [], [], [], []
    group = 0
    with MODEL_PATH.open(encoding='utf-8', errors='replace') as source:
        for line in source:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vt':
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == 'usemtl':
                group = 1 if parts[1] == 'Acc' else 0
            elif parts[0] == 'f':
                indices, texture_indices = [], []
                for token in parts[1:]:
                    fields = token.split('/')
                    vi = int(fields[0])
                    ti = int(fields[1]) if len(fields) > 1 and fields[1] else 0
                    if ti == 0:
                        raise ValueError('OBJ face lacks texture coordinates')
                    indices.append(vi - 1 if vi > 0 else len(vertices) + vi)
                    texture_indices.append(ti - 1 if ti > 0 else len(uvs) + ti)
                for j in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[j], indices[j + 1]])
                    uvfaces.append([texture_indices[0], texture_indices[j], texture_indices[j + 1]])
                    groups.append(group)
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(triangles), POINT_COUNT, p=areas / areas.sum())
    a = np.sqrt(rng.random(POINT_COUNT))
    b = rng.random(POINT_COUNT)
    weights = np.column_stack((1 - a, a * (1 - b), a * b))
    xyz = np.einsum('ij,ijk->ik', weights, triangles[chosen])
    uv = np.einsum('ij,ijk->ik', weights, np.asarray(uvs)[np.asarray(uvfaces)[chosen]])
    material = np.asarray(groups)[chosen]
    # UDIM: each integer UV square refers to a different numbered image.
    tile_x, tile_y = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    tiles = 1001 + tile_x + 10 * tile_y
    rgba = np.zeros((POINT_COUNT, 4), dtype=np.uint8)
    for mat in (0, 1):
        prefix = 'Acc' if mat else 'Suit'
        for tile in np.unique(tiles[material == mat]):
            ids = np.flatnonzero((material == mat) & (tiles == tile))
            path = TEXTURE_DIR / f'{prefix}_BaseColor.{tile}.png'
            if not path.exists():
                raise FileNotFoundError(f'Missing UDIM texture: {path}')
            with Image.open(path) as source:
                texture = np.asarray(source.convert('RGBA'))
            h, w = texture.shape[:2]
            x = np.clip(((uv[ids, 0] - tile_x[ids]) * w).astype(int), 0, w - 1)
            y = np.clip(((1 - uv[ids, 1] + tile_y[ids]) * h).astype(int), 0, h - 1)
            rgba[ids] = texture[y, x]
            alpha_path = TEXTURE_DIR / f'{prefix}_Alpha.{tile}.png'
            if alpha_path.exists():
                with Image.open(alpha_path) as source:
                    alpha = np.asarray(source.convert('L'))
                ah, aw = alpha.shape
                ax = np.minimum((x / w * aw).astype(int), aw - 1)
                ay = np.minimum((y / h * ah).astype(int), ah - 1)
                rgba[ids, 3] = (rgba[ids, 3].astype(float) * alpha[ay, ax] / 255).astype(np.uint8)
    visible = rgba[:, 3] > 20
    xyz, rgba = xyz[visible], rgba[visible]
    # Y-up OBJ -> Z-up display.
    xyz = np.column_stack((xyz[:, 0], -xyz[:, 2], xyz[:, 1]))
    # Slightly irregular reveal front gives a softer particle-building effect.
    order = np.argsort(xyz[:, 2])
    print(f'Loaded {len(triangles)} triangles; {len(xyz)} textured Gaussian points', flush=True)
    body = triangles[np.asarray(groups) == 0].copy()
    body = body[:, :, [0, 2, 1]]
    body[:, :, 1] *= -1
    return np.ascontiguousarray(xyz[order], dtype=np.float32), np.ascontiguousarray(rgba[order]), body


def solid_model(triangles):
    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(triangles.reshape(-1, 3), deep=True))
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    indices = np.column_stack((np.full(len(triangles), 3), np.arange(len(triangles) * 3).reshape(-1, 3))).astype(np.int64)
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(indices.ravel(), deep=True))
    poly.SetPolys(cells)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(1e-6)
    clean.Update()
    bounds = np.array(clean.GetOutput().GetBounds()).reshape(3, 2)
    height = np.max(np.ptp(bounds, axis=1))
    holes = vtk.vtkFillHolesFilter()
    holes.SetInputConnection(clean.GetOutputPort())
    holes.SetHoleSize(float(height * 0.08))
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(holes.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(normals.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    edge_count = edges.GetOutput().GetNumberOfCells()
    print(f'Interior suit mesh: {edge_count} boundary/nonmanifold edges after seam merge and small-hole closure', flush=True)
    if edge_count:
        raise RuntimeError('Suit mesh is not closed; refusing unreliable interior fill')
    step = float(height / SOLID_RESOLUTION)
    dims = np.maximum(1, np.ceil((bounds[:, 1] - bounds[:, 0]) / step).astype(int))
    origin = bounds[:, 0] + step / 2
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputConnection(normals.GetOutputPort())
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(step, step, step)
    stencil.SetOutputWholeExtent(0, int(dims[0])-1, 0, int(dims[1])-1, 0, int(dims[2])-1)
    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar()
    image.Update()
    mask = vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).reshape(tuple(dims[::-1]))
    iz, iy, ix = np.nonzero(mask)
    centers = np.ascontiguousarray(origin + np.column_stack((ix, iy, iz)) * step, dtype=np.float32)
    print(f'Solid interior: {len(centers)} opaque voxels', flush=True)
    if not len(centers):
        raise RuntimeError('Interior is empty')
    cube = vtk.vtkCubeSource()
    cube.SetXLength(step)
    cube.SetYLength(step)
    cube.SetZLength(step)
    data = vtk.vtkPolyData()
    voxpoints = vtk.vtkPoints()
    data.SetPoints(voxpoints)
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(data)
    mapper.SetSourceConnection(cube.GetOutputPort())
    mapper.ScalingOff()
    mapper.OrientOff()
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.22, 0.045, 0.065)
    actor.GetProperty().SetAmbient(0.5)
    actor.GetProperty().SetDiffuse(0.5)
    actor.SetVisibility(False)
    # Clip the current top voxel row at the exact reveal height.
    plane = vtk.vtkPlane()
    plane.SetNormal(0, 0, -1)
    mapper.AddClippingPlane(plane)

    def reveal(z):
        n = int(np.searchsorted(centers[:, 2], z + step / 2, side='right'))
        actor.SetVisibility(n > 0)
        if n:
            voxpoints.SetData(numpy_to_vtk(centers[:n], deep=True))
            data.Modified()
        plane.SetOrigin(0, 0, z)
        return n
    return actor, reveal


def main(check=False, screenshot=None):
    xyz, rgba, body = sample_model()
    solid_actor, reveal_solid = solid_model(body)
    assert len(xyz) > 0 and np.isfinite(xyz).all()
    points = vtk.vtkPoints()
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(points)
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetInputData(cloud)
    mapper.SetScaleFactor(float(np.ptp(xyz[:, 2])) * SPLAT_SIZE)
    mapper.EmissiveOff()
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.88)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.012, 0.018, 0.035)
    renderer.AddActor(actor)
    renderer.AddActor(solid_actor)
    text = vtk.vtkTextActor()
    text.SetPosition(22, 22)
    text.GetTextProperty().SetFontSize(19)
    text.GetTextProperty().SetColor(0.75, 0.85, 1)
    renderer.AddViewProp(text)
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1100, 850)
    window.SetWindowName('Spider-Man - Textured Gaussian Point Animation')
    window.SetMultiSamples(0)
    if check:
        window.SetOffScreenRendering(1)
    camera = renderer.GetActiveCamera()
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    height = float(np.ptp(xyz[:, 2]))
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + np.array([height * 0.25, -height * 2.1, height * 0.15])))
    camera.SetViewUp(0, 0, 1)
    camera.SetClippingRange(height * 0.01, height * 10)
    state = dict(elapsed=0.0, last=time.monotonic(), paused=False, count=-1)

    def reveal(count):
        if count == state['count']:
            return
        state['count'] = count
        actor.SetVisibility(count > 0)
        cutoff = float(xyz[count - 1, 2]) if count else float(xyz[0, 2] - 1)
        filled = reveal_solid(cutoff)
        if count:
            points.SetData(numpy_to_vtk(xyz[:count], deep=True))
            colors = numpy_to_vtk(rgba[:count], deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            cloud.GetPointData().SetScalars(colors)
            cloud.Modified()
        text.SetInput(f'Textured Gaussian points: {count:,} / {len(xyz):,} | Solid cells: {filled:,}\n'
                      f'{"PAUSED" if state["paused"] else "SPACE: pause/resume | R: restart"}')

    def tick(caller, event):
        now = time.monotonic()
        dt = min(now - state['last'], 0.1)
        state['last'] = now
        if state['paused']:
            return
        state['elapsed'] = (state['elapsed'] + dt) % (BUILD_SECONDS + HOLD_SECONDS)
        camera.Azimuth(ROTATION_SPEED * dt)
        reveal(min(len(xyz), int(len(xyz) * state['elapsed'] / BUILD_SECONDS)))
        window.Render()

    def key(caller, event):
        if caller.GetKeySym().lower() == 'space':
            state['paused'] = not state['paused']
        elif caller.GetKeySym().lower() == 'r':
            state['elapsed'] = 0
        else:
            return
        state['last'] = time.monotonic()
        old = state['count']
        state['count'] = -1
        reveal(max(0, old))
        window.Render()

    if check:
        def save_frame(path):
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(window)
            capture.ReadFrontBufferOff()
            capture.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(path))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
        for count in (1, len(xyz) // 2, len(xyz)):
            reveal(count)
            window.Render()
            if screenshot and count == len(xyz) // 2:
                camera.Elevation(22)
                window.Render()
                save_frame(Path(screenshot).with_name(Path(screenshot).stem + '_half.png'))
                camera.Elevation(-22)
        if screenshot:
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(window)
            capture.ReadFrontBufferOff()
            capture.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(screenshot))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
        print('PASS: UDIM colors, progressive point rendering, full-frame render')
        window.Finalize()
        return
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.AddObserver('TimerEvent', tick)
    interactor.AddObserver('KeyPressEvent', key)
    reveal(0)
    interactor.Initialize()
    window.Render()
    interactor.CreateRepeatingTimer(FRAME_MS)
    state['last'] = time.monotonic()
    interactor.Start()
    window.Finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--screenshot')
    args = parser.parse_args()
    main(args.check, args.screenshot)
