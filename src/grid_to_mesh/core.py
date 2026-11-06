from pathlib import Path
from grid_to_mesh.grid import Grid
from grid_to_mesh.mesh import Mesh
from grid_to_mesh.projector_parallel import OpenSurfaceProjector, ClosedSurfaceProjector
from grid_to_mesh.import_config import Config
import time
import numpy as np


def project_grid(config: str | Path, mesh_file: str | Path,
                 grid_file: str | Path, output_dir: str | Path,
                 output_name: str = "projected_grid"):

    # ── Configuration ─────────────────────────────────────
    config = Config(Path(config), Path(mesh_file),
                    Path(grid_file), Path(output_dir))
    config.print_config()
    time.sleep(2)

    if not config.validate_paths():
        exit(1)
    time.sleep(2)

    config.create_output_folder()

    # ── Path ───────────────────────────────────────────────
    lamiamesh   = config.mesh_path
    imieipunti  = config.point_path
    ilmiooutput = config.output_path / f"{Path(output_name).stem}.xyz"

    # ── Load the data (ONCE) ──────────────────
    grid_obj = Grid(imieipunti)
    mesh = Mesh(lamiamesh)       

    grid      = grid_obj.getGrid()
    grid_step = grid_obj.getGridStep()
    vertices  = mesh.getVertices()
    faces     = mesh.getFaces()
    centroids = mesh.getCentroids()

    # ── Report dimension ──────────────────────────────────
    vertices_gb  = np.array(vertices).nbytes  / (1024**3)
    faces_gb     = np.array(faces).nbytes     / (1024**3)
    centroids_gb = np.array(centroids).nbytes / (1024**3)
    grid_gb      = np.array(grid).nbytes      / (1024**3)

    print("=" * 50)
    print("DATA SIZE REPORT")
    print("=" * 50)
    print(f"Mesh_vertices : {vertices_gb:.3f} GB | shape: {np.array(vertices).shape}")
    print(f"Mesh_faces    : {faces_gb:.3f} GB | shape: {np.array(faces).shape}")
    print(f"Centroids     : {centroids_gb:.3f} GB | shape: {np.array(centroids).shape}")
    print(f"Grid points   : {grid_gb:.3f} GB | shape: {np.array(grid).shape}")
    print(f"Total data   : {vertices_gb + faces_gb + centroids_gb + grid_gb:.3f} GB")
    print(f"Estimation using {config.cores} workers : "
          f"{(vertices_gb + faces_gb + centroids_gb) * config.cores:.3f} GB")
    print("=" * 50)
    print()

    # ── Projector ────────────────────────────────
    # config.closed_surface = True  → closed surface (KD-tree 3D)
    # config.closed_surface = False → open surface (KD-tree 2D)
    ProjectorClass = ClosedSurfaceProjector if config.closed_surface \
                     else OpenSurfaceProjector

    # ── Projection  ─────────────────────────────────────────
    projected = ProjectorClass(
        grid,
        vertices,
        faces,
        tolerance=config.tolerance,
        buffer=config.buffer,
        radius=config.radius,
        normal=config.normal_grid,
        centroids=centroids,
        output=ilmiooutput,
        core=config.cores,
        grid_step=grid_step
    )

    projected_points, triangle_indices, best_distance = \
        projected.project_ort_points_on_mesh()