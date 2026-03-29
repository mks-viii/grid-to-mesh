import os
from out_name import get_output_name
from grid import Grid
from mesh import Mesh
from projector_parallel import OpenSurfaceProjector, ClosedSurfaceProjector
from import_config import Config
import time
import numpy as np

if __name__ == '__main__':

    # ── Configurazione ─────────────────────────────────────
    config = Config()
    config.print_config()
    time.sleep(2)

    if not config.validate_paths():
        exit(1)
    time.sleep(2)

    config.create_output_folder()

    # ── Path ───────────────────────────────────────────────
    script_dir   = config.get_script_dir
    lamiamesh    = config.mesh_path
    imieipunti   = config.point_path
    output_name  = get_output_name(config.mesh_name)
    ilmiooutput  = os.path.join(script_dir, config.output_path, output_name)

    # ── Caricamento dati (UNA VOLTA SOLA) ──────────────────
    grid_obj = Grid(imieipunti)
    mesh = Mesh(lamiamesh)       # ← eseguito solo dal main

    grid      = grid_obj.getGrid()
    grid_step = grid_obj.getGridStep()
    vertices  = mesh.getVertices()
    faces     = mesh.getFaces()
    centroids = mesh.getCentroids()

    # ── Report dimensioni ──────────────────────────────────
    vertices_gb  = np.array(vertices).nbytes  / (1024**3)
    faces_gb     = np.array(faces).nbytes     / (1024**3)
    centroids_gb = np.array(centroids).nbytes / (1024**3)
    grid_gb      = np.array(grid).nbytes      / (1024**3)

    print("=" * 50)
    print("DATA SIZE REPORT")
    print("=" * 50)
    print(f"mesh_vertices : {vertices_gb:.3f} GB | shape: {np.array(vertices).shape}")
    print(f"mesh_faces    : {faces_gb:.3f} GB | shape: {np.array(faces).shape}")
    print(f"centroids     : {centroids_gb:.3f} GB | shape: {np.array(centroids).shape}")
    print(f"grid points   : {grid_gb:.3f} GB | shape: {np.array(grid).shape}")
    print(f"Totale dati   : {vertices_gb + faces_gb + centroids_gb + grid_gb:.3f} GB")
    print(f"Stima con {config.cores} workers : "
          f"{(vertices_gb + faces_gb + centroids_gb) * config.cores:.3f} GB")
    print("=" * 50)
    print()

    # ── Scelta del Projector ────────────────────────────────
    # config.closed_surface = True  → superficie chiusa (KD-tree 3D)
    # config.closed_surface = False → superficie aperta (KD-tree 2D)
    ProjectorClass = ClosedSurfaceProjector if config.closed_surface \
                     else OpenSurfaceProjector
    
    # ── Proiezione ─────────────────────────────────────────
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