import os
import math
import time
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from g2m_projector.geometric_functions import (
    _is_point_in_triangle_barycentric_2D,
    _intersezione_piano_retta_NUM,
    _filtro_piano_NUM,
    _compute_candidates
)



# ============================================================
# WORKER FUNCTIONS (standalone for the ProcessPoolExecutor).
# Don't move. Pickle requires these function here 
# ============================================================

def process_point_chunk_open(args):
    """
    Worker for OPEN surface.
    - KD-tree 2D (X, Y)
    - Chunk pre-filtering active
    - local_origin for numerical stability
    """
    (points_chunk,
     mesh_vertices,
     chunk_mesh_faces,
     chunk_centroids,
     tolerance,
     radius,
     mynormal,
     local_origin) = args

    mesh_vertices_local = mesh_vertices   - local_origin
    centroids_local     = chunk_centroids - local_origin
    points_local        = points_chunk    - local_origin

    # KD-tree 2D: solo X e Y
    kdtree = cKDTree(centroids_local[:, :2])

    chunk_results = []

    for t_point in points_local:
        point_bbox_min  = t_point - radius
        point_bbox_max  = t_point + radius
        center          = (point_bbox_min + point_bbox_max) / 2
        search_radius   = np.linalg.norm(point_bbox_max - center) * 2

        # Query 2D
        nearest_indices = kdtree.query_ball_point(center[:2], search_radius)

        candidates = _compute_candidates(
            t_point, nearest_indices,
            mesh_vertices_local, chunk_mesh_faces,
            tolerance, mynormal
        )

        if candidates:
            best        = min(candidates, key=lambda c: c['distance'])
            point_world = best['point_local'] + local_origin
            chunk_results.append({
                'east':   float(point_world[0]),
                'north':  float(point_world[1]),
                'height': float(point_world[2]),
                'soglia': best['soglia']
            })

    return chunk_results


def process_point_chunk_closed(args):
    """
    Worker for CLOSED surface.
    - KD-tree 3D (X, Y, Z) built on all centroids
    - No chunk pre-filtering
    - local_origin for numerical stability
    - Direct 3D query
    """
    (points_chunk,
     mesh_vertices,
     mesh_faces_all,
     centroids_all,
     tolerance,
     radius,
     mynormal,
     local_origin) = args

    mesh_vertices_local = mesh_vertices  - local_origin
    centroids_local     = centroids_all  - local_origin
    points_local        = points_chunk   - local_origin

    # KD-tree 3D over all centroids: X, Y e Z
    kdtree = cKDTree(centroids_local)

    chunk_results = []

    for t_point in points_local:
        point_bbox_min  = t_point - radius
        point_bbox_max  = t_point + radius
        center          = (point_bbox_min + point_bbox_max) / 2
        search_radius   = np.linalg.norm(point_bbox_max - center) * 2

        # Query the 3d centroids
        nearest_indices = kdtree.query_ball_point(center, search_radius)

        candidates = _compute_candidates(
            t_point, nearest_indices,
            mesh_vertices_local, mesh_faces_all,
            tolerance, mynormal
        )

        if candidates:
            best        = min(candidates, key=lambda c: c['distance'])
            point_world = best['point_local'] + local_origin
            chunk_results.append({
                'east':   float(point_world[0]),
                'north':  float(point_world[1]),
                'height': float(point_world[2]),
                'soglia': best['soglia']
            })

    return chunk_results


# ============================================================
# BASE CLASS
# ============================================================

class Projector:
    """
    Abstract base class for point projection on mesh.
    Handles chunking, parallelization and result saving.
    Subclasses implement the specific logic for
    open or closed surfaces
    """

    def __init__(self, grid, vertices, faces, tolerance, buffer,
                 radius, normal, centroids, output, core, grid_step):

        self.points        = np.array(grid,      dtype=np.float64)
        self.mesh_vertices = np.array(vertices,  dtype=np.float64)
        self.mesh_faces    = np.array(faces,      dtype=int)
        self.tolerance     = tolerance
        self.buffer        = buffer
        self.radius        = radius
        self.mynormal      = np.array(normal,    dtype=np.float64)
        self.centroids     = np.array(centroids, dtype=np.float64)
        self.output        = output
        self.core          = core
        self.grid_step     = grid_step
        self.kdtree        = None

        self.local_origin = np.mean(self.mesh_vertices, axis=0)
        print(f"Local origin (mesh centroid): {self.local_origin}")

        self.build_kdtree()
        print()

    # ----------------------------------------------------------
    # ABSTRACT METHODS  —  mandatory for the subclasses
    # ----------------------------------------------------------

    def build_kdtree(self):
        """Construct the KD-tree. 2D is ofr open, not used in closed."""
        raise NotImplementedError("Implementare nelle sottoclassi")

    def _filter_triangles_for_chunk(self, chunk):
        """Relevant triangles filter for chunk."""
        raise NotImplementedError("Implement in the child class")

    def _get_worker_function(self):
        """Return  the function worker standalone."""
        raise NotImplementedError("Implement in the child class")

    # ----------------------------------------------------------
    # COMMON METHODS
    # ----------------------------------------------------------

    def _create_chunks(self, points, n_chunks):
        """
        Splits points into spatially contiguous chunks
        using PCA to find the main direction
        of the grid.
        """
        from numpy.linalg import eigh

        pts_2d     = points[:, :2]
        center     = np.mean(pts_2d, axis=0)
        pts_c      = pts_2d - center
        cov        = np.cov(pts_c.T)
        vals, vecs = eigh(cov)

        main_dir      = vecs[:, np.argmax(vals)]
        projections   = pts_c @ main_dir
        sorted_idx    = np.argsort(projections)
        points_sorted = points[sorted_idx]

        chunk_size = math.ceil(len(points_sorted) / n_chunks)
        return [
            points_sorted[i:i + chunk_size]
            for i in range(0, len(points_sorted), chunk_size)
        ]

    def _make_chunk_args(self, chunk, chunk_faces, chunk_centroids):
        """
        Builds the argument tuple for a single chunk.
        Use it for the open and closed. 
        """
        return (
            chunk,
            self.mesh_vertices,
            chunk_faces,
            chunk_centroids,
            self.tolerance,
            self.radius,
            self.mynormal,
            self.local_origin
        )

    def _build_args_list(self, chunks):
        """
        Builds the argument list for all chunks.
        Uses _filter_triangles_for_chunk from the subclass.
        """
        args_list = []

        for i, chunk in enumerate(chunks):
            relevant_idx    = self._filter_triangles_for_chunk(chunk)
            chunk_faces     = self.mesh_faces[relevant_idx]
            chunk_centroids = self.centroids[relevant_idx]

            print(f"  Chunk {i}: {len(chunk)} points | "
                  f"{len(relevant_idx)} relevant triangles")

            args_list.append(
                self._make_chunk_args(chunk, chunk_faces, chunk_centroids)
            )

        return args_list

    def _save_results(self, results_list):
        """
        Creates DataFrame, filters by minimum threshold and saves to CSV.
        Common to all subclasses.
        """
        print("\nCreate results dataframe...")
        panda = pd.DataFrame(results_list)

        if panda.empty:
            print("⚠️  NO POINTS PROJECTED!")
            return pd.DataFrame(), [], []

        panda_filtered = panda.loc[
            panda.groupby(
                [panda['east'].round(6), panda['north'].round(6)]
            )['soglia'].idxmin()
        ]

        print("\n***********************************")
        print("*      PROJECTION COMPLETED       *")
        print("***********************************")
        print(f"Projected points: {len(panda_filtered)}\n")

        output_dir = os.path.dirname(self.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output folder created: {output_dir}")

        xyz_data = panda_filtered[['east', 'north', 'height']]
        xyz_data.to_csv(self.output, index=False, header=False, sep=' ')
        print(f"Output saved: {self.output}")

        return panda_filtered, [], []

    def project_ort_points_on_mesh(self):
        """
        Main projection method.
        It's the same for open and closed clasees. 
        Thee differences are in the chidl cleasses.
        """
        print("\n***********************************")
        print("*    PROJECTION POINT ON MESH     *")
        print("***********************************\n")

        n_workers = self.core
        print(f"Parallelization: {n_workers} workers (ProcessPoolExecutor)")
        print(f"Total points to project: {len(self.points)}")

        # 1. Divide i punti in chunk
        chunks = self._create_chunks(self.points, n_workers)
        print(f"Chunks: {len(chunks)} x ~{len(chunks[0])} points each")

        # 2. Debug bboxes
        print("\nDEBUG chunk bboxes:")
        for i, chunk in enumerate(chunks):
            bbox_min = np.min(chunk, axis=0)
            bbox_max = np.max(chunk, axis=0)
            print(f"  Chunk {i}: "
                  f"X[{bbox_min[0]:.3f} -> {bbox_max[0]:.3f}] "
                  f"Y[{bbox_min[1]:.3f} -> {bbox_max[1]:.3f}] "
                  f"Z[{bbox_min[2]:.3f} -> {bbox_max[2]:.3f}]")

        # 3. Costruct args_list
        print("\nPreparing chunks...")
        args_list = self._build_args_list(chunks)

        if not args_list:
            print("T.T No chunks to process!")
            return pd.DataFrame(), [], []

        # 4. Esecuzione parallela
        worker_fn    = self._get_worker_function()
        results_list = []

        print(f"\n[{time.strftime('%H:%M:%S')}] Starts ProcessPoolExecutor...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            print(f"[{time.strftime('%H:%M:%S')}] Worker ready, start chunks...")

            futures = {
                executor.submit(worker_fn, args): i
                for i, args in enumerate(args_list)
            }

            print(f"[{time.strftime('%H:%M:%S')}] Chunks sent, waiting for the results...")

            with tqdm(
                total=len(self.points),
                desc="Projection",
                unit="pt",
                colour="green",
                ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_result = future.result()
                        results_list.extend(chunk_result)
                    except Exception as e:
                        print(f"\nT.T  Error in chunk {chunk_idx}: {e}")
                    finally:
                        pbar.update(len(chunks[chunk_idx]))

        print(f"[{time.strftime('%H:%M:%S')}] All workers completed")

        # 5. Salvataggio risultati
        return self._save_results(results_list)


# ============================================================
# CHILD CLASS — OPEN SURFACE
# ============================================================

class OpenSurfaceProjector(Projector):
    """
    Projection on OPEN surface.
    - Class-level KD-tree 2D (X, Y) for chunk pre-filtering
    - Worker with KD-tree 2D and 2D query
    """

    def build_kdtree(self):
        print("Build the KD-tree 2D (X, Y) of the centroids...")
        self.kdtree = cKDTree(self.centroids[:, :2])
        print(f"KD-tree 2D built using {len(self.centroids)} centroids")

    def _filter_triangles_for_chunk(self, chunk):
        """
        2D Pre-filter: for each point in the chunk searches
        the nearest triangles using X and Y only.
        """
        relevant_idx = set()
        for point in chunk:
            indices = self.kdtree.query_ball_point(point[:2], self.grid_step)
            relevant_idx.update(indices)
        return np.array(list(relevant_idx), dtype=int)

    def _get_worker_function(self):
        return process_point_chunk_open


# ============================================================
# CHILD CLASS — CLOSED SURFACE
# ============================================================

class ClosedSurfaceProjector(Projector):
    """
    Projection on CLOSED surface.
    - No chunk pre-filtering: all global centroids
      are passed to the worker
    - KD-tree 3D (X, Y, Z) built inside the worker
    - Direct 3D query
    """

    def build_kdtree(self):
        # don't need a KD-tree for the closed surface
        # the worker uses all the global centroids
        print("ClosedSurfaceProjector: no class-level KD-tree needed")
        self.kdtree = None

    def _filter_triangles_for_chunk(self, chunk):
        """
        No pre-filtering: returns all mesh triangle indices.
        """
        return np.arange(len(self.mesh_faces), dtype=int)

    def _get_worker_function(self):
        return process_point_chunk_closed