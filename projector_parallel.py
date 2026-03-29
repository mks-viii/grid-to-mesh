import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from multiprocessing import Queue
# ============================================================
# FUNZIONI GEOMETRICHE (NumPy puro su coordinate locali)
# ============================================================
import time  # ← timestamp

from multiprocessing import Queue

def process_point_chunk(args):
    (points_chunk,
     mesh_vertices,
     chunk_mesh_faces,
     chunk_centroids,
     tolerance,
     radius,
     mynormal,
     local_origin,
     progress_queue) = args  # ← aggiunta queue

    mesh_vertices_local = mesh_vertices  - local_origin
    centroids_local     = chunk_centroids - local_origin
    points_local        = points_chunk   - local_origin

    kdtree = cKDTree(centroids_local[:, :2])  # solo X e Y

    chunk_results = []

    for t_point in points_local:

        point_bbox_min  = t_point - radius
        point_bbox_max  = t_point + radius
        center          = (point_bbox_min + point_bbox_max) / 2
        search_radius   = np.linalg.norm(point_bbox_max - center) * 2
        nearest_indices = kdtree.query_ball_point(center[:2], search_radius)

        candidates = []

        for local_index in nearest_indices:
            face     = chunk_mesh_faces[local_index]
            triangle = mesh_vertices_local[face]

            if not _is_point_in_triangle_barycentric_2D(
                t_point, triangle, tolerance, mynormal
            ):
                continue

            result = _intersezione_piano_retta_NUM(triangle, t_point, mynormal)

            if not result[2] or result[0] is None:
                continue

            try:
                projected_point = np.array(result[0])
                t_param         = result[3]


                filtered = _filtro_piano_NUM(triangle, projected_point, mynormal)

                if not filtered[2] or filtered[0] is None:
                    continue

                proj   = np.array(filtered[0])
                soglia = float(filtered[1])
                dist   = np.linalg.norm(proj - t_point)

                candidates.append({
                    'point_local': proj,
                    'soglia':      soglia,
                    'distance':    dist,
                    't_param':     t_param
                })

            except (TypeError, IndexError, AttributeError):
                continue

        if candidates:
            best        = min(candidates, key=lambda c: c['distance'])
            point_world = best['point_local'] + local_origin

            chunk_results.append({
                'east':   float(point_world[0]),
                'north':  float(point_world[1]),
                'height': float(point_world[2]),
                'soglia': best['soglia']
            })

        # ── Aggiorna progress bar dopo OGNI punto ─────────
        progress_queue.put(1)

    return chunk_results

def _is_point_in_triangle_barycentric_2D(point, triangle, tolerance, mynormal):
    """Test baricentrico 2D - invariante alla traslazione."""
    p  = np.array([point[0],       point[1]])
    v0 = np.array([triangle[0][0], triangle[0][1]])
    v1 = np.array([triangle[1][0], triangle[1][1]])
    v2 = np.array([triangle[2][0], triangle[2][1]])

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p  = p  - v0

    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1.0 + tolerance)


def _intersezione_piano_retta_NUM(triangle, point, normalGrid, tol=1e-12):
    """
    Intersezione piano-retta con NumPy puro.
    Corretto su coordinate locali (valori piccoli,
    no cancellazione numerica).
    """
    A  = np.array(triangle[0], dtype=np.float64)
    B  = np.array(triangle[1], dtype=np.float64)
    C  = np.array(triangle[2], dtype=np.float64)
    P0 = np.array(point,       dtype=np.float64)
    v  = np.array(normalGrid,  dtype=np.float64)

    AB = B - A
    AC = C - A
    n  = np.cross(AB, AC)

    norm_n = np.linalg.norm(n)
    if norm_n < tol:
        return None, None, False, None

    n = n / norm_n

    denom = np.dot(n, v)
    if abs(denom) < tol:
        return None, None, False, None

    t      = np.dot(n, A - P0) / denom
    P      = P0 + t * v

    # Proiezione esatta sul piano
    dist_scalar = np.dot(n, P - A)
    P_proj      = P - dist_scalar * n
    dist        = abs(np.dot(n, P_proj - A))

    if dist < tol:
        return P_proj.tolist(), float(dist), True, float(t)
    else:
        return None, float(dist), False, None


def _filtro_piano_NUM(triangle, point, normalGrid, tol=1e-12):
    """
    Verifica che il punto sia sul piano del triangolo.
    NumPy puro su coordinate locali.
    """
    A = np.array(triangle[0], dtype=np.float64)
    B = np.array(triangle[1], dtype=np.float64)
    C = np.array(triangle[2], dtype=np.float64)
    P = np.array(point,       dtype=np.float64)

    AB = B - A
    AC = C - A
    n  = np.cross(AB, AC)

    norm_n = np.linalg.norm(n)
    if norm_n < tol:
        return None, 0.0, False

    n    = n / norm_n
    dist = abs(np.dot(n, P - A))

    if dist < tol:
        return P.tolist(), float(dist), True
    else:
        return None, float(dist), False


# ============================================================
# WORKER FUNCTION (standalone per ProcessPoolExecutor)
# ============================================================

def process_point_chunk(args):
    """
    Processa un chunk di punti in sistema di coordinate locale.

    Flusso:
        1. Riceve solo i triangoli pre-filtrati per il chunk (BB chunk)
        2. Trasla tutto nel sistema locale (origin = centroide mesh)
        3. Calcola proiezioni con NumPy puro
        4. Ri-trasla i risultati nel sistema originale

    Args:
        args: tupla con tutti i parametri necessari

    Returns:
        lista di dizionari con east, north, height, soglia
    """
    (points_chunk,
     mesh_vertices,       # vertici GLOBALI (array completo, indici originali)
     chunk_mesh_faces,    # solo le facce rilevanti per il chunk (indici globali)
     chunk_centroids,     # solo i centroidi rilevanti per il chunk
     tolerance,
     radius,
     mynormal,
     local_origin) = args

    # ── 1. Traslazione nel sistema locale ─────────────────────
    # mesh_vertices è l'array COMPLETO traslato:
    # gli indici in chunk_mesh_faces puntano a questo array
    # quindi funzionano correttamente dopo la traslazione
    mesh_vertices_local = mesh_vertices  - local_origin  # shape: (N_totale_vertici, 3)
    centroids_local     = chunk_centroids - local_origin  # shape: (N_chunk_faces, 3)
    points_local        = points_chunk   - local_origin  # shape: (N_chunk_points, 3)

    # ── 2. KD-tree sui soli centroidi del chunk ───────────────
    # Più veloce perché contiene solo i triangoli rilevanti
    kdtree = cKDTree(centroids_local)

    chunk_results = []

    for t_point in points_local:

        # BB locale per questo specifico punto
        point_bbox_min  = t_point - radius
        point_bbox_max  = t_point + radius
        center          = (point_bbox_min + point_bbox_max) / 2
        search_radius   = np.linalg.norm(point_bbox_max - center) * 2

        # Indici locali nel chunk (0..N_chunk_faces)
        nearest_indices = kdtree.query_ball_point(center, search_radius)

        candidates = []

        for local_index in nearest_indices:
            # chunk_mesh_faces[local_index] → indici GLOBALI in mesh_vertices
            # mesh_vertices_local usa gli stessi indici globali → corretto
            face     = chunk_mesh_faces[local_index]
            triangle = mesh_vertices_local[face]  # shape: (3, 3)

            # Test baricentrico 2D
            if not _is_point_in_triangle_barycentric_2D(
                t_point, triangle, tolerance, mynormal
            ):
                continue

            # Intersezione piano-retta
            result = _intersezione_piano_retta_NUM(
                triangle, t_point, mynormal
            )

            if not result[2] or result[0] is None:
                continue

            try:
                projected_point = np.array(result[0])
                t_param         = result[3]

                # Filtro piano
                filtered = _filtro_piano_NUM(
                    triangle, projected_point, mynormal
                )

                if not filtered[2] or filtered[0] is None:
                    continue

                proj   = np.array(filtered[0])
                soglia = float(filtered[1])
                dist   = np.linalg.norm(proj - t_point)

                candidates.append({
                    'point_local': proj,
                    'soglia':      soglia,
                    'distance':    dist,
                    't_param':     t_param
                })

            except (TypeError, IndexError, AttributeError):
                continue

        if candidates:
            best = min(candidates, key=lambda c: c['distance'])

            # ── 3. Ri-traslazione nel sistema originale ────────
            point_world = best['point_local'] + local_origin

            chunk_results.append({
                'east':   float(point_world[0]),
                'north':  float(point_world[1]),
                'height': float(point_world[2]),
                'soglia': best['soglia']
            })

    return chunk_results


# ============================================================
# CLASSE PROJECTOR
# ============================================================

class Projector:
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
        self.kdtree        = None
        self.grid_step     = grid_step

        # ── Origine locale: centroide della mesh ──────────────
        # Calcolata UNA VOLTA, condivisa da tutti i worker sul centroide della mesh
        self.local_origin = np.mean(self.mesh_vertices, axis=0)
        print(f"Local origin (mesh centroid): {self.local_origin}")

        self.build_kdtree()
        print()

    def build_kdtree(self):
        print("Build the KD-tree of the centroids...")
        self.kdtree = cKDTree(self.centroids[:, :2]) # solo X e Y
        print(f"KD-tree built using {len(self.centroids)} centroids")

    def _compute_chunk_bbox(self, chunk):
        """
        Calcola il bounding box di un chunk di punti,
        espanso di radius in ogni direzione.
        """
        bbox_min = np.min(chunk, axis=0) - self.radius
        bbox_max = np.max(chunk, axis=0) + self.radius
        return bbox_min, bbox_max

    def _filter_triangles_for_chunk(self, chunk):
        """
        Filtra i triangoli rilevanti per un chunk
        cercando vicini per OGNI punto del chunk
        e unendo gli indici trovati.
        """
        relevant_idx = set()
        
        for point in chunk:
            indices = self.kdtree.query_ball_point(point[:2], self.grid_step)
            relevant_idx.update(indices)
        return np.array(list(relevant_idx), dtype=int)



    def _create_chunks(self, points, n_chunks):
        """
        Divide i punti in chunk contigui spazialmente
        usando PCA per trovare la direzione principale
        della griglia (funziona anche con griglie ruotate)
        """
        from numpy.linalg import eigh
        
        # ── PCA: trova la direzione principale ────────────────
        pts_2d  = points[:, :2]  # usa solo X e Y
        center  = np.mean(pts_2d, axis=0)
        pts_c   = pts_2d - center
        cov     = np.cov(pts_c.T)
        vals, vecs = eigh(cov)
        
        # Vettore della direzione principale (autovettore max)
        main_dir = vecs[:, np.argmax(vals)]
        
        # ── Proietta i punti sulla direzione principale ────────
        projections = pts_c @ main_dir
        
        # ── Ordina per proiezione ──────────────────────────────
        sorted_idx    = np.argsort(projections)
        points_sorted = points[sorted_idx]
        
        # ── Divide in chunk ───────────────────────────────────
        chunk_size = math.ceil(len(points_sorted) / n_chunks)
        return [
            points_sorted[i:i + chunk_size]
            for i in range(0, len(points_sorted), chunk_size)
        ]

    def project_ort_points_on_mesh(self):
        print()
        print("***********************************")
        print("*    PROJECTION POINT ON MESH     *")
        print("***********************************")
        print()

        n_workers = self.core
        print(f"Parallelization: {n_workers} workers (ProcessPoolExecutor)")

        # ── 1. Divide i punti in chunk ────────────────────────
        print(f"Total points to project: {len(self.points)}")
        chunks = self._create_chunks(self.points, n_workers)
        print(f"Chunks: {len(chunks)} x ~{len(chunks[0])} points each")

        print("\nDEBUG chunk bboxes:")
        for i, chunk in enumerate(chunks):
            bbox_min = np.min(chunk, axis=0)
            bbox_max = np.max(chunk, axis=0)
            print(f"  Chunk {i}: "
                f"X[{bbox_min[0]:.3f} -> {bbox_max[0]:.3f}] "
                f"Y[{bbox_min[1]:.3f} -> {bbox_max[1]:.3f}] "
                f"Z[{bbox_min[2]:.3f} -> {bbox_max[2]:.3f}]")
        print()

        # ── 2. Pre-filtro per chunk ───────────────────────────
        print("\nPre-filtering triangles per chunk...")
        args_list = []

        for i, chunk in enumerate(chunks):

            # ← cerca triangoli punto per punto nel chunk
            relevant_idx = self._filter_triangles_for_chunk(chunk)

            if len(relevant_idx) == 0:
                print(f"  Chunk {i}: no relevant triangles, skipped")
                continue

            chunk_faces     = self.mesh_faces[relevant_idx]
            chunk_centroids = self.centroids[relevant_idx]

            print(f"  Chunk {i}: {len(chunk)} points | "
                f"{len(relevant_idx)} relevant triangles")

            args_list.append((
                chunk,
                self.mesh_vertices,
                chunk_faces,
                chunk_centroids,
                self.tolerance,
                self.radius,
                self.mynormal,
                self.local_origin
            ))

        if not args_list:
            print("⚠️  No chunks to process!")
            return pd.DataFrame(), [], []

        # ── 3. Esecuzione parallela ───────────────────────────
        print(f"\nStarting parallel projection...")

       

        print(f"[{time.strftime('%H:%M:%S')}] Avvio ProcessPoolExecutor...")

        results_list = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:

            print(f"[{time.strftime('%H:%M:%S')}] Worker pronti, invio chunks...")

            futures = {
                executor.submit(process_point_chunk, args): i
                for i, args in enumerate(args_list)
            }

            print(f"[{time.strftime('%H:%M:%S')}] Chunks inviati, attendo risultati...")

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
                        print(f"\n⚠️  Error in chunk {chunk_idx}: {e}")
                    finally:
                        pbar.update(len(chunks[chunk_idx]))

        print(f"[{time.strftime('%H:%M:%S')}] Tutti i worker completati")



        # ── 4. DataFrame e output ─────────────────────────────
        print("\nCreate results dataframe...")
        panda = pd.DataFrame(results_list)

        if panda.empty:
            print("⚠️  NO POINTS PROJECTED!")
            return pd.DataFrame(), [], []

        # Per ogni (east, north) tieni la soglia minima
        panda_filtered = panda.loc[
            panda.groupby(
                [panda['east'].round(6), panda['north'].round(6)]
            )['soglia'].idxmin()
        ]

        print()
        print("***********************************")
        print("*      PROJECTION COMPLETED       *")
        print("***********************************")
        print(f"Projected points: {len(panda_filtered)}")
        print()

        # Salvataggio
        output_dir = os.path.dirname(self.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output folder created: {output_dir}")

        xyz_data = panda_filtered[['east', 'north', 'height']]
        xyz_data.to_csv(self.output, index=False, header=False, sep=' ')
        print(f"Output saved: {self.output}")

        return panda_filtered, [], []