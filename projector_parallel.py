import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from sympy import Matrix, sqrt
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# ============================================================
# FUNZIONI STANDALONE (necessarie per ProcessPoolExecutor
# su Windows - devono essere al di fuori della classe)
# ============================================================

def process_point_chunk(args):
    """
    Processa un chunk di punti.
    Deve essere una funzione standalone per essere serializzabile
    con pickle su Windows.
    
    Args:
        args: tupla con tutti i parametri necessari
    
    Returns:
        lista di risultati per il chunk
    """
    (points_chunk, mesh_vertices, mesh_faces, centroids,
     tolerance, radius, mynormal) = args

    # Ricostruisce il KD-tree nel processo figlio
    kdtree = cKDTree(centroids)

    chunk_results = []

    for t_point in points_chunk:
        # Bounding box locale per il punto
        point_bbox_min = t_point - radius
        point_bbox_max = t_point + radius

        # Filtra i triangoli vicini
        center = (point_bbox_min + point_bbox_max) / 2
        search_radius = np.linalg.norm(point_bbox_max - center) * 2
        nearest_indices = kdtree.query_ball_point(center, search_radius)

        candidates = []

        for local_index in nearest_indices:
            face = mesh_faces[local_index]
            triangle = mesh_vertices[face]

            # Test baricentrico 2D
            if _is_point_in_triangle_barycentric_2D(t_point, triangle, tolerance, mynormal):

                # Intersezione piano-retta
                result = _intersezione_piano_retta_SYM(triangle, t_point, mynormal)

                if result[2] == False or result[0] is None:
                    continue

                try:
                    x = float(result[0][0])
                    y = float(result[0][1])
                    z_value = float(result[0][2])
                    t_param = result[3]
                except (TypeError, IndexError, AttributeError):
                    continue

                projected_point = np.array([x, y, z_value])

                # Filtro piano
                filtered = _filtro_piano_SYM(triangle, projected_point, mynormal)

                if filtered[2] == False or filtered[0] is None:
                    continue

                try:
                    est = float(filtered[0][0])
                    ovest = float(filtered[0][1])
                    elev = float(filtered[0][2])
                    soglia = float(filtered[1])

                    distance = np.sqrt((est - t_point[0])**2 +
                                       (ovest - t_point[1])**2 +
                                       (elev - t_point[2])**2)

                    candidates.append({
                        'east': est,
                        'north': ovest,
                        'height': elev,
                        'soglia': soglia,
                        'distance': distance,
                        't_param': t_param
                    })
                except (TypeError, IndexError, AttributeError):
                    continue

        if candidates:
            best = min(candidates, key=lambda c: c['distance'])
            chunk_results.append({
                'east': best['east'],
                'north': best['north'],
                'height': best['height'],
                'soglia': best['soglia']
            })

    return chunk_results


def _is_point_in_triangle_barycentric_2D(point, triangle, tolerance, mynormal):
    """Versione standalone del metodo baricentrico."""
    p = np.array([point[0], point[1]])
    v0 = np.array([triangle[0][0], triangle[0][1]])
    v1 = np.array([triangle[1][0], triangle[1][1]])
    v2 = np.array([triangle[2][0], triangle[2][1]])

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

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


def _intersezione_piano_retta_SYM(triangle, point, normalGrid, tol=1e-12):
    """Versione standalone dell'intersezione piano-retta."""
    A, B, C = triangle
    P0 = point
    v = normalGrid

    A = Matrix(A)
    B = Matrix(B)
    C = Matrix(C)
    P0 = Matrix(P0)
    v = Matrix(v)

    AB = B - A
    AC = C - A
    n = AB.cross(AC)
    n = n.normalized()

    denom = n.dot(v)
    if denom == 0:
        return None, None, False, None

    t = n.dot(A - P0) / denom

    def proietta_sul_piano(P, A, n):
        dist = (n.dot(P - A)) / n.dot(n)
        return P - dist * n

    P = P0 + t * v
    P = proietta_sul_piano(P, A, n)
    dist = abs(n.dot(P - A)) / sqrt(n.dot(n))

    if dist < tol:
        return P, dist, True, float(t)
    else:
        return None, dist, False, None


def _filtro_piano_SYM(triangle, point, normalGrid, tol=1e-12):
    """Versione standalone del filtro piano."""
    A, B, C = triangle
    P = point

    A = Matrix(A)
    B = Matrix(B)
    C = Matrix(C)
    P = Matrix(P)

    AB = B - A
    AC = C - A
    n = AB.cross(AC)
    n = n.normalized()

    dist = abs(n.dot(P - A)) / sqrt(n.dot(n))

    if dist < tol:
        return P, dist, True
    else:
        return None, dist, False


# ============================================================
# CLASSE PROJECTOR
# ============================================================

class Projector:
    def __init__(self, grid, vertices, faces, tolerance, buffer, radius, normal, centroids, output, core):
        self.points = np.array(grid)
        self.mesh_vertices = vertices
        self.mesh_faces = faces
        self.tolerance = tolerance
        self.buffer = buffer
        self.radius = radius
        self.mynormal = normal
        self.centroids = centroids
        self.kdtree = None
        self.build_kdtree()
        self.output = output
        self.core = core
        print()
        print()

    def build_kdtree(self):
        print("Build the KD-tree of the centroids...")
        self.kdtree = cKDTree(self.centroids)
        print(f"KD-tree built using {len(self.centroids)} points")

    def is_point_in_triangle_barycentric_2D(self, point, triangle):
        """Check if a point is inside a triangle using 2D barycentric coordinates."""
        return _is_point_in_triangle_barycentric_2D(
            point, triangle, self.tolerance, self.mynormal
        )

    def compute_bounding_box(self):
        print()
        print("****************************************")
        print("* class: Projector method: Compute Bounding Box *")
        print("****************************************")
        print()
        print("Compute Bounding Grid")
        if self.points is None or len(self.points) == 0:
            raise ValueError("I don't have points for calculating the bounding box.")
        inner_point = np.array(self.points)
        min_coords = inner_point - self.buffer
        max_coords = inner_point + self.buffer
        return min_coords, max_coords

    def compute_point_bounding_box(self, point):
        mypoint = np.array(point)
        if mypoint is None or len(mypoint) == 0:
            raise ValueError("Non ho punti per calcolare il bounding box.")
        min_coords = mypoint - self.radius
        max_coords = mypoint + self.radius
        return min_coords, max_coords

    def filter_triangles(self, bbox_min, bbox_max):
        center = (bbox_min + bbox_max) / 2
        radius = np.linalg.norm(bbox_max - center) * 2
        indices = self.kdtree.query_ball_point(center, radius)
        return indices

    @staticmethod
    def intersezione_piano_retta_SYM(triangle, point, normalGrid, tol=1e-12):
        return _intersezione_piano_retta_SYM(triangle, point, normalGrid, tol)

    @staticmethod
    def filtro_piano_SYM(triangle, point, normalGrid, tol=1e-12):
        return _filtro_piano_SYM(triangle, point, normalGrid, tol)

    def _create_chunks(self, points, n_chunks):
        """
        Divide i punti in n_chunks bilanciati.
        
        Args:
            points: array di punti da dividere
            n_chunks: numero di chunk
        
        Returns:
            lista di array (chunk)
        """
        chunk_size = math.ceil(len(points) / n_chunks)
        return [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]

    def project_ort_points_on_mesh(self):
        print()
        print("***********************************")
        print("* PROJECTION POINT ON MESH *")
        print("***********************************")
        print()

        # Numero di core da usare
        n_workers = self.core
        print(f"Parallelization: {n_workers} workers (ProcessPoolExecutor)")

        # 1. Global BB
        print("Compute bounding box")
        obbox_min, obbox_max = self.compute_bounding_box()

        print("Filter relevant triangles")
        relevant_triangle_indices = self.filter_triangles(obbox_min, obbox_max)

        # 2. Cleaning indices
        flat_indices = [
            item for sublist in relevant_triangle_indices
            for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
        ]
        
        relevant_triangle_indices = np.array(flat_indices, dtype=int)

        # 3. Extract relevant faces
        relevant_faces = self.mesh_faces[relevant_triangle_indices]

        # 4. Divide i punti in chunk
        print(f"\nProjection of {len(self.points)} points on the mesh...")
        chunks = self._create_chunks(self.points, n_workers)
        print(f"Points divided into {len(chunks)} chunks of ~{len(chunks[0])} points each")

        # 5. Prepara gli argomenti per ogni worker
        args_list = [
            (
                chunk,
                self.mesh_vertices,
                self.mesh_faces,
                self.centroids,
                self.tolerance,
                self.radius,
                self.mynormal
            )
            for chunk in chunks
        ]

        # 6. Esecuzione parallela con ProcessPoolExecutor
        results_list = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Sottometti tutti i chunk
            futures = {
                executor.submit(process_point_chunk, args): i
                for i, args in enumerate(args_list)
            }

            # Raccoglie i risultati con progress bar
            with tqdm(total=len(self.points),
                      desc="Proiezione punti",
                      unit="pt",
                      colour="green",
                      ncols=100) as pbar:

                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_result = future.result()
                        results_list.extend(chunk_result)
                        # Aggiorna la progress bar con il numero di punti nel chunk
                        pbar.update(len(chunks[chunk_idx]))
                    except Exception as e:
                        print(f"\n⚠️ Errore nel chunk {chunk_idx}: {e}")
                        pbar.update(len(chunks[chunk_idx]))

        # 7. Crea il DataFrame
        print("\nCreate the results dataframe")
        panda = pd.DataFrame(results_list)

        if panda.empty:
            print("NO POINT PROJECTED!")
            return pd.DataFrame(), [], []

        print("Filtered.")

        # Arrotonda e filtra per evitare errori floating-point
        panda_filtered = panda.loc[
            panda.groupby(
                [panda['east'].round(6), panda['north'].round(6)]
            )['soglia'].idxmin()
        ]

        print()
        print("***********************************")
        print("* PROJECTION COMPLETED *")
        print("***********************************")
        print(f"Projected points: {len(panda_filtered)}")
        print()

        xyz_data = panda_filtered[['east', 'north', 'height']]

        # Verifica e crea la directory di output
        output_dir = os.path.dirname(self.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output folder created: {output_dir}")

        print(f"Output saved in: {self.output}")
        xyz_data.to_csv(self.output, index=False, header=False, sep=' ')

        return panda, [], []

