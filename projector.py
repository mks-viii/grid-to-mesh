from config import normalGrid

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from sympy import Matrix, sqrt

class Projector:
    def __init__(self, grid, vertices, faces, tolerance, buffer, radius, normal, centroids, output, core):
        self.points = np.array(grid)
        self.mesh_vertices = vertices
        self.mesh_faces = faces
        self.tolerance = tolerance
        self.buffer = buffer
        self.radius = radius
        self.normal = normal
        self.centroids = centroids
        self.kdtree = None
        self.build_kdtree()
        self.output=output
        self.core=core
        print()
        print()

    def build_kdtree(self):
        print("Build the KD-tree of the centroids...")
        self.kdtree = cKDTree(self.centroids)
        print(f"KD-tree built using {len(self.centroids)} points")

    def is_point_in_triangle_barycentric_2D(self, point, triangle):
        """
        Check if a point is inside a triangle using 2D barycentric coordinates.
        """
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
        
        #Check for degenerate triangle with zero area
        if abs(denom) < 1e-10:
            return False
        
        #Compute the barycentric coordinates
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        #Check if the point is inside the triangle using edge tolerance
        return (u >= -self.tolerance) and (v >= -self.tolerance) and (u + v <= 1.0 + self.tolerance)

    # This function search for the bounding box around the grid 
    def compute_bounding_box(self):
        print()
        print("****************************************")
        print("* class: Projector method: Compute Bounding Box *")
        print("****************************************")
        print()
        print("Compute Bounding Grid")
        if self.points is None or len(self.points) == 0:
           raise ValueError("I dont't have point for calulating the bounding box.")
        inner_point = np.array(self.points)
        min_coords = inner_point - self.buffer
        max_coords = inner_point + self.buffer
        return min_coords, max_coords
    
    #Funzione che trova il bounding box attorno ad un punto singolo
    def compute_point_bounding_box(self, point):
        mypoint = point
        if mypoint is None or len(mypoint) == 0:
            raise ValueError("Non ho punti per calcolare il bounding box.")

        mypoint = np.array(mypoint)
        min_coords = mypoint - self.radius
        max_coords = mypoint + self.radius
        return min_coords, max_coords

    
    #Filter the triangles using the pre-calculated KD-tree
    def filter_triangles(self, bbox_min, bbox_max):
        center = (bbox_min + bbox_max) / 2
        radius = np.linalg.norm(bbox_max - center) * 2
        indices = self.kdtree.query_ball_point(center, radius)
        return indices

    """
    A, B, C: points of the plane
    point: point for mthe line
    normalGrid: normal to the grid  
    tol: tolerance along the border
    """
    @staticmethod
    def intersezione_piano_retta_SYM(triangle, point, normalGrid, tol=1e-12):
        A, B, C = triangle
        P0 = point
        v = normalGrid

        # Convert inputs to SymPy symbolic objects
        A = Matrix(A)
        B = Matrix(B)
        C = Matrix(C)
        P0 = Matrix(P0)
        v = Matrix(v)

        # Compute plane vectors
        AB = B - A
        AC = C - A

        # Plane vector
        n = AB.cross(AC)
        # Normalized plane vectors
        n = n.normalized()

        denom = n.dot(v)
        if denom == 0:
            return None, None, False

        # T defines the position along the line 
        t = n.dot(A - P0) / denom

        def proietta_sul_piano(P, A, n):
            dist = (n.dot(P - A)) / n.dot(n)
            P_proj = P - dist * n
            return P_proj
        
        # Intersection point
        P = P0 + t * v

        # Search on the plane
        P = proietta_sul_piano(P, A, n)
        # Verify the distance between the point an the plane
        dist = abs(n.dot(P - A)) / sqrt(n.dot(n))

        if dist < tol:
            return P, dist, True
        else:
            return None, dist, False

    @staticmethod
    def filtro_piano_SYM(triangle, point, normalGrid, tol=1e-12):
        """
        A, B, C: points of the plane
        point: point for mthe line
        normalGrid: normal to the grid  
        tol: tolerance along the border
        """
        A, B, C = triangle
        P = point
        v = normalGrid

        A = Matrix(A)
        B = Matrix(B)
        C = Matrix(C)
        P = Matrix(P)

        # Compute plane vetors
        AB = B - A
        AC = C - A

        # Normal vector
        n = AB.cross(AC)
        n = n.normalized()      
        
        # Verify the distance between the projected point and the plane
        dist = abs(n.dot(P - A)) / sqrt(n.dot(n))

        if dist < tol:
            return P, dist, True
        else:
            return None, dist, False
    
    def project_ort_points_on_mesh(self):
        print()
        print("***********************************")
        print("* PROJECTION POINT ON MESH *")
        print("***********************************")
        print()
        
        # List for collet the risults
        results_list = []

        # 1 global BB 
        print("Compute bounding box")
        obbox_min, obbox_max = self.compute_bounding_box()
        
        print("Filter relevant triangles")
        relevant_triangle_indices = self.filter_triangles(obbox_min, obbox_max)

        # 2 CLEANING OF THE INDICES (flat int cast) - VECTORIZATION
        flat_indices = [item for sublist in relevant_triangle_indices for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
        relevant_triangle_indices = np.array(flat_indices, dtype=int)

        # 3. EXTRACT THE RELEVANT FACES INTO THE GLOBAL BB
        relevant_faces = self.mesh_faces[relevant_triangle_indices]

        # 4. CYCLE THROUGH EACH POINT WITH PROGRESSS BAR
        print(f"\nProjection of {len(self.points)} points on the mesh...")
        
        # Use tqdm for the progress bar
        for index, t_point in enumerate(tqdm(self.points, 
                                              desc="Proiezione punti", 
                                              unit="pt",
                                              colour="green",
                                              ncols=100)):
           
            # 4a. LOCAL BB for the actual point
            point_bbox_min, point_bbox_max = self.compute_point_bounding_box(t_point)

            # 4b. FILTER THE NEAREST triangles 
            relevant_triangle_to_point_indices = self.filter_triangles(point_bbox_min, point_bbox_max)

            # 4c. GLOBAL INDICES
            nearest_indices = relevant_triangle_to_point_indices

            # 4d. FOR EACH TRIANGKLE
            for local_index in nearest_indices:
                if len(relevant_faces) == 0:
                    continue                                      
                
                face = self.mesh_faces[local_index]
                triangle = self.mesh_vertices[face]    

                # use the barycentric method 
                if self.is_point_in_triangle_barycentric_2D(t_point, triangle): 
                    
                    # 4e. # Line-plane intersection (vertical projection)
                    z_interpolated2 = self.intersezione_piano_retta_SYM(triangle, t_point, normalGrid)      
                    print(z_interpolated2)
                    # z_interpolated2[0] is SymPy Matrix, convert to number
                    if z_interpolated2[2] == False or z_interpolated2[0] is None:
                        continue

                    try:
                        z_value = float(z_interpolated2[0][2])
                    except (TypeError, IndexError, AttributeError):
                        continue

                    # original coordinates  
                    #x, y, _ = t_point
                    x= float(z_interpolated2[0][0])
                    y= float(z_interpolated2[0][1])
                    # projected coordinates
                    projected_point = np.array([x, y, z_value])
                    
                    """
                    Apply filter to find only the point lying on the plane of the face
                    directly below the considered point.
                    Avoids projection onto adjacent face planes.
                    """  
                    filtered = self.filtro_piano_SYM(triangle, projected_point, normalGrid)
                    
                    if filtered[2] == False or filtered[0] is None:
                        continue

                    try:
                        est = float(filtered[0][0])
                        ovest = float(filtered[0][1])
                        elev = float(filtered[0][2])
                        soglia = float(filtered[1])
                    except (TypeError, IndexError, AttributeError):
                        continue
                    
                    # append list
                    results_list.append({
                        'east': est, 
                        'north': ovest, 
                        'height': elev, 
                        'soglia': soglia
                    })

        print("\nCreate the results dataframe")
        panda = pd.DataFrame(results_list)

        #controllo sul DF
        if panda.empty:
            print("NO POINT PROJECTED!")
            return pd.DataFrame(), [], []       

        print("Filtered.")


        # round the result for avoiding floating-point errors
        # group by and filter
        panda_filtered = panda.loc[
            panda.groupby([panda['east'].round(6), panda['north'].round(6)])['soglia'].idxmin()
        ]

        print()
        print("***********************************")
        print("* PROJECTION COMPLETED *")
        print("***********************************")
        print(f"Projected points: {len(panda_filtered)}")
        print()

        xyz_data = panda_filtered[['east', 'north', 'height']]

            # Verifica e crea la directory di output se non esiste
        output_dir = os.path.dirname(self.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Output folder created: {output_dir}")

        print(f"Output saved in: {self.output}")
        xyz_data.to_csv(self.output, index=False, header=False, sep=' ')
        
        return panda, [], []