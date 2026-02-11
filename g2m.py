import pandas as pd
from plyfile import PlyData
import os
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from sympy import Matrix, sqrt

script_dir = os.path.dirname(os.path.abspath(__file__))

inputFolder = 'input_debug'
meshName = "100GOR.ply"
pointName = '36PT_GOR.xyz'
output_name = 'RIS_GOR.xyz'
outputFolder = 'output_debug'

lamiamesh = os.path.join(script_dir, inputFolder, meshName)
imieipunti = os.path.join(script_dir, inputFolder, pointName)
ilmiooutput = os.path.join(script_dir, outputFolder, output_name)

#Tolerance attorno al triangolo
#A causa delle limitazioni della precisione dei numeri in virgola mobile, un punto che dovrebbe essere esattamente sul bordo di un triangolo 
#potrebbe risultare leggermente all'interno o all'esterno
#Punti che sono molto vicini al bordo vengono considerati all'interno del triangolo
#4e-1 per mesh alta risoluzione
#10 per mesh bassa risoluzione
tolerance = 0.01
buffer = 1.0
radius = 5.0

normalGrid = np.array([0.0, 0.0, 1.0])


class Grid:
    def __init__(self, grid_file_name):         
        self.grid = grid_file_name
        self.pointsGrid = []
        self.read_xyz()
        
    def read_xyz(self):    
        try:
            points = []
            with open(self.grid, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            self.pointsGrid = np.array(points)
        except Exception as e:
            print(f"Errore durante la lettura del file XYZ: {e}")
    
    #ritorna i punti
    def getGrid(self):
        return self.pointsGrid

    #ritorna il numero dei punti 
    def getNGridPt(self):
        return len(self.pointsGrid)


class Mesh:
    def __init__(self, ply_file_name):   
        self.ply = ply_file_name
        self.mesh_vertices = None
        self.mesh_faces = None
        self.centroids = None
        self.read_ply()
        self.precompute_centroids()
        
    def read_ply(self):
        try:
            plydata = PlyData.read(self.ply)
            vertices = np.array([(v[0], v[1], v[2]) for v in plydata['vertex']])
            faces = np.array([f[0] for f in plydata['face']])
            self.mesh_vertices, self.mesh_faces = vertices, faces
        except Exception as e:
            print(f"Errore durante la lettura del file PLY: {e}")

    def precompute_centroids(self):
        print("Pre-calcolo centroidi dei triangoli...")
        self.centroids = np.mean(self.mesh_vertices[self.mesh_faces], axis=1)
        print(f"Calcolati {len(self.centroids)} centroidi")
    
    def getCentroids(self):
        return self.centroids

    def getFaces(self):
        return self.mesh_faces
    
    def getVertices(self):
        return self.mesh_vertices


class Projector:
    def __init__(self, grid, vertices, faces, tolerance, buffer, radius, normal, centroids):
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
        print()
        print()

    def build_kdtree(self):
        print("Costruzione KD-tree dei centroidi...")
        self.kdtree = cKDTree(self.centroids)
        print(f"KD-tree costruito con {len(self.centroids)} punti")

    def is_point_in_triangle_barycentric_2D(self, point, triangle):
        """
        Verifica se un punto è dentro un triangolo usando coordinate baricentriche in 2D
        Molto più efficiente e accurato del metodo precedente
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
        
        #controllo triangolo degenero di area 0
        if abs(denom) < 1e-10:
            return False
        
        # Calcola coordinate baricentriche
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Verifica se il punto è nel triangolo usando la tolleranza al bordo
        return (u >= -self.tolerance) and (v >= -self.tolerance) and (u + v <= 1.0 + self.tolerance)

    #Funzione che trova il bounding box attorno alal griglia
    def compute_bounding_box(self):
        print()
        print("****************************************")
        print("* class: Projector method: Compute Bounding Box *")
        print("****************************************")
        print()
        print("Compute Bounding Grid")
        if self.points is None or len(self.points) == 0:
           raise ValueError("Non ho punti per calcolare il bounding box.")
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

    """
    Filtra i triangoli usando il KD-tree pre-calcolato.
    """
    def filter_triangles(self, bbox_min, bbox_max):
        center = (bbox_min + bbox_max) / 2
        radius = np.linalg.norm(bbox_max - center) * 2
        indices = self.kdtree.query_ball_point(center, radius)
        return indices

    """
    A, B, C: punti del piano 
    point: punto della retta 
    normalGrid: normale alla griglia 
    tol: tolleranza numerica per verificare la vicinanza al piano
    """
    @staticmethod
    def intersezione_piano_retta_SYM(triangle, point, normalGrid, tol=1e-12):
        A, B, C = triangle
        P0 = point
        v = normalGrid

        # Converte input in oggetti simbolici di SymPy (Coordinata cartesiana 3D)
        A = Matrix(A)
        B = Matrix(B)
        C = Matrix(C)
        P0 = Matrix(P0)
        v = Matrix(v)

        # Calcolo dei vettori del piano
        AB = B - A
        AC = C - A

        # Vettore normale al piano
        n = AB.cross(AC)
        # Normalizzo il vettore normale
        n = n.normalized()

        denom = n.dot(v)
        if denom == 0:
            return None, None, False

        # Parametro t che definisce la posizione lungo la retta
        t = n.dot(A - P0) / denom

        def proietta_sul_piano(P, A, n):
            dist = (n.dot(P - A)) / n.dot(n)
            P_proj = P - dist * n
            return P_proj
        
        # Punto di intersezione
        P = P0 + t * v

        # Cerca sul piano
        P = proietta_sul_piano(P, A, n)
        # Verifica la distanza del punto dal pian
        dist = abs(n.dot(P - A)) / sqrt(n.dot(n))

        if dist < tol:
            return P, dist, True
        else:
            return None, dist, False

    @staticmethod
    def filtro_piano_SYM(triangle, point, normalGrid, tol=1e-12):
        """
        A, B, C: punti del piano 
        point: punto della retta 
        normalGrid: normale alla griglia 
        tol: tolleranza numerica per verificare la vicinanza al piano
        """
        A, B, C = triangle
        P0 = point
        v = normalGrid

        A = Matrix(A)
        B = Matrix(B)
        C = Matrix(C)
        P0 = Matrix(P0)
        v = Matrix(v)

        # Calcolo dei vettori del piano
        AB = B - A
        AC = C - A

        # Vettore normale al piano
        n = AB.cross(AC)
        n = n.normalized()
        
        # Assicurati che P, i lpunto di intersezione, sia una Matrix
        P = Matrix(point)
        
        # Verifica la distanza del punto dal piano (proiezione)
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
        
        # Lista per accumulare i risultati invece del DataFrame
        results_list = []

        # 1 BB globale
        print("Calcolo bounding box")
        obbox_min, obbox_max = self.compute_bounding_box()
        
        print("Filtro triangoli rilevanti")
        relevant_triangle_indices = self.filter_triangles(obbox_min, obbox_max)

        # 2 PULIZIA DEGLI INDICI (appiattimento e conversione a int) - VETTORIZZATO
        flat_indices = [item for sublist in relevant_triangle_indices for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
        relevant_triangle_indices = np.array(flat_indices, dtype=int)

        # 3. ESTRAE LE FACCE RILEVANTI del BB globale
        relevant_faces = self.mesh_faces[relevant_triangle_indices]

        # 4. CICLO PER OGNI PUNTO CON BARRA DI PROGRESSO
        print(f"\nProiezione di {len(self.points)} punti sulla mesh...")
        
        # Usa tqdm per mostrare la barra di progresso
        for index, t_point in enumerate(tqdm(self.points, 
                                              desc="Proiezione punti", 
                                              unit="pt",
                                              colour="green",
                                              ncols=100)):
           
            # 4a. BB Locale il punto corrente
            point_bbox_min, point_bbox_max = self.compute_point_bounding_box(t_point)

            # 4b.  ILTRA TRIANGOLI vicini al punto
            relevant_triangle_to_point_indices = self.filter_triangles(point_bbox_min, point_bbox_max)

            # 4c. indice globale
            nearest_indices = relevant_triangle_to_point_indices

            # 4d. PER OGNI TRIANGOLO VICINO
            for local_index in nearest_indices:
                if len(relevant_faces) == 0:
                    continue                                      
                
                face = self.mesh_faces[local_index]
                triangle = self.mesh_vertices[face]    

                # Usa il nuovo metodo barycentric ottimizzato
                if self.is_point_in_triangle_barycentric_2D(t_point, triangle): 
                    
                    # 4e. INTERSEZIONE retta-piano (proiezione verticale) 
                    z_interpolated2 = self.intersezione_piano_retta_SYM(triangle, t_point, normalGrid)      
      
                    # z_interpolated2[0] è un SymPy Matrix, devi estrarre il valore numerico
                    if z_interpolated2[2] == False or z_interpolated2[0] is None:
                        continue

                    try:
                        z_value = float(z_interpolated2[0][2])
                    except (TypeError, IndexError, AttributeError):
                        continue

                    # coordinate del punto da proiettare    
                    x, y, _ = t_point

                    #punto proiettato
                    projected_point = np.array([x, y, z_value])
                    
                    """
                    Applica il filtro per trovare solo il punto che giace sul piano della faccia nadirale al punto considerato
                    Evita la proiezione su piani di facce adiacenti                        
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
                    
                    # append lista più veloce 
                    results_list.append({
                        'east': est, 
                        'north': ovest, 
                        'height': elev, 
                        'soglia': soglia
                    })

        print("\nCreo il DataFrame risultati.")
        panda = pd.DataFrame(results_list)

        #controllo sul DF
        if panda.empty:
            print("NESSUN PUNTO PROIETTATO!")
            return pd.DataFrame(), [], []       

        print("Filtro i risultati ottimali.")

        #qui raggruppo per coordiante uguali e filtro solo quello con la soglia minima
        panda_filtered = panda.loc[panda.groupby(['east', 'north'])['soglia'].idxmin()]    
        
        print()
        print("***********************************")
        print("* PROIEZIONE COMPLETATA *")
        print("***********************************")
        print(f"Punti proiettati: {len(panda_filtered)}")
        print()

        xyz_data = panda_filtered[['east', 'north', 'height']]

        print(f"Salvataggio risultati in: {ilmiooutput}")
        xyz_data.to_csv(ilmiooutput, index=False, header=False, sep=',')
        
        return panda, [], []

        
grid = Grid(imieipunti)
mesh = Mesh(lamiamesh)

projected = Projector(
    grid.getGrid(), 
    mesh.getVertices(), 
    mesh.getFaces(), 
    tolerance, 
     
    buffer=buffer, 
    radius=radius, 
    normal=normalGrid,
    centroids=mesh.getCentroids())
      
projected_points, triangle_indices, best_distance = projected.project_ort_points_on_mesh()