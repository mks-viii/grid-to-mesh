#import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from sympy import Matrix, sqrt
from plyfile import PlyData

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
            print(f"Error reading PLY file: {e}")

    def precompute_centroids(self):
        print("Pre-computing centroids...")
        self.centroids = np.mean(self.mesh_vertices[self.mesh_faces], axis=1)
        print(f"Camputed {len(self.centroids)} centroids")
    
    def getCentroids(self):
        return self.centroids

    def getFaces(self):
        return self.mesh_faces
    
    def getVertices(self):
        return self.mesh_vertices