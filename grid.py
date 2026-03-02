from plyfile import PlyData
import numpy as np

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
            print(f"Error reading XYZ file: {e}")
    
    #Return the points
    def getGrid(self):
        return self.pointsGrid

    #Return the number of the points
    def getNGridPt(self):
        return len(self.pointsGrid)
    
        # Calcola la normale al piano usando il metodo dei minimi quadrati
    def getNormalGrid(self):
        """
        Calcola la normale al piano usando il metodo dei minimi quadrati.
        Fittiamo un piano: ax + by + cz + d = 0
        
        Returns:
            normal (numpy.ndarray): Vettore normale al piano (a, b, c) normalizzato
            d (float): Coefficiente d dell'equazione del piano
        """
        if len(self.pointsGrid) < 3:
            raise ValueError("Servono almeno 3 punti per calcolare un piano")
        
        # Estrai coordinate
        x = self.pointsGrid[:, 0]
        y = self.pointsGrid[:, 1]
        z = self.pointsGrid[:, 2]
        
        # Costruisci la matrice A per il sistema Ax = b
        # Risolviamo: z = ax + by + c
        A = np.c_[x, y, np.ones(len(self.pointsGrid))]
        
        # Risolvi il sistema ai minimi quadrati
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        
        a, b, c = coeffs[0], coeffs[1], -1
        d = coeffs[2]
        
        # Normalizza il vettore normale
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        return normal, d