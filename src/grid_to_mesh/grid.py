from plyfile import PlyData
import numpy as np
from scipy.spatial import cKDTree

class Grid:
    def __init__(self, grid_file_name):         
        self.grid = grid_file_name
        self.pointsGrid = []
        self.read_xyz()
        self.step = self._compute_grid_step()
        
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
        Calculate the normal to the plane using least square
        Fit a plane: ax + by + cz + d = 0
        
        Returns:
            normal (numpy.ndarray): normal vector  (a, b, c) normalized
            d (float): Coefficient of the plane equation
        """
        if len(self.pointsGrid) < 3:
            raise ValueError("I need at least 3 point to calculate the plane")
        
        # Estrai coordinate
        x = self.pointsGrid[:, 0]
        y = self.pointsGrid[:, 1]
        z = self.pointsGrid[:, 2]
        
        # Define matrix A for the system  Ax = b
        # solve: z = ax + by + c
        A = np.c_[x, y, np.ones(len(self.pointsGrid))]
        
        # solve the system. Least square 
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        
        a, b, c = coeffs[0], coeffs[1], -1
        d = coeffs[2]
        
        # Normalized the normal
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        return normal, d
    
    def _compute_grid_step(self):
        """
        Automatically computes the grid step as the minimum distance between points in 2D.
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(self.pointsGrid[:, :2])     
        # distance to the second colsest point 
        distances, _ = tree.query(self.pointsGrid[:, :2], k=2)
        step = np.min(distances[:, 1])

        if step < 1e-6:
            raise ValueError(
                f"Grid step is {step:.6f} m — too small or zero.\n"
                f"Check your grid for duplicate or near-duplicate points."
            )

        print(f"Grid step auto-computed: {step:.4f} m")
        return step
    
    def getGridStep(self):
        return self.step