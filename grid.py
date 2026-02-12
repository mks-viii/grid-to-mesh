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