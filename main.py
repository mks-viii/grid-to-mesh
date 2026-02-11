import os
from config import *
from out_name import get_output_name
from grid import Grid
from mesh import Mesh
#from projector_parallel import Projector
from projector import Projector

lamiamesh = os.path.join(script_dir, inputFolder, meshName)
imieipunti = os.path.join(script_dir, inputFolder, pointName)
output_name = get_output_name(meshName)
ilmiooutput = os.path.join(script_dir, outputFolder, output_name)

grid = Grid(imieipunti)
mesh = Mesh(lamiamesh)
grid=grid.getGrid()
vertices=mesh.getVertices()
faces=mesh.getFaces()
centroids=mesh.getCentroids()
projected = Projector(
        grid, 
        vertices, 
        faces, 
        tolerance,       
        buffer=buffer, 
        radius=radius, 
        normal=normalGrid,
        centroids=centroids,
        output=ilmiooutput)
projected_points, triangle_indices, best_distance = projected.project_ort_points_on_mesh()
"""""
if __name__=='__main__':
    projected = Projector(
        grid, 
        vertices, 
        faces, 
        tolerance,       
        buffer=buffer, 
        radius=radius, 
        normal=normalGrid,
        centroids=centroids,
        output=ilmiooutput,
        core=core
        )
        
    projected_points, triangle_indices, best_distance = projected.project_ort_points_on_mesh()
"""""    