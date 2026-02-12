import os
from config import *
from out_name import get_output_name
from grid import Grid
from mesh import Mesh
#from projector_parallel import Projector
from projector import Projector
from import_config import Config
import time

# Load the configuration file
config = Config()  # search atomatically the config.json

# Print the configuration
config.print_config()
time.sleep(2)  # Pausa dopo caricamento

# Check the path
if not config.validate_paths():
    exit(1)

time.sleep(2)  # Pause to read the message

# Create the output folder
config.create_output_folder()

# Variables defined htough config.py
#print(f"Carico mesh da: {config.mesh_path}")
#print(f"Carico punti da: {config.point_path}")
#print(f"Usando {config.cores} cores")
#print(f"Tolleranza: {config.tolerance}")
#print(f"Normale griglia: {config.normal_grid}")

""""
lamiamesh = os.path.join(script_dir, inputFolder, meshName)
imieipunti = os.path.join(script_dir, inputFolder, pointName)
output_name = get_output_name(meshName)
ilmiooutput = os.path.join(script_dir, outputFolder, output_name)
"""

lamiamesh = config.mesh_path
imieipunti = config.point_path
output_name = get_output_name(meshName)
ilmiooutput = os.path.join(script_dir, config.output_path, output_name)

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
        output=ilmiooutput,
        core=core)

iprojected_points, triangle_indices, best_distance = projected.project_ort_points_on_mesh()

""""
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
""" 