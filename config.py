import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

inputFolder = 'input_gor'
meshName = "30.ply"
pointName = '100PT_GOR_OPT2.xyz'
outputFolder = 'output_gor'


# Tolerance around the triangle
# Due to floating-point precision limitations, a point that should be exactly on the triangle edge
# might appear slightly inside or outside
# Points very close to the edge are considered inside the triangle
# 4e-1 for high-resolution mesh
# 10 for low-resolution mesh
# radius is the search radius near a point. Consider the grid-to-terrain distance when setting the radius
tolerance = 0.01
buffer = 1.0
radius = 20.0
core=6

#Set the normal to the grid
normalGrid = np.array([0.0, 0.0, 1.0])