import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

inputFolder = 'input_gor'
meshName = "40.ply"
pointName = '36PT_GOR.xyz'
outputFolder = 'output_gor'



#Tolerance attorno al triangolo
#A causa delle limitazioni della precisione dei numeri in virgola mobile, un punto che dovrebbe essere esattamente sul bordo di un triangolo 
#potrebbe risultare leggermente all'interno o all'esterno
#Punti che sono molto vicini al bordo vengono considerati all'interno del triangolo
#4e-1 per mesh alta risoluzione
#10 per mesh bassa risoluzione
tolerance = 0.01
buffer = 1.0
radius = 5.0
core=6

#imposta la normale alla griglia che 
normalGrid = np.array([0.0, 0.0, 1.0])