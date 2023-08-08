import os 
import numpy as np 
from utils import *

# generate initial configurations
initial_path = 'initial_samples'
# generate_initials(initial_path)

# convert to input files for packing algorithm
r = 0.06
root_packing = 'initial_samples_for_packing'
# convert4packing(r, root_packing, initial_path)

# run the packing algorithm
packed_path = 'samples_packed'
# run_packing(root_packing, packed_path)

# generate mesh using gmsh
mesh_path = 'samples_mesh' # this is where the final mesh are stored
generate_gmsh_multi(mesh_path, packed_path)


