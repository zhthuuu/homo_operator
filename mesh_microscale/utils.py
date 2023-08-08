import numpy as np
import gmsh 
import math 
import sys
import os

# generate the stochastic initial configurations 
# the random seed is set as time in millisecond resolution
def generate_initials(initial_path):
	if not os.path.exists(initial_path):
		os.mkdir(initial_path)
	N = 5 # number os generated samples
	for i in range(N):
		os.system('source/run_initial_configs source/input_initial')
		os.rename('write.dat', initial_path + '/initial_sample'+str(i+1)+'.dat')

# convert the input files for packing algorithm of TJ
def convert4packing(r, root_packing, initial_path):
	if not os.path.exists(root_packing):
		os.mkdir(root_packing)
	files = os.listdir(initial_path)
	print('write files to {}'.format(root_packing))
	for i, file in enumerate(files):
		file_path = initial_path + '/' + file
		circles = read_initial_sample(file_path)
		filename = root_packing + '/iconfig_' + str(i+1) + '.dat'
		write_sample4packing(r, circles, filename)
	print('Done')


# run the packing algorihtm of TJ
def run_packing(root_packing, packed_path):
	files = os.listdir(root_packing)
	if not os.path.exists(packed_path):
		os.mkdir(packed_path)
	for i, file in enumerate(files):
		os.system('source/tj_2d {} source/parameters_2d.txt output'.format(
			'initial_samples_for_packing/'+file))
		os.system('mv output.dat {}/sample{:d}.dat'.format(packed_path, i+1))


# generate mesh using gmsh
def generate_gmsh_multi(mesh_path, packed_path):
	if not os.path.exists(mesh_path):
		os.mkdir(mesh_path)
	files = os.listdir(packed_path)
	for i, file in enumerate(files):
		file_packed = packed_path + '/' + file
		circles, r = read_circles(file_packed)
		file_gmsh = mesh_path + '/sample' + str(i+1) + '.m'
		generate_gmsh(file_gmsh, r, circles)
	print('Done')



# read circle coordinates in the initial configuration file
def read_initial_sample(file_initial):
	with open(file_initial, 'r') as f:
		num = 0
		num_circles = 0
		while True:
			num = num+1
			line = f.readline()
			if num == 3:
				num_circles = int(line)
				circles = np.zeros([num_circles, 2])
			if num > num_circles+6:
				break
			if num > 6:
				circles[num-7,:] = np.fromstring(line, sep=' ')
	return circles


# write the input files for the TJ packing algorithm 
def write_sample4packing(r, circles, filename):
	N = circles.shape[0]
	with open(filename, 'w') as f:
		f.write('2\n')
		f.write('{:d}\n'.format(N))
		f.write('{:.4f}\n'.format(r))
		f.write('1 0\n0 1\n')
		for i in range(N):
			f.write('{:.4f} {:.4f}\n'.format(circles[i, 0], circles[i, 1]))


# read circles from the packed contact-free configurations
def read_circles(file):
	with open(file, 'r') as f:
		num = 0
		lattice = np.zeros([2,2])
		N_circles = 0
		r = 0
		circles = 0
		line = f.readline()
		while True:
			num = num+1
			if num == 2:
				N_circles = int(line)
				circles = np.zeros([N_circles, 2])
			if num == 3:
				r = float(line) # radius
			if num == 4:
				lattice[0, :] = np.fromstring(line, sep=' ')
			if num == 5:
				lattice[1, :] = np.fromstring(line, sep=' ')
			if num > N_circles + 5:
				break
			if num > 5:
				circles[num-6, :] = np.fromstring(line, sep=' ')
			line = f.readline()
	circles = np.matmul(circles, np.linalg.inv(lattice))
	return circles, r

# use gmsh python scripts to generate mesh files
# r: radius, circles: circle center coordinates, file: output file (.m)
def generate_gmsh(file, r, circles):
	gmsh.initialize(readConfigFiles=False)
	gmsh.model.add('tq')
	gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
	N = circles.shape[0]
	holes = []
	for i in range(N):
		circle_id = i+5
		curve_loop_id = i+2
		surface_id = i+2
		x = circles[i, 0]
		y = circles[i, 1]
		gmsh.model.occ.addCircle(x, y, 0, r, tag=circle_id)
		gmsh.model.occ.addCurveLoop([circle_id], curve_loop_id)
		gmsh.model.occ.addPlaneSurface([curve_loop_id], surface_id)
		holes.append((2, surface_id)) # here 2 means 2 dimensions
	
	ov, ovv = gmsh.model.occ.fragment([(2,1)], holes)
	gmsh.model.occ.synchronize()

	# # print the segmented pices
	# print("fragment produced volumes:")
	# for e in ov:
	# 	print(e)

	# # ovv contains the parent-child relationships for all the input entities:
	# print("before/after fragment relations:")
	# for e in zip([(2, 1)] + holes, ovv):
	# 	print("parent " + str(e[0]) + " -> child " + str(e[1]))

	# assign physical group id
	matrix_id = ovv[0][0][1]
	circle_id = []
	for e in ov:
		if e[1] == matrix_id:
			continue
		else:
			circle_id.append(e[1])
	gmsh.model.addPhysicalGroup(2, [matrix_id], 0) # matrix
	gmsh.model.addPhysicalGroup(2, circle_id, 1) # inclusions

	# generate mesh
	# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
	gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)
	gmsh.model.mesh.generate()
	gmsh.write(file)
	# show the UI system
	# if '-nopopup' not in sys.argv:
	# 	gmsh.fltk.run()
	gmsh.finalize()


#  convert the circle coordinates into gmsh geo file 
# (not used here, python scripts can be used instead)
def write_geo(file, r, circles):
	with open(file, 'w') as f:
		# set kernel
		f.write('SetFactory("OpenCASCADE");\n')
		# create square (matrix)
		f.write('Rectangle(1) = {0, 0, 0, 1, 1, 0};\n//+\n')
		# create circles (inclusions)
		N = circles.shape[0]
		for i in range(N):
			circle_id = i+5
			curve_loop_id = i+2
			surface_id = i+2
			f.write('Circle({:d}) = {{ {:.4f}, {:.4f}, 0, {:.4f}, 0, 2*Pi }};\n'.format(
            		circle_id, circles[i, 0], circles[i, 1], r))
			f.write('Curve Loop({:d}) = {{ {:d} }};\n'.format(curve_loop_id, circle_id))
			f.write('Plane Surface({:d}) = {{ {:d} }};\n//+\n'.format(surface_id, curve_loop_id))
		# boolean operation
		f.write('BooleanFragments{{ Surface{{ 2:{:d} }}; Surface{{ 1 }}; Delete; }}{{ }}\n//+\n'.format(
		        surface_id))
		# # create physics group for matrix
		# f.write('Physical Surface("matrix", 8) = {{ {:d} }};\n//+\n'.format(N+2))
		# # create physics group for inclusions
		# f.write('Physical Surface("inclusion", 9) = {{ 2:{:d} }};\n//+\n'.format(N+1))






