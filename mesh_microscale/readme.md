# How to generate the random packing of a two-dimensional two-phase composite material with matrix and circle inclusions:

### currently all the codes are run on Macbook M1 chip

1. generate random initial packing configuration using code in folder sphere 
(see reference in M. Skoge, A. Donev, F. H. Stillinger and S. Torquato, Packing Hyperspheres in High-Dimensional Euclidean Spaces, Physical Review E 74, 041127 (2006).)
- script: ./run input
- Note: currently the radius of inclusion is set as 0.06, number of circles is 20, such that the volume fraction is 0.23
- to generate excutive file:  'cd src_initial, g++ -o run neighbor.C spheres.C box.C sphere.C event.C heap.C read_input.C'

2. generate the contactless packing configuration using the TJ algorithm in folder TJ, modify the input file
script: ./tj_2d iconfig_2d.dat parameters_2d.txt output
Note: two parameters in the parameters_2d.txt file are important to influence the final packing performance: 
- #influence_sphere 0.1  // the larger the better
- #max_iterations 200  // cannot be to large, if so the final packing will be the same as in regular lines
- to generate excutive file: change parameter DIM in src_tj/makefile, then run 'cd src_tj, make'

