clc; clear;
load data/trimesh_N1000.mat
%%
id = 3;
p = trimesh(id).p;
t = trimesh(id).t;
PHASES = trimesh(id).PHASES;
patch('Faces', t, 'Vertices', p, 'FaceColor', 'flat', ...
      'FaceVertexCData', PHASES, 'EdgeColor','none');
axis off equal