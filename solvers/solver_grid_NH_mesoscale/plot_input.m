clc; clear;
load ../../data/mesoscale_grid/input_s128.mat
f = figure;
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_bulk,'FaceColor','flat','EdgeColor', 'none');
axis equal off
hcb = colorbar;
hcb.Label.String = 'Bulk modulus [MPa]';
hcb.FontSize = 26;
f.Position = [400 400 900 600];

%%
f = figure;
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_shear,'FaceColor','flat','EdgeColor', 'none');
axis equal off
hcb = colorbar;
hcb.FontSize = 26;
hcb.Label.String = 'Shear modulus [MPa]';
f.Position = [500 500 800 600];
