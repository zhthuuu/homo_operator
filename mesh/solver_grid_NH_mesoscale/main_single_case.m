%% EXAMPLE FOR THE MULTISCALE CLASS: solve Neo-Hookean material with grid mesh
%% HYPERELASTIC MICROSTRUCTURE WITH MR MATERIALS
clc
clear
close all

%% Load and plot the mesh
load ../data/mesoscale_grid/input_s128.mat
f = figure;
subplot(1,2,1)
title('bulk moduli');
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_bulk,'FaceColor','flat','EdgeColor', 'white');
colorbar;
subplot(1,2,2)
title('shear moduli');
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_shear,'FaceColor','flat','EdgeColor', 'none');
colorbar;
f.Position = [200 200 1100 400];

%% preprocessing
s = 128;
BULK = mean(eta_bulk(t), 2);
SHEAR = mean(eta_shear(t), 2);
NE = s * s; ND = s*4; % number of fixed nodes
N = (s+1)*(s+1);
Ae = 1 / (s*s);

%% FE COMPUTATIONS
GDOF = 2*N;
DOFe = zeros(8, NE); 
ISPARSE = zeros(64*NE, 1); 
JSPARSE = zeros(64*NE, 1);
KSPARSE = zeros(8*NE, 1);
% Preprocessing for sparse assembly
for e = 1:NE
    NODES = t(e, :);
    DOFe(:, e) = [2*NODES(1)-1, 2*NODES(1), 2*NODES(2)-1, 2*NODES(2), ...
                    2*NODES(3)-1, 2*NODES(3),2*NODES(4)-1, 2*NODES(4)]';        
    ISPARSE(1+(e-1)*64:e*64) = repmat(DOFe(:, e), 8, 1);
    JSPARSE(1+(e-1)*64:e*64) = [repmat(DOFe(1, e), 8, 1); ...
                            repmat(DOFe(2, e), 8, 1); ...
                            repmat(DOFe(3, e), 8, 1); ...
                            repmat(DOFe(4, e), 8, 1); ...
                            repmat(DOFe(5, e), 8, 1); ...
                            repmat(DOFe(6, e), 8, 1); ...
                            repmat(DOFe(7, e), 8, 1); ...
                            repmat(DOFe(8, e), 8, 1)];
    KSPARSE(1+(e-1)*8:e*8) = DOFe(:, e);
end

DOFe = int32(DOFe); % if not convert, matlab will crash for c++ code

%% NONLINEAR FEM
% BIAXIAL TENSION
% C1 = linspace(0.9, 1.5, 10);
% C2 = linspace(0.9, 1.5, 10);
% [MATC1, MATC2] = meshgrid(C1,C2);
% C1 = MATC1(:);
% C2 = MATC2(:);
% C3 = zeros(length(C1),1);
% UNIAXIAL TENSION
C1 = 1.5;
C2 = 1.0;
C3 = 0.0;
CauchyGreen = [C1, C3; 
               C3, C2];
F = chol(CauchyGreen);
H = F - eye(2);
tic();
[WEFF, DISP] = NL2DSOLVER(FIXEDNODES, p, BULK, SHEAR, NE, GDOF, ND, H, ...
s, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);
toc();
disp(['weff = ', num2str(WEFF)]);

%% plot
UX = DISP(1:2:end); UY = DISP(2:2:end);
f = figure;
subplot(1,2,1)
title('UX');
patch('Faces',t,'Vertices',p,'FaceVertexCData',UX,'FaceColor','flat','EdgeColor', 'none');
colorbar;
subplot(1,2,2)
title('UY');
patch('Faces',t,'Vertices',p,'FaceVertexCData',UY,'FaceColor','flat','EdgeColor', 'none');
colorbar;
f.Position = [200 200 1100 400];

%% plot
UX = DISP(1:2:end); UY = DISP(2:2:end);
UX_grid = reshape(UX, s+1, s+1);
UY_grid = reshape(UY, s+1, s+1)';
bulk_grid = reshape(eta_bulk, s+1, s+1)';
BULK_grid = reshape(BULK, s, s)';
h = figure();
subplot(2,2,1);
title('bulk (pt)')
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_bulk,'FaceColor','flat','EdgeColor', 'none');
colorbar();
subplot(2,2,2);
title('UY (pt)');
patch('Faces',t,'Vertices',p,'FaceVertexCData',UY,'FaceColor','flat','EdgeColor', 'none');
colorbar();
subplot(2,2,3);
title('bulk (grid)');
[X, Y] = meshgrid(linspace(0,1,s), linspace(0,1,s));
mesh(X, Y, BULK_grid, 'FaceColor','flat'); view(2);
colorbar;
subplot(2,2,4);
title('UY (grid)');
[X, Y] = meshgrid(linspace(0,1,s+1), linspace(0,1,s+1));
mesh(X, Y, UY_grid, 'FaceColor','flat'); view(2);
colorbar;
% set(h, 'Position', [100,100,1200,300]);



