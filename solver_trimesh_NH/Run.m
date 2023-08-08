%% EXAMPLE FOR THE MULTISCALE CLASS
%% HYPERELASTIC MICROSTRUCTURE WITH MR MATERIALS
clc
clear
close all
%% Load and plot the mesh
load('../generate_random_field/data/mesh.mat', 'p', 't', 'bulk', 'shear','FIXEDNODES');
p = p(:, 1:2);
NE = length(t);
N = length(p);
ND = length(FIXEDNODES);
hold on
triplot(t, p(:,1), p(:,2));
scatter(p(FIXEDNODES, 1), p(FIXEDNODES,2), 'red');
hold off

%% FE COMPUTATIONS
[sb, sc, Ae] = shape(p, t, NE);
GDOF = 2*N;
%
DOFe = zeros(6, NE); 
ISPARSE = zeros(36*NE, 1); 
JSPARSE = zeros(36*NE, 1);
KSPARSE = zeros(6*NE, 1);
% Preprocessing for sparse assembly
for e = 1:NE
    NODES = t(e, :);
    DOFe(:, e) = [2*NODES(1)-1, 2*NODES(1), 2*NODES(2)-1, 2*NODES(2), ...
                    2*NODES(3)-1, 2*NODES(3)]';        
    ISPARSE(1+(e-1)*36:e*36) = repmat(DOFe(:, e), 6, 1);
    JSPARSE(1+(e-1)*36:e*36) = [repmat(DOFe(1, e), 6, 1); ...
                            repmat(DOFe(2, e), 6, 1); ...
                            repmat(DOFe(3, e), 6, 1); ...
                            repmat(DOFe(4, e), 6, 1); ...
                            repmat(DOFe(5, e), 6, 1); ...
                            repmat(DOFe(6, e), 6, 1)];
    KSPARSE(1+(e-1)*6:e*6) = DOFe(:, e);
end
%
DOFe = int32(DOFe);

%% NONLINEAR FEM
NUM = 2;
% BIAXIAL TENSION
C1 = linspace(0.9, 1.5, NUM);
C2 = linspace(0.9, 1.5, NUM);
[MATC1, MATC2] = meshgrid(C1,C2);
C1 = MATC1(:);
C2 = MATC2(:);
C3 = zeros(length(C1),1);
% UNIAXIAL TENSION
% C1 = linspace(1, 1.5, 2);
% C2 = ones(length(C1),1);
% C3 = zeros(length(C1),1);
WEFF = zeros(length(C1),1);
for i = 1 : length(C1)
    CauchyGreen = [C1(i), C3(i); 
                   C3(i), C2(i)];
    F = chol(CauchyGreen);
    H = F - eye(2);
    tic();
    WEFF(i) = NL2DSOLVER(FIXEDNODES, p, NE, GDOF, ND, H, bulk, shear, ...
    sb, sc, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);
    toc();
end

global DISPTD

UX = DISPTD(1:2:end); UY = DISPTD(2:2:end);

PDEF = p + [UX, UY];

%% visualization of displacement
f = figure;
subplot(1,2,1)
patch('Faces',t,'Vertices',p,'FaceVertexCData',UX,'FaceColor','interp','EdgeColor', 'none');
colorbar;
subplot(1,2,2)
patch('Faces',t,'Vertices',p,'FaceVertexCData',UY,'FaceColor','interp','EdgeColor', 'none');
colorbar;
f.Position = [200 200 900 400];








