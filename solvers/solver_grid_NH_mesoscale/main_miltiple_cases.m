%% EXAMPLE FOR THE MULTISCALE CLASS
%% HYPERELASTIC MICROSTRUCTURE WITH MR MATERIALS
clc
clear
close all

%% Load and plot the mesh
load ../../data/mesoscale_grid/input_N5000_s128.mat
f = figure;
subplot(1,2,1)
title('bulk moduli');
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_bulk(:,1),'FaceColor','flat','EdgeColor', 'white');
colorbar;
subplot(1,2,2)
title('shear moduli');
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_shear(:,1),'FaceColor','flat','EdgeColor', 'none');
colorbar;
f.Position = [200 200 1100 400];

%% preprocessing
s = 128;
BULK = zeros(size(t,1), size(eta_bulk,2));
SHEAR = zeros(size(t,1), size(eta_bulk,2));
for i = 1:size(eta_bulk, 2)
    bulk_tmp = eta_bulk(:,i);
    shear_tmp = eta_shear(:,i);
    BULK(:,i) = mean(bulk_tmp(t), 2);
    SHEAR(:,i) = mean(shear_tmp(t), 2);
end
NE = s * s; ND = s*4; % number of fixed nodes
N = (s+1)*(s+1);
Ae = 1 / (s*s);

%% FE COMPUTATIONS
GDOF = 2*N;
%
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

% deformation info
C1 = 1.;
C2 = 1.;
C3 = 0.2;
CauchyGreen = [C1, C3; 
               C3, C2];
F = chol(CauchyGreen);
H = F - eye(2);

%% NONLINEAR FEM
disp('Now running the solver...');
N = 50;
DISP = zeros(N, (s+1)*(s+1)*2);
Weff = zeros(N, 1);
BULK = BULK(:,1:N);
SHEAR = SHEAR(:, 1:N);
tic
for id = 1:N
    BULK_id = BULK(:, id);
    SHEAR_id = SHEAR(:,id);
    [Weff(id), DISP(id,:)] = NL2DSOLVER(FIXEDNODES, p, BULK_id, SHEAR_id, NE, GDOF, ND, H, ...
    s, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);
end
toc

disp('Now transform DISP...');
DISP_grid = zeros(N, s+1, s+1, 2);
for id = 1:N
    UX = DISP(id, 1:2:end); UY = DISP(id, 2:2:end);
    UX = reshape(UX, s+1, s+1)';
    UY = reshape(UY, s+1, s+1)';
    DISP_grid(id,:,:,1) = UX;
    DISP_grid(id,:,:,2) = UY;
end


%% save data
disp('Now saving dataset...')
save ../../data/mesoscale_grid/shear/output_N50_s128 DISP_grid Weff BULK SHEAR
disp('Finished!')





