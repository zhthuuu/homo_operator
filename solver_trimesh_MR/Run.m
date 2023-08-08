%% EXAMPLE FOR THE MULTISCALE CLASS
%% HYPERELASTIC MICROSTRUCTURE WITH MR MATERIALS
clc
clear
close all
%% PROPERTIES
% Matrix properties
alpham = 0.5*328.1250;   % Alpha in MPa
% alpham = 1;
betam = 10;              % Beta in MPa
km = 1750;               % Bulk modulus
um = 2*alpham;           % Shear modulus
lambdam = km - (2/3)*um; % Lambda
sm = lambdam - 4*betam;  % s1 = Lambda - 4*Beta
% Inclusion properties
alphaf = 100*alpham;
betaf = 10*betam;
kf = 100*km;
uf = 2*alphaf; 
lambdaf = kf - (2/3)*uf;
sf = lambdaf - 4*betaf;
%
PROP_CPP = [alpham, betam, sm;
            alphaf, betaf, sf];

%% Load and plot the mesh
% load('data/MESH_P1.mat', 'p', 't', 'PHASES', 'NE', 'ND', 'N', 'FIXEDNODES');
load data/trimesh_N1000.mat
mesh = trimesh(100);
p = mesh.p; p = p(:, 1:2);
t = mesh.t; PHASES = mesh.PHASES+1; NE = mesh.NE; ND = mesh.ND; N = mesh.N;
FIXEDNODES = double(mesh.FIXEDNODES);

p = p(:, 1:2);
figure;
hold on
triplot(t(PHASES==1,:), p(:,1), p(:,2), '-k', 'LineWidth', 0.5);
triplot(t(PHASES==2,:), p(:,1), p(:,2), '-r', 'LineWidth', 0.5);
axis equal
hold off
set(gca,'visible','off')

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
PHASES = int32(PHASES);

%% NONLINEAR FEM
% BIAXIAL TENSION
% C1 = linspace(0.9, 1.5, 10);
% C2 = linspace(0.9, 1.5, 10);
% [MATC1, MATC2] = meshgrid(C1,C2);
% C1 = MATC1(:);
% C2 = MATC2(:);
% C3 = zeros(length(C1),1);
% UNIAXIAL TENSION
% C1 = linspace(1, 1.5, 10);
C1 = 1.5;
C2 = ones(length(C1),1);
C3 = zeros(length(C1),1);
WEFF = zeros(length(C1),1);
tic();
for i = 1 : length(C1)
    CauchyGreen = [C1(i), C3(i); 
                   C3(i), C2(i)];
    F = chol(CauchyGreen);
    H = F - eye(2);
    tic();
    WEFF(i) = NL2DSOLVER(FIXEDNODES, p, PROP_CPP, NE, GDOF, ND, H, PHASES, ...
    sb, sc, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);
    toc();
end
toc();

global DISPTD
UX = DISPTD(1:2:end); UY = DISPTD(2:2:end);

PDEF = p + [UX, UY];

figure;
hold on
triplot(t(PHASES==1,:), PDEF(:,1), PDEF(:,2), '-k', 'LineWidth', 0.5);
triplot(t(PHASES==2,:), PDEF(:,1), PDEF(:,2), '-r', 'LineWidth', 0.5);
axis equal
hold off
set(gca,'visible','off')


%% tri2grid
s = 61;
x = linspace(0,1,s); y = x;
ux_interp = tri2grid(p',t',UX,x,y);
uy_interp = tri2grid(p',t',UY,x,y);
subplot(1,2,1);
surface(x, y, ux_interp,'EdgeColor','none'); title('interpolation (ux)');
colorbar; colormap('jet');
subplot(1,2,2);
surface(x, y, uy_interp,'EdgeColor','none'); title('interpolation (uy)');
colorbar; colormap('jet');
% subplot(2,2,3);
% patch('Faces', t, 'Vertices', p, 'FaceColor', 'flat', ...
%       'FaceVertexCData', UX, 'EdgeColor','none');
% title('ground truth (ux)');
% xlim([0,1]); ylim([0,1]);
% subplot(2,2,4);
% patch('Faces', t, 'Vertices', p, 'FaceColor', 'flat', ...
%       'FaceVertexCData', UY, 'EdgeColor','none');
% title('ground truth (uy)');
% xlim([0,1]); ylim([0,1]);

%% save data
u_interp = cat(3, ux_interp, uy_interp);
save data/s61_sample100 WEFF u_interp

%%
close all
WEFF = mex_EFFSEF(DISPTD, NE, PROP_CPP, PHASES, sb, sc, Ae, DOFe);




