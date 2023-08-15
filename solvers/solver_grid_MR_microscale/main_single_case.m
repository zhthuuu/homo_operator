%% EXAMPLE FOR THE MULTISCALE CLASS
%% HYPERELASTIC MICROSTRUCTURE WITH MR MATERIALS
clc
clear
close all
x1 = 10; x2 = 2; 
% x1=100; x2=10;
%% PROPERTIES
% Matrix properties
alpham = 0.5*328.1250;   % Alpha in MPa
betam = 10;              % Beta in MPa
km = 1750;               % Bulk modulus
um = 2*alpham;           % Shear modulus
lambdam = km - (2/3)*um; % Lambda
sm = lambdam - 4*betam;  % s1 = Lambda - 4*Beta
% Inclusion properties
alphaf = x1*alpham;
betaf = x2*betam;
kf = x1*km;
uf = 2*alphaf; 
lambdaf = kf - (2/3)*uf;
sf = lambdaf - 4*betaf;
%
PROP_CPP = [alpham, betam, sm;
            alphaf, betaf, sf];

%% Load and plot the mesh
load ../data/phase_N1000_s64.mat
s = 64;
sample = 100;
PHASES = reshape(phase(sample, :, :), s, s);
PHASES = reshape(flip(PHASES)', [],1)+1;
NE = s * s; ND = s*4; % number of fixed nodes
N = (s+1)*(s+1);
[p, t, FIXEDNODES] = generate_mesh(s);
FIXEDNODES = double(FIXEDNODES);
Ae = 1 / (s*s);
% a = reshape(phase(sample, :, :), s, s);
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
C1 = 1.5;
C2 = 1.0;
C3 = 0.0;
CauchyGreen = [C1, C3; 
               C3, C2];
F = chol(CauchyGreen);
H = F - eye(2);
tic();
[WEFF, DISP] = NL2DSOLVER(FIXEDNODES, p, PROP_CPP, NE, GDOF, ND, H, PHASES, ...
s, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);
toc();
disp(['weff = ', num2str(WEFF)]);

%% plot
UX = DISP(1:2:end); UY = DISP(2:2:end);
UX = flip(reshape(UX, s+1, s+1)');
UY = flip(reshape(UY, s+1, s+1)');
a = flip(reshape(PHASES, s, s)');
h = figure();
subplot(1,3,1);
[X, Y] = meshgrid(linspace(0,1,s), linspace(0,1,s));
mesh(X, Y, a, 'FaceColor','flat'); view(2);
subplot(1,3,2);
[X, Y] = meshgrid(linspace(0,1,s+1), linspace(0,1,s+1));
mesh(X, Y, UX, 'FaceColor','flat'); view(2);
colorbar; colormap('jet');
subplot(1,3,3);
mesh(X, Y, UY, 'FaceColor','flat'); view(2);
colorbar; colormap('jet');
set(h, 'Position', [100,100,1200,300]);



