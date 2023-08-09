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

%% mesh info
load phase_N1000_s64.mat
phase = int32(phase+1);
s = 64;
NE = s * s; ND = s*4; % number of fixed nodes
N = (s+1)*(s+1);
[p, t, FIXEDNODES] = generate_mesh(s);
FIXEDNODES = double(FIXEDNODES);
Ae = 1 / (s*s);

%% FE COMPUTATIONS
% [sb, sc, Ae] = shape(p, t, NE);
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
C1 = 1.5;
C2 = ones(length(C1),1);
C3 = zeros(length(C1),1);
CauchyGreen = [C1, C3; 
               C3, C2];
F = chol(CauchyGreen);
H = F - eye(2);

%% NONLINEAR FEM
DISP = zeros(1000, s+1, s+1, 2);
Weff = zeros(1000, 1);
for id = 1:1000
    PHASES = reshape(phase(id, :, :), s, s);
    PHASES = reshape(flip(PHASES)', [],1);
    [Weff_id, DISP_id] = NL2DSOLVER(FIXEDNODES, p, PROP_CPP, NE, GDOF, ND, H, PHASES, ...
    s, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);
    UX = DISP_id(1:2:end); UY = DISP_id(2:2:end);
    UX = flip(reshape(UX, s+1, s+1)');
    UY = flip(reshape(UY, s+1, s+1)');
    Weff(id) = Weff_id;
    DISP(id,:,:,1) = UX;
    DISP(id,:,:,2) = UY;
    if rem(id, 100) == 0
        disp([num2str(id), '/1000 cases finished.']);
    end
end

% save ../data/griddisp_s64_N1000 DISP
% save ../data/Weff_s64_N1000 Weff
%% plot
% global DISPTD
% UX = DISPTD(1:2:end); UY = DISPTD(2:2:end);
UX = flip(reshape(UX, s+1, s+1)');
UY = flip(reshape(UY, s+1, s+1)');
a = flip(reshape(PHASES, s, s)');
figure();
subplot(1,2,1);
[X, Y] = meshgrid(linspace(0,1,s), linspace(0,1,s));
mesh(X, Y, a, 'FaceColor','flat'); view(2);
subplot(1,2,2);
[X, Y] = meshgrid(linspace(0,1,s+1), linspace(0,1,s+1));
mesh(X, Y, UX, 'FaceColor','flat'); view(2);
colorbar; colormap('jet');

