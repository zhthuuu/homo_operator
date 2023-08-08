clc; clear;
load data/trimesh_N1000.mat

%% PROPERTIES
% Matrix properties
alpham = 0.5*328.1250;   % Alpha in MPa
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

% Cauchy Green matrix
C1 = 1.5; C2 = 1; C3 = 0;

%% load mesh
for i = 1:1000
    if rem(i, 50) == 0
        disp([num2str(i), '/1000 finished']);
    end
    try
        [WEFF, UX, UY] = run_solver(PROP_CPP, trimesh(i), C1, C2, C3);
    catch
        warning(['case ', num2str(i), ' cannot be calculated!']);
        continue;
    end
    tridisp(i).WEFF = WEFF;
    tridisp(i).UX = UX;
    tridisp(i).UY = UY;
end

%%
% note: #496 mesh cannot be computed
% trimesh(496) = [];
% tridisp(496) = [];
% save data/tridisp_N999.mat tridisp
load data/trimesh_N1000.mat
trimesh(496) = [];
load data/tridisp_N999.mat
s = 61;
x = linspace(0,1,s); y = x;
n_sample = 999;
U = zeros(s, s, n_sample, 2);
for i = 1:999
    UX = tridisp(i).UX;
    UY = tridisp(i).UY;
    p = trimesh(i).p;
    t = trimesh(i).t;
    U(:,:,i, 1) = tri2grid(p',t',UX,x,y);
    U(:,:,i, 2) = tri2grid(p',t',UY,x,y);
end
% save data/griddisp_N999.mat U

