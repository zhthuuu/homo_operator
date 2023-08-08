% clc; clear;
% load data/phase_N1000_s420.mat
% load data/s421_sample100.mat
load ../data/dataset_train_s64_N600.mat
s = 64; sample = 100;
PHASES = reshape(a(sample,:,:), s, s);
u_grid = reshape(u(sample,:,:,:), s+1,s+1,2);

%% properties
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

%% 
[Weff] = calc_Weff_gaussian_quadrature(PHASES, u_grid, PROP_CPP);
% tmp = reshape(tmp, s, s);
% [X, Y] = meshgrid(linspace(0,1,s), linspace(0,1,s));
% figure(1);
% mesh(X, Y, tmp, 'FaceColor','flat'); view(2); colorbar; colormap('jet');

