% Generate stochastic bulk and shear moduli on a plane square domain
% @Hao Zhang, 8/5/2023

clc; clear;
addpath ./FEM_toolbox/model
addpath ./FEM_toolbox/2d
seed = RandStream('mt19937ar', 'Seed', 1); RandStream.setGlobalStream(seed); %set seed
file = 'FEM_toolbox/geometry/square/square.stl'; % input brain tissue boundary mesh
dir = 'data/'; 
if ~exist(dir, 'dir')
    mkdir(dir); % generated mesh folder
end

l = 0.1; % the coorrelation length
N = 10; % number of generated meshes
delta = 0.57; % coefficient for beta distribution
rho = 0.9; % coefficient of correlation between stochastic bulk and shear modulio

%% preprocessing
kappa = 1/l;  
mu = 0; sigma = 1;  % the mean and standard deviation of gaussian field
nu = 1; d = 2; % the parameters in the SPDE
normconst = sigma^2*(4*pi)^(d/2)*gamma(nu+d/2)/gamma(nu);
normconst = normconst*l^(d-4); % alpha multiplied to the white noise vector
msh = stlread(file);
P = msh.Points;
t = msh.ConnectivityList;
numNodes = size(P, 1);
disp(['Done reading mesh ', file]);

%% solve SPDE
[R, flag, transP] = get_precision_mat_isotropy(kappa, P, t, normconst);  % the important matrix Q R
disp("Done calculating precision matrix Q");

% modify multiply geometries
g_bulk = normrnd(mu,sigma,numNodes,N); % the white noise vector following Gaussian distribution
eta_bulk = transP * (R \ g_bulk); % the random field
g_shear = normrnd(mu,sigma,numNodes,N); % the white noise vector following Gaussian distribution
eta_shear = transP * (R \ g_shear); % the random field

eta_beta_bulk = convert_beta(height, delta, eta_bulk);
eta_beta_shear = convert_beta(height, delta, eta_shear);

%% visualization
patch('Faces',t,'Vertices',P,'FaceVertexCData',eta_beta(:,1),'FaceColor','interp','EdgeColor', 'none');
colorbar;

