% demo to generate isoptopic non-Gaussian random field on 2d manifold or 2d flat
% surface. Here we use a flat square.
clc; clear;
addpath ../FEM_toolbox/model
addpath ../FEM_toolbox/2d_grid
seed = RandStream('mt19937ar', 'Seed', 1); RandStream.setGlobalStream(seed); %set seed

%% preprocessing
s = 128; % number of elements per edge
l = 0.1; % the coorrelation length
N = 10; % number of samples
kappa = 1/l;  
mu = 0; sigma = 1;  % the mean and standard deviation of gaussian field
nu = 1; d = 2; % the parameters in the SPDE
normconst = sigma^2*(4*pi)^(d/2)*gamma(nu+d/2)/gamma(nu);
normconst = normconst*l^(d-4); % alpha multiplied to the white noise vector
delta = 0.3;
rho = 0.9;


%% material properties
bulk_moduli = 1.75e4; % bulk modulus in MPa
shear_moduli = 3281.25; % shear modulus in MPa
bulk_moduli_height = bulk_moduli * 0.1;
shear_moduli_height = shear_moduli*0.1;

%% generate grid mesh
[p, t, FIXEDNODES] = generate_grid_mesh(s);
hold on
patch('Faces',t,'Vertices',p,'EdgeColor', '#4DBEEE');
scatter(p(FIXEDNODES, 1), p(FIXEDNODES, 2), 'red');
hold off
numNodes = length(p);

%% solve SPDE
[R, flag, transP] = get_precision_mat_isotropy(kappa, p, t, normconst);  % the important matrix Q R
disp("Done calculating precision matrix Q");

% modify multiply geometries
g = normrnd(mu,sigma,numNodes,N); % the white noise vector following Gaussian distribution
eta = transP * (R \ g); % the random field
% visualization
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta(:,2),'FaceColor','flat','EdgeColor', 'none');
colorbar;

%% generate dataset
rng(41);
N = 100;
g1 = normrnd(mu,sigma,numNodes,N); 
eta1 = transP * (R \ g1); 
g2 = normrnd(mu,sigma,numNodes,N); 
eta2 = transP * (R \ g2); 
clear g1 g2
eta_bulk = convert_beta_bulk(bulk_moduli_height, delta, eta1)+bulk_moduli;
eta_shear = convert_beta_shear(shear_moduli_height, delta, eta1, eta2, rho)+shear_moduli;
save ../../data/mesoscale_grid/input_N100_s128.mat eta_bulk eta_shear p t FIXEDNODES


