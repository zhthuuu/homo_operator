% Generate stochastic bulk and shear moduli on a plane square domain
% @Hao Zhang, 8/5/2023

clc; clear;
addpath ../FEM_toolbox/model/
addpath ../FEM_toolbox/2d_trimesh/
seed = RandStream('mt19937ar', 'Seed', 1); RandStream.setGlobalStream(seed); %set seed
file = '../FEM_toolbox/geometry/square/mesh_composite.mat'; % input brain tissue boundary mesh
dir = 'data/'; 
if ~exist(dir, 'dir')
    mkdir(dir); % generated mesh folder
end

l = 0.1; % the coorrelation length
N = 10; % number of generated meshes
delta = 0.57; % coefficient for beta distribution
rho = 0.9; % coefficient of correlation between stochastic bulk and shear modulio
height = 1;

% material properties
bulk_moduli = 1.75e4; % bulk modulus in MPa
shear_moduli = 3281.25; % shear modulus in MPa
bulk_moduli_height = bulk_moduli * 0.1;
shear_moduli_height = shear_moduli*0.1;

%% preprocessing
kappa = 1/l;  
mu = 0; sigma = 1;  % the mean and standard deviation of gaussian field
nu = 1; d = 2; % the parameters in the SPDE
normconst = sigma^2*(4*pi)^(d/2)*gamma(nu+d/2)/gamma(nu);
normconst = normconst*l^(d-4); % alpha multiplied to the white noise vector
% msh = stlread(file);
% p = msh.Points;
% t = msh.ConnectivityList;
load(file, 'p', 't', 'FIXEDNODES');
numNodes = size(p, 1);
if size(p, 2) ~= 3
    p = [p, zeros(numNodes, 1)];
end
disp(['Done reading mesh ', file]);

%% solve SPDE
[R, flag, transP] = get_precision_mat_isotropy(kappa, p, t, normconst);  % the important matrix Q R
disp("Done calculating precision matrix Q");

% modify multiply geometries
g_bulk = normrnd(mu,sigma,numNodes,N); % the white noise vector following Gaussian distribution
eta_bulk = transP * (R \ g_bulk); % the random field
g_shear = normrnd(mu,sigma,numNodes,N); % the white noise vector following Gaussian distribution
eta_shear = transP * (R \ g_shear); % the random field

eta_beta_bulk = convert_beta_bulk(bulk_moduli_height, delta, eta_bulk)+bulk_moduli;
eta_beta_shear = convert_beta_shear(shear_moduli_height, delta, eta_bulk, eta_shear, rho)+shear_moduli;

%% visualization
id = 4;
f = figure;
subplot(1,2,1)
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_beta_bulk(:,id),'FaceColor','interp','EdgeColor', 'white');
colorbar;
subplot(1,2,2)
patch('Faces',t,'Vertices',p,'FaceVertexCData',eta_beta_shear(:,id),'FaceColor','interp','EdgeColor', 'none');
colorbar;
f.Position = [200 200 900 400];

% %% save single data
% bulk = eta_beta_bulk(:,1);
% shear = eta_beta_shear(:,1);
% bulk = mean(bulk(t), 2);
% shear = mean(shear(t), 2);
% save data/mesh p t FIXEDNODES bulk shear
