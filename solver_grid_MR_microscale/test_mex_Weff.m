% Now the code mex_EFFSEF.cpp has been validated
clc; clear;
% load ../data/phase_N1000_s420.mat
% load ../data/s421_sample100.mat
load ../data/dataset_train_s64_N600.mat
s = 64; sample = 100;
PHASES = reshape(a(sample,:,:), s, s);
PHASES = reshape(flip(PHASES)', [], 1) + 1;
PHASES = int32(PHASES);
u_grid = reshape(u(sample,:,:,:), s+1,s+1,2);
NE = s * s; ND = s*4; % number of fixed nodes
N = (s+1)*(s+1);
[p, t, FIXEDNODES] = generate_mesh(s);
FIXEDNODES = double(FIXEDNODES);
Ae = 1 / (s*s);

%% DISP
ux = u_grid(:,:,1); ux = reshape(flip(ux)', [], 1);
uy = u_grid(:,:,2); uy = reshape(flip(uy)', [], 1);
DISP = zeros(2*N,1);
DISP(1:2:end-1) = ux; 
DISP(2:2:end) = uy;

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
%% DOFe
DOFe = zeros(8, NE); 
for e = 1:NE
    NODES = t(e, :);
    DOFe(:, e) = [2*NODES(1)-1, 2*NODES(1), 2*NODES(2)-1, 2*NODES(2), ...
                    2*NODES(3)-1, 2*NODES(3),2*NODES(4)-1, 2*NODES(4)]';        
end
DOFe = int32(DOFe);
%% Weff
Weff = mex_EFFSEF(DISP, NE, PROP_CPP, PHASES, s, Ae, DOFe)


