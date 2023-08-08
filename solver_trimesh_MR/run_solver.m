function [WEFF, UX, UY] = run_solver(PROP_CPP, mesh, C1, C2, C3)
% Load and plot the mesh
p = mesh.p; p = p(:, 1:2);
t = mesh.t;  NE = mesh.NE; ND = mesh.ND; N = mesh.N;
% need to pay attention to the next two varaibles:
PHASES = mesh.PHASES+1;
FIXEDNODES = double(mesh.FIXEDNODES); 

% FE COMPUTATIONS
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

% NONLINEAR FEM
CauchyGreen = [C1, C3; 
               C3, C2];
F = chol(CauchyGreen);
H = F - eye(2);
[WEFF, DISPTD] = NL2DSOLVER(FIXEDNODES, p, PROP_CPP, NE, GDOF, ND, H, PHASES, ...
sb, sc, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE);

UX = DISPTD(1:2:end); UY = DISPTD(2:2:end);

end








