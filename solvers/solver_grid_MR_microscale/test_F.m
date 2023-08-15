% Now the gradient has been calibrated with the c++ code using trimesh
clc; clear;
% load ../data/s421_sample100.mat
load ../data/dataset_train_s64_N600.mat
s = 64; sample = 100;
PHASES = reshape(a(sample,:,:), s, s);
PHASES = reshape(flip(PHASES)', [], 1) + 1;
PHASES = int32(PHASES);
u_grid = reshape(u(sample,:,:,:), s+1,s+1,2);
ux = u_grid(:,:,1); ux = reshape(flip(ux)', [], 1);
uy = u_grid(:,:,2); uy = reshape(flip(uy)', [], 1);
u = [ux, uy];
N_ele = s*s;
[p, t] = generate_mesh(s);
F = zeros(N_ele, 4);
for e = 1:length(t)
    uE = zeros(8,1);
    U = u(t(e,:), :)';
    D = [-1, -1;
          1, -1;
          1,  1;
         -1,  1;] * s/2;
    Fe = U*D;
    F(e, :) = reshape(Fe, 1, 4);
end

%% plot
[X, Y] = meshgrid(linspace(0,1,s), linspace(0,1,s));
figure();
subplot(2,2,1)
F1 = flip(reshape(F(:,1), s, s)');
mesh(X, Y, F1, 'FaceColor','flat'); view(2); colorbar; colormap('jet');
subplot(2,2,2)
F2 = flip(reshape(F(:,2), s, s)');
mesh(X, Y, F2, 'FaceColor','flat'); view(2); colorbar; colormap('jet');
subplot(2,2,3)
F3 = flip(reshape(F(:,3), s, s)');
mesh(X, Y, F3, 'FaceColor','flat'); view(2); colorbar; colormap('jet');
subplot(2,2,4)
F4 = flip(reshape(F(:,4), s, s)');
mesh(X, Y, F4, 'FaceColor','flat'); view(2); colorbar; colormap('jet');


%% C
C11 = F(:,1).*F(:,1)+F(:,2).*F(:,2); C11 = flip(reshape(C11, s, s)');
C12 = F(:,1).*F(:,3)+F(:,2).*F(:,4); C12 = flip(reshape(C12, s, s)');
C21 = F(:,1).*F(:,3)+F(:,2).*F(:,4); C21 = flip(reshape(C21, s, s)');
C22 = F(:,3).*F(:,3)+F(:,4).*F(:,4); C22 = flip(reshape(C22, s, s)');
subplot(2,2,1)
mesh(X, Y, C11, 'FaceColor','flat'); view(2); colorbar; colormap('jet'); subtitle('C11');
subplot(2,2,2)
mesh(X, Y, C12, 'FaceColor','flat'); view(2); colorbar; colormap('jet');title('C12');
subplot(2,2,3)
mesh(X, Y, C21, 'FaceColor','flat'); view(2); colorbar; colormap('jet');title('C21');
subplot(2,2,4)
mesh(X, Y, C22, 'FaceColor','flat'); view(2); colorbar; colormap('jet');title('C22');

%% Weff
close all
load ../data/PROP.mat
Weff = 0; Ae = 1/(s*s);
tmp_cpp = zeros(s*s, 1);
for e = 1:length(t)
    % prop
    phaseElem = PHASES(e);
    alpha1 = PROP_CPP(phaseElem, 1);
    beta1 = PROP_CPP(phaseElem, 2);
    s1 = PROP_CPP(phaseElem, 3);
    s2 = 2*(alpha1+2*beta1);
    % F
    Fe = reshape(F(e,:), 2, 2) + eye(2);
    J = det(Fe);
    C = Fe' * Fe;
    We = alpha1*(trace(C) - 2) + ...
        beta1*(C(1,1)+C(2,2)+C(1,1)*C(2,2)-C(1,2)*C(2,1)-3) + ...
        s1*(J-1)*(J-1)/2 - s2*log(J);
    Weff = Weff + We * Ae;
    tmp_cpp(e) = Ae * alpha1*(trace(C) - 2);
end
tmp_cpp = flip(reshape(tmp_cpp, s, s)');
figure(2);
mesh(X, Y, tmp_cpp, 'FaceColor','flat'); view(2); colorbar; colormap('jet');



