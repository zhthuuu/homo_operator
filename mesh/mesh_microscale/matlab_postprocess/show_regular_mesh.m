clc; clear;

load data/phase_N1000_s256.mat
phase = permute(phase, [2, 3, 1]);
s = 256;
[X, Y] = meshgrid(linspace(0,1, s), linspace(0,1,s));
%%
i = 205;
figure(1)
mesh(X, Y, phase(:,:,i), 'FaceColor','flat');
fig = gcf();
fig.Position = [200,200, 400, 400]; view(2)
axis equal off
