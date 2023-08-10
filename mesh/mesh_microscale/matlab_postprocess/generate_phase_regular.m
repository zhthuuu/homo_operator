% generate regular mesh of RVE given the positions of circle inclusions

clc; clear;
N = 1500; % number of samples
N_node = 421; % mesh resolution

% generate mesh
x = linspace(0,1, N_node);
[X, Y] = meshgrid(x, x);
phase = zeros(N, N_node, N_node);
N_conflict = 0;
conflict_list = [];

for i = 1:N
    % read circle infomation
    file = ['../samples_packed/sample', num2str(i), '.dat'];
    [circles, r] = read_circles(file);
    % add periodic circles
    circles = convert_periodic(circles, r);
    % identify the conflicted RVEs
    [conflict, ~] = check_conflict(circles, r);
    if conflict
        N_conflict = N_conflict + 1;
        conflict_list = [conflict_list, i];
        continue;
    end
    N_circle = size(circles, 1);
    distmat = zeros(N_node, N_node, N_circle);
    for k = 1:N_circle
        distmat(:,:,k) = (X-circles(k,1)).^2 + (Y-circles(k,2)).^2;
        phase(i, distmat(:,:,k)<r*r) = 1;
    end
end

% remove the conflicted RVEs
phase(conflict_list, :,:) = [];
phase = phase(1:1000, :,:);
save data/phase_N1000_s421.mat phase

%% show mesh
phase = permute(phase, [2,3,1]); 
imshow(phase(:,:,1));
% [X, Y] = meshgrid(1:N_circle, 1:N_circle);
% idx = (dist < r) & (dist > 0);
% coord = X(idx);


