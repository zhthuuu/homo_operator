clc; clear;

% read circle infomation
file = '../samples_packed/sample100.dat';
fid = fopen(file, 'r');
line = 0;
lattice = zeros(2);
circles = [];
r = 0;
N_circle = 0;
string = 0;
while true
    line = line + 1;
    string = fgetl(fid);
    if string == -1
        break;
    end
    if line == 2
        N_circle = str2double(string);
        circles = zeros(N_circle, 2);
    end
    if line == 3
        r = str2double(string);
    end
    if line == 4
        lattice(1,:) = str2num(string);
    end
    if line == 5
        lattice(2,:) = str2num(string);
    end
    if line > 5
        circles(line-5, :) = str2num(string);
    end
end
circles = circles / lattice;

%% identify circles close to edge
% 1. identify the circles close to any edge
% 2. add a new mirrow circle to the circle list
deltar = [1,0; -1,0; 0,-1; 0,1]; % left, right, up, down
for i = 1:size(circles, 1)
    x = circles(i, 1);
    y = circles(i, 2);
    d = [x, 1-x, 1-y, y] - r;
    idx = (d<0); % circle is either close to only one edge or two edges (in the conor)
    if sum(idx) > 0
        circle_new = repmat(circles(i,:),sum(idx),1) + deltar(idx,:);
        circles = [circles; circle_new];
    end
end

%% generate mesh
N_node = 256;
N_circle = size(circles, 1);
x = linspace(0,1, N_node);
[X, Y] = meshgrid(x, x);
distmat = zeros(N_node, N_node, N_circle);
phase = zeros(N_node);
for i = 1:N_circle
    distmat(:,:,i) = (X-circles(i,1)).^2 + (Y-circles(i,2)).^2;
    phase(distmat(:,:,i)<r*r) = 1;
end

mesh(X,Y,phase, 'FaceColor','flat'); view(2); 
axis equal off



