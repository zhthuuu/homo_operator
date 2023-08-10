clc; clear;
N = 1000;
for i = 1:1000
    file = ['../samples_mesh/sample', num2str(i), '.m'];
    run(file);
    p = msh.POS;
    t = msh.TRIANGLES;

    % identify the outside elements
    eps = 1e-4;
    outside_id = (p(:,1)+eps<0) + (p(:,1)-eps>1) + (p(:,2)+eps<0) + (p(:,2)-eps>1);
    p_id = linspace(1, length(p), length(p));
    outside_id = p_id(outside_id>0);
    triangle_id = zeros(size(t,1), 1);
    for k = 1:length(outside_id)
        for j = 1:3
            triangle_id = triangle_id + (t(:,j)==outside_id(k));
        end
    end

    t(triangle_id>0, :) = []; % delete the outside triangles
    [C, IA, IC] = unique(t(:,1:3));
    C2 = 1:length(C);
    t2 = C2(IC); t2 = reshape(t2, size(t(:,1:3)));
    p2 = p(C, 1:2);
    phase = t(:,4);
    
    NE = length(phase); % number of elements
    N = length(p2); % number of nodes
    [FIXEDNODES, ND] = find_fixednode(p2, t2);
    
    trimesh(i).p = p2;
    trimesh(i).t = t2;
    trimesh(i).PHASES = phase;
    trimesh(i).NE = NE;
    trimesh(i).N = N;
    trimesh(i).ND = ND;
    trimesh(i).FIXEDNODES = FIXEDNODES;
end

%% save data
save data/trimesh_N1000 trimesh

%% plot
id = 5;
p = trimesh(id).p; t = trimesh(id).t; PHASES = trimesh(id).PHASES;
FIXEDNODES = trimesh(id).FIXEDNODES;
figure(2);
triplot(t(PHASES==0, :), p(:,1), p(:,2), 'r');
hold on;
triplot(t(PHASES==1, :), p(:,1), p(:,2), 'b');
axis equal off
hold on;
scatter(p(FIXEDNODES,1), p(FIXEDNODES, 2),'k', 'filled');
hold off;



