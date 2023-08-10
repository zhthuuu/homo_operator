clc; clear;
%%
run ../samples_mesh/sample40.m
p = msh.POS;
t = msh.TRIANGLES;

% identify the outside elements
eps = 1e-4;
outside_id = (p(:,1)+eps<0) + (p(:,1)-eps>1) + (p(:,2)+eps<0) + (p(:,2)-eps>1);
p_id = linspace(1, length(p), length(p));
outside_id = p_id(outside_id>0);
triangle_id = zeros(size(t,1), 1);
for i = 1:length(outside_id)
    for j = 1:3
        triangle_id = triangle_id + (t(:,j)==outside_id(i));
    end
end


%% reshape p and t
t(triangle_id>0, :) = []; % delete the outside triangles
[C, IA, IC] = unique(t(:,1:3));
C2 = 1:length(C);
t2 = C2(IC); t2 = reshape(t2, size(t(:,1:3)));
p2 = p(C, 1:2);

%%
phase = t(:,4);
figure(2)
triplot(t2(phase==0, :), p2(:,1), p2(:,2), 'r');
hold on;
triplot(t2(phase==1, :), p2(:,1), p2(:,2), 'b');
axis equal off
