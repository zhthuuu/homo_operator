clc; clear;

run samples_mesh/sample1.m
p = msh.POS;
t = msh.TRIANGLES;

% remove the outside elements
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

t(triangle_id>0, :) = []; % delete the outside triangles

phase = t(:,4);
t = t(:,1:3);
triplot(t(phase==0, :), p(:,1), p(:,2), 'r');
hold on;
triplot(t(phase==1, :), p(:,1), p(:,2), 'b');
axis equal off

