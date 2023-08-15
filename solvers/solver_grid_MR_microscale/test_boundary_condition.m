clc; clear;
s = 50;
% pleft = [zeros(s,1), linspace(0,1,s)'];
% pright = [ones(s,1), linspace(0,1,s)'];
% pdown = [linspace(0,1,s)', zeros(s,1)];
% pup = [linspace(0,1,s)', ones(s,1)];

C1 = 1.5; C2 = 1.; C3 = 0;
C = [C1, C3; C3,C2];
F = chol(C);
H = F - eye(2);
p = [0,0;0,1;1,1;1,0];
p2 = p * F;
figure(1)
hold on
patch(p(:,1), p(:,2), 'r','facealpha', 0.3);
patch(p2(:,1), p2(:,2), 'g','facealpha', 0.3)
ylim([0,1]);
axis equal
hold off
% set(gcf, 'Position', [200, 200, 1000, 400])
% p = [pleft; pright; pdown; pup];
% p_deformed = p*F;
% u = p*H;
% scatter(p_deformed(:,1), p_deformed(:,2)); 
% axis equal
% scatter(u(:,1), u(:,2));
