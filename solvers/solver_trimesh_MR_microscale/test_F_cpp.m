clc; clear;
load data/coef.mat
N_ele = size(t, 1);
F = zeros(N_ele, 4);
p2 = zeros(N_ele, 2);
for e = 1:length(t)
    uE = zeros(6,1);
    for j = 1:6
        uE(j) = DISPTD(t2D(DOFe, j, e-1, 6));
    end
    U = [uE(1), uE(3),uE(5); 
         uE(2),uE(4),uE(6)];
    D = [t2D(sb,1,e-1,3), t2D(sc,1,e-1,3);
         t2D(sb,2,e-1,3), t2D(sc,2,e-1,3);
         t2D(sb,3,e-1,3), t2D(sc,3,e-1,3)];
    Fe = U*D;
    F(e, :) = reshape(Fe, 1, 4);
    con = t(e, :);
    pe = p(con, :);
    p2(e, :) = mean(pe, 1);
end

% plot
subplot(2,2,1);
hold on
scatter(p2(:,1), p2(:,2), 10, F(:,1), 'filled'); colorbar; 
colormap('jet');
subplot(2,2,2);
scatter(p2(:,1), p2(:,2), 10, F(:,2), 'filled'); colorbar; 
colormap('jet');
subplot(2,2,3);
scatter(p2(:,1), p2(:,2), 10, F(:,3), 'filled'); colorbar; 
colormap('jet');
subplot(2,2,4);
scatter(p2(:,1), p2(:,2), 10, F(:,4), 'filled'); colorbar; 
colormap('jet');
hold off
function x = t2D(A, i, j, N)
    x = A(j*N+i);
end

