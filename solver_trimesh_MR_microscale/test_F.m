load data/coef.mat u_interp
s = size(u_interp, 1)-1;
uxx = (u_interp(1:end-1, 2:end,1) - u_interp(1:end-1, 1:end-1,1)) * s;
uxy = (u_interp(2:end, 1:end-1,1) - u_interp(1:end-1, 1:end-1,1)) * s;
uyx = (u_interp(1:end-1, 2:end,2) - u_interp(1:end-1, 1:end-1,2)) * s;
uyy = (u_interp(2:end, 1:end-1,2) - u_interp(1:end-1, 1:end-1,2)) * s;

F = calc_F(u_interp, 0, 0);
uxx2 = reshape(F(:,1), s, s);
uxy2 = reshape(F(:,2), s, s);
uyx2 = reshape(F(:,3), s, s);
uyy2 = reshape(F(:,4), s, s);

%
% figure(1)
% contourf(a);
subplot(1,2,1)
contourf(uyy); colorbar;
subplot(1,2,2)
contourf(uyy2); colorbar;