function F = calc_F(u, x, y)
    % calclate the gradient field: [u_x,x, u_x,y; u_y,x, u_y,y]
    % return: [u,x, u,y]
    % Jacobian matrix
    s = size(u, 1) - 1; % number of elements per edge
    l = 1/s;
    J = l/2*[0,1;-1,0];
    ux = cat(3, u(2:end, 1:end-1,1), u(2:end, 2:end,1), u(1:end-1, 2:end,1), u(1:end-1, 1:end-1,1));
    uy = cat(3, u(2:end, 1:end-1,2), u(2:end, 2:end,2), u(1:end-1, 2:end,2), u(1:end-1, 1:end-1,2));
    ux = flip(ux, 3);
    uy = flip(uy, 3);
    ux = reshape(ux, s*s, 4); % (s*s) x 4
    uy = reshape(uy, s*s, 4); % (s*s) x 4
    B = J \ getGradN_Q4(x, y); % 2x4
    gradux = ux * B'; % (s*s) x 2
    graduy = uy * B'; % (s*s) x 2
    I = [1,0,0,1];
    F = cat(2, gradux, graduy) + I;
end