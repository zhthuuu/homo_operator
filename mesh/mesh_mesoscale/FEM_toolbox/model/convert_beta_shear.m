function eta_beta = convert_beta_shear(height, delta, eta_norm1, eta_norm2, rho)
    % calculate stochastic shear moduli 
    % rho: coefficient of correlation between the stochastic bulk and shear moduli
    % delta = 0.57: free model parameter, delta = 2*std
    alpha = (1/delta^2-1)/2; % change of parameterization
    eta_norm = rho*eta_norm1 + sqrt(1-rho*rho)*eta_norm2;
    eta_beta = betainv(normcdf(eta_norm, 0, 1), alpha, alpha); % assuming that your Gaussian field is stored as G_CSF for the CSF
    eta_beta = height * eta_beta;

end

% height = 1;
% x = linspace(-5,5,100);
% y_normcdf = normcdf(x,0,1);
% y_betacdf = y_normcdf;
% x_beta = betainv(y_betacdf, 12, 12);
% y_betapdf = betapdf(x_beta, 12, 12);
% x_beta_modified = -height + 2*height*x_beta;
% plot(x_beta_modified, y_betapdf);