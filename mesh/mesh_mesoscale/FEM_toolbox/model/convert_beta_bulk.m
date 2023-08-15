function eta_beta = convert_beta_bulk(height, delta, eta_norm)
    % calculate stochastic shear moduli 
    % rho: coefficient of correlation between the stochastic bulk and shear moduli
    % delta = 0.57: free model parameter, delta = 2*std
    alpha = (1/delta^2-1)/2; % change of parameterization
    eta_beta = betainv(normcdf(eta_norm, 0, 1), alpha, alpha); 
    eta_beta = 2*height * eta_beta-height;

end

% height = 1;
% x = linspace(-5,5,100);
% y_normcdf = normcdf(x,0,1);
% y_betacdf = y_normcdf;
% x_beta = betainv(y_betacdf, 12, 12);
% y_betapdf = betapdf(x_beta, 12, 12);
% x_beta_modified = -height + 2*height*x_beta;
% plot(x_beta_modified, y_betapdf);