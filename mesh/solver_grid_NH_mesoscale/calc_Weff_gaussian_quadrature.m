function [Weff_total] = calc_Weff_gaussian_quadrature(a, u, prop)
    gpt = 1/sqrt(3);
    x = [-gpt, gpt, gpt, -gpt];
    y = [-gpt, -gpt, gpt, gpt];  
    wt = [1., 1., 1., 1.];
    % area per grid
    Nele = size(a, 1);
    Ae = 1/(Nele * Nele);
    Jac_det = Ae/4.;
    % calcualte W_eff
    Weff_total = 0;
    for i = 1:length(x)
    F = calc_F(u, x(i), y(i));
%     F = calc_F(u, 0, 0);
    % Jacobian
    J = F(:,1).*F(:,4) - F(:,2).*F(:,3);
    % J must be strictly positive
%         J(J<0) = 1e-8;
    % C = F.transpose() * F
%         C00 = F(:,1).*F(:,1)+F(:,2).*F(:,2);
%         C01 = F(:,1).*F(:,3)+F(:,2).*F(:,4);
%         C10 = F(:,1).*F(:,3)+F(:,2).*F(:,4);
%         C11 = F(:,3).*F(:,3)+F(:,4).*F(:,4);
    C00 = F(:,1).*F(:,1)+F(:,3).*F(:,3);
    C01 = F(:,1).*F(:,2)+F(:,3).*F(:,4);
    C10 = F(:,1).*F(:,2)+F(:,3).*F(:,4);
    C11 = F(:,2).*F(:,2)+F(:,4).*F(:,4);
    % properties
    a_vec = reshape(a, Nele*Nele, 1);
    alpha1 = zeros(size(a_vec));
    beta1 = zeros(size(a_vec));
    s1 = zeros(size(a_vec));
    alpha1(a_vec==0) = prop(1,1);
    alpha1(a_vec==1) = prop(2,1);
    beta1(a_vec==0) = prop(1,2);
    beta1(a_vec==1) = prop(2,2);
    s1(a_vec==0) = prop(1,3);
    s1(a_vec==1) = prop(2,3);
    s2 = 2*(alpha1 + 2*beta1);
    % weff
    Weff = alpha1 .* (C00 + C11 - 2) + ...
            beta1 .* (C00+C11+C00.*C11-C01.*C10-3) + ...
            s1.*(J-1).*(J-1)/2 - s2.*log(J);
%     Weff_total = Ae * sum(Weff, "all");
%     tmp = tmp + wt(i) * C00 / 4;
    Weff_total = Weff_total + wt(i) * Jac_det * sum(Weff, "all");
    end

end