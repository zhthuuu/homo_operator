% define the graduate matrix
% [\frac{\partial N}{\partial x};\frac{\partial N}{\partial y}
function [G] = getGradN_Q4(x, y)
G = [-1/4*(1+y),-1/4*(1-y),1/4*(1-y),1/4*(1+y); ...
    1/4*(1-x),-1/4*(1-x),-1/4*(1+x),1/4*(1+x)];
end