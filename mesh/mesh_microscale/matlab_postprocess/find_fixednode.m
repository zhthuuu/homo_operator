function [fixedid, ND] = find_fixednode(p, t)
    % find the node id on the boundary
    t1 = cast(t, "int64");
    a = unique(t1);
    eps = 1e-4;
    fixedid = (abs(p(a,1))<eps) + (abs(p(a,1)-1)<eps) + (abs(p(a,2))<eps) + (abs(p(a,2)-1)<eps);
    fixedid = a(fixedid>0);
    ND = length(fixedid);
%     scatter(p(fixedid, 1), p(fixedid, 2));
end