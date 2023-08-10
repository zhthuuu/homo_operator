function [conflict, dist] = check_conflict(circles, r)
    N = size(circles, 1);
    dist = zeros(N, N);
    for i = 1:N
        dist(:,i) = vecnorm(circles - circles(i,:), 2, 2);
    end
    dist(dist==0) = 1;
    mindist = min(dist, [], 'all');
    if mindist < 2*r
        conflict = true;
    else
        conflict = false;
    end
end