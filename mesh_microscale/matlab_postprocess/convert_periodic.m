function circles = convert_periodic(circles, r)

deltar = [1,0; -1,0; 0,-1; 0,1]; % left, right, up, down
for i = 1:size(circles, 1)
    x = circles(i, 1);
    y = circles(i, 2);
    d = [x, 1-x, 1-y, y] - r;
    idx = (d<0); % circle is either close to only one edge or two edges (in the conor)
    if sum(idx) > 0
        circle_new = repmat(circles(i,:),sum(idx),1) + deltar(idx,:);
        circles = [circles; circle_new];
    end
end

end