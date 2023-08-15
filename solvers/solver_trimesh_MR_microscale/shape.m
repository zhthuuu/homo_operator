function [sb, sc, Ae] = shape(p, t, NE)

sb = zeros(3, NE); sc = zeros(3, NE); Ae = zeros(NE, 1);

for e = 1 : NE
    
    NODES = t(e, :);
    
    X = p(NODES, 1); Y = p(NODES, 2); 
    
    [Ae(e, 1), sb(:, e), sc(:, e)] = shape_constants(X, Y);
    
end

end

function [Aire, b, c] = shape_constants(x, y)

C = [ones(3, 1), x, y];

u = C\[1; 0; 0];
v = C\[0; 1; 0];
w = C\[0; 0; 1];

%a = [u(1), v(1), w(1)];
b = [u(2), v(2), w(2)]';
c = [u(3), v(3), w(3)]';

Aire = abs( 0.5*( (x(2)-x(1))*(y(3)-y(1)) - (y(2)-y(1))*(x(3)-x(1)) ) );

end
    



