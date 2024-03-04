function [xdot, u] = admire_ss_randn(t, x, w, A, B, WB, R, tf, x0, tspan)
    
    u = -B'*expm(A'*(tf-t))*(WB\expm(A*tf))*x0 + -B'*expm(A'*(tf-t))*(WB\R);
    
    xdot = A*x + B*u + [interp1(tspan, w(1,:), t); interp1(tspan, w(2,:), t); interp1(tspan, w(3,:), t)];

end