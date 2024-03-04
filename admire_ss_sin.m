function [xdot, u] = admire_ss_sin(t, x, A, B, WB, R, tf, x0)
    
    u = -B'*expm(A'*(tf-t))*(WB\expm(A*tf))*x0 + -B'*expm(A'*(tf-t))*(WB\R);
    
    xdot = A*x + B*u + [sin(t) ; sin(t) ; sin(t)];

end