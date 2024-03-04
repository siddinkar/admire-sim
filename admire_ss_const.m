function [xdot, u] = admire_ss_const(t, x, w, A, B, WB, R, tf, x0)
    
    u = -B'*expm(A'*(tf-t))*(WB\expm(A*tf))*x0 + -B'*expm(A'*(tf-t))*(WB\R);
    
    xdot = A*x + B*u + w;

end