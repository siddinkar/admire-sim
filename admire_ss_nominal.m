function xdot = admire_ss_nominal(t, x, A, B, WB, tf, x0)
    
    u = -B'*expm(A'*(tf-t))*(WB\expm(A*tf))*x0;
    
    xdot = A*x + B*u;

end