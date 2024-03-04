function u_norm = control_energy(t, A, B, WB, R, tf, x0)
    u = -B'*expm(A'*(tf-t))*(WB\expm(A*tf))*x0 + -B'*expm(A'*(tf-t))*(WB\R);
    u_norm = u' * u;
end