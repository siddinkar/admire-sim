close all
clear
clc
format compact

A = [-0.9967 0 0.6176; 0 -0.5057 0; -0.0939  0 -0.2127];
B = [0 -4.2423 4.2423 1.4871; 1.6532 -1.2735 -1.2735 0.0024; 0 -0.2805 0.2805 -0.8823];

tf = 5;
WB = integral(@(tau) expm(A*tau)*(B*B')*expm(A'*tau), 0, tf, 'ArrayValued', true);


x0 = [3 / sqrt(3); 3 / sqrt(3); 3 / sqrt(3)];


tspan = linspace(0, tf, 1000);

% nominal trajectory
[t_nominal, x_nominal] = ode45(@(t, x) admire_ss_nominal(t, x, A, B, WB, tf, x0), tspan, x0);

y_nominal = randn(length(x_nominal), 1);
for i = 1:length(x_nominal)
    y_nominal(i) = norm(x_nominal(i, :));
end

% random disturbance
w_rand = randn(3, length(tspan)) * 5;
R_rand = integral(@(tau) expm(A*(tf - tau)) * [interp1(tspan, w_rand(1,:), tau); interp1(tspan, w_rand(2,:), tau); interp1(tspan, w_rand(3,:), tau)], 0, tf, 'ArrayValued', true);
[t_rand,x_rand] = ode45(@(t,x) admire_ss_randn(t, x, w_rand, A, B, WB, R_rand, tf, x0, tspan), tspan, x0);

y_rand = randn(length(x_rand), 1);
for i = 1:length(x_rand)
    y_rand(i) = norm(x_rand(i, :));
end

% sinusoidal disturbance
R_sin = integral(@(tau) expm(A * (tf - tau)) * [sin(tau) ; sin(tau); sin(tau)], 0, tf, 'ArrayValued', true);
[t_sin, x_sin] = ode45(@(t, x) admire_ss_sin(t, x, A, B, WB, R_sin, tf, x0), tspan, x0);

y_sin = randn(length(x_sin), 1);
for i = 1:length(x_sin)
    y_sin(i) = norm(x_sin(i, :));
end

% constant disturbance
w_const = [1 / sqrt(3); 1 / sqrt(3); 1 / sqrt(3)];
R_const = integral(@(tau) expm(A *(tf - tau)) * w_const, 0, tf, 'ArrayValued', true);
[t_const, x_const] = ode45(@(t,x) admire_ss_const(t, x, w_const, A, B, WB, R_const, tf, x0), tspan, x0);

y_const = randn(length(x_const), 1);
for i = 1:length(x_const)
    y_const(i) = norm(x_const(i, :));
end

 % control energies
[tn,En] = ode45(@(t,E) control_energy(t, A, B, WB, [0 ; 0; 0], tf, x0), tspan, 0);
[tr,Er] = ode45(@(t,E) control_energy(t, A, B, WB, R_rand, tf, x0), tspan, 0);
[ts,Es] = ode45(@(t,E) control_energy(t, A, B, WB, R_sin, tf, x0), tspan, 0);
[tc,Ec] = ode45(@(t,E) control_energy(t, A, B, WB, R_const, tf, x0), tspan, 0);

[V, D] = eig(inv(WB));
q_bar = norm(V, 1) * integral(@(tau) norm(expm(A * (tf - tau)), inf), 0, tf, 'ArrayValued', true);
sum = 0;
for lambda=1:3
    sum = sum + q_bar^2 * D(lambda, lambda);
end
E_bound = En(length(En)) + 2 * q_bar * norm(D * V' * expm(A * tf) * x0, 1) + sum;


figure(1)
set(gcf, 'DefaultLineLineWidth', 2)
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')
plot(t_nominal, y_nominal, 'b')
hold on
plot(t_rand, y_rand, 'r')
plot(t_sin, y_sin, 'k')
plot(t_const, y_const, 'g')
xlabel('Time $t$ (s)')
ylabel('$||x(t)||$')
legend('$w = 0$', '$w \sim \mathcal{N}(0,1)$', '$w = sin(t)$', '$w = \bar{w} * sgn(x0)$')
grid on
box on
set(gca, 'FontSize', 18)

figure(2)
set(gcf, 'DefaultLineLineWidth', 2)
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')
plot(tn, En, 'b')
hold on
plot(tr, Er, 'r')
plot(ts, Es, 'k')
plot(tc, Ec, 'g')
% plot(tc, E_bound, 'o', LineStyle='--')
xlabel('Time $t$ (s)')
ylabel('$Control Energy$')
legend('$w = 0$', '$w \sim \mathcal{N}(0,1)$', '$w = sin(t)$', '$w = \bar{w} * sgn(x0)$')
grid on
box on
set(gca, 'FontSize', 18)