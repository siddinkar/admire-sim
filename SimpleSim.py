import numpy as np
from scipy.integrate import odeint, quad, quad_vec, solve_ivp, ode
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm
import sdeint


class Controller:
    def __init__(self, R, duration):
        # self.IC = np.linspace(np.ones(3).T / np.sqrt(3), np.ones(3).T * R / np.sqrt(3), R)
        self.x0 = np.ones(3).T * R / np.sqrt(3)
        # self.x0 = self.IC[0]
        self.t = np.linspace(0, duration, 100)
        self.tspan = [0, duration]
        # self.w = np.ones(3).T / np.sqrt(3) * np.sign(R)
        self.w = lambda x: np.array([np.tanh(x), np.sin(x), np.cos(x)]).T

        self.A = np.array([[-0.9967, 0, 0.6176], [0, -0.5057, 0], [-0.0939, 0, -0.2127]])
        self.B = np.array([[0, -4.2423, 4.2423, 1.44871],
                           [1.6532, -1.2735, -1.2735, 0.0024],
                           [0, -0.2805, 0.2805, -0.8823]])

        self.E_N = 0
        self.E_D = 0
        self.E_D_series = []
        self.E_N_series = []

        self.prev_t_N = 0.0
        self.prev_t_D = 0.0

        self.tf = self.t[-1]
        self.W = self.gramian()

    def gramian(self):
        func = lambda x: expm(self.A * x) @ self.B @ self.B.T @ expm(self.A.T * x)
        return quad_vec(func, 0, self.tf)[0]

    def R(self):
        func = lambda x: expm(self.A * (self.tf - x)) @ self.w(x)
        return quad_vec(func, 0, self.tf)[0]

    def dxdt(self, x, timestep):
        u_N = -self.B.T @ expm(self.A.T * (self.tf - timestep)) @ np.linalg.inv(self.W) @ expm(
            self.A * self.tf) @ self.x0
        self.E_N += (u_N.T @ u_N) * (timestep - self.prev_t_N)
        self.E_N_series.append([self.E_N, timestep])
        x_ = self.A @ x + self.B @ u_N
        self.prev_t_N = timestep
        return x_

    def dxdt_disturbed(self, x, timestep):
        # w = np.random.uniform(-1, 1, (3,))
        u_D = -self.B.T @ expm(self.A.T * (self.tf - timestep)) @ np.linalg.inv(self.W) @ expm(
            self.A * self.tf) @ self.x0
        u_D += -self.B.T @ expm(self.A.T * (self.tf - timestep)) @ np.linalg.inv(self.W) @ self.R()
        self.E_D += (u_D.T @ u_D) * (timestep - self.prev_t_D)
        self.E_D_series.append([self.E_D, timestep])
        x_ = self.A @ x + self.B @ u_D + self.w(timestep)
        self.prev_t_D = timestep
        return x_




if __name__ == "__main__":
    c = Controller(10, 20)
    w_bar = 1;


    nominal = np.array(odeint(c.dxdt, c.x0, c.t))
    nominal = np.array([np.linalg.norm(i) for i in nominal])
    disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
    disturbed = np.array([np.linalg.norm(i) for i in disturbed])

    y = plt.plot(np.array(c.E_D_series).T[1], np.array(c.E_D_series).T[0], label="10m")
    c = Controller(50, 20)
    nominal = np.array(odeint(c.dxdt, c.x0, c.t))
    nominal = np.array([np.linalg.norm(i) for i in nominal])
    disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
    disturbed = np.array([np.linalg.norm(i) for i in disturbed])

    r = plt.plot(np.array(c.E_D_series).T[1], np.array(c.E_D_series).T[0], label="50m")
    c = Controller(200, 20)
    nominal = np.array(odeint(c.dxdt, c.x0, c.t))
    nominal = np.array([np.linalg.norm(i) for i in nominal])
    disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
    disturbed = np.array([np.linalg.norm(i) for i in disturbed])

    w = plt.plot(np.array(c.E_D_series).T[1], np.array(c.E_D_series).T[0], label="200m")
    c = Controller(1000, 20)
    nominal = np.array(odeint(c.dxdt, c.x0, c.t))
    nominal = np.array([np.linalg.norm(i) for i in nominal])
    disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
    disturbed = np.array([np.linalg.norm(i) for i in disturbed])

    n = plt.plot(np.array(c.E_D_series).T[1], np.array(c.E_D_series).T[0], label="1000")

    E_D_bound = c.x0.T * expm(c.A.T * c.tf) * np.linalg.inv(c.W) @ expm(c.A * c.tf) @ c.x0
    values, vectors = np.linalg.eig(np.linalg.inv(c.W))
    integrand = lambda x: expm(c.A * (c.tf - x)).max()
    print(quad(integrand, 0, c.tf))

    q_bar = w_bar * np.linalg.norm(vectors, ord=1, axis=0) * quad(integrand, 0, c.tf)
    sum = 0
    for value in values:
        sum += value

    E_D_bound += (2 * q_bar
                  * np.linalg.norm(np.diag(values) @ vectors.T @ expm(c.A * c.tf) @ c.x0, ord=1, axis=0)
                  + q_bar**2 * sum)
    z = plt.plot(E_D_bound, label="bound")


    # n = plt.plot(np.array(c.E_N_series).T[1], np.array(c.E_N_series).T[0], label="Nominal")

    # MM_series = []
    # AM_series = []
    #
    # for i in range(len(c.IC)):
    #     c.x0 = c.IC[i]
    #     nominal = np.array(odeint(c.dxdt, c.x0, c.t))
    #     disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
    #     MM_series.append(c.E_N / c.E_D)
    #     AM_series.append(c.E_D - c.E_N)

    # plt.plot(MM_series, label='E_N / E_D')
    # d = plt.plot(disturbed, label="Disturbed")
    # n = plt.plot(nominal, label="Nominal")
    plt.legend()
    plt.title("Energy Expenditure")
    plt.xlabel('Range')
    plt.ylabel('Energy')
    plt.savefig('./admire_figures_v2/energy_sinusoidal_w.png')
    plt.show()
