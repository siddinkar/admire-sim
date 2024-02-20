import numpy as np
from scipy.integrate import odeint, quad, quad_vec
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm


class Controller:
    def __init__(self, R, duration):
        self.IC = np.linspace(np.ones(3).T / np.sqrt(3), np.ones(3).T * R / np.sqrt(3), R)
        # self.x0 = np.ones(3).T * R / np.sqrt(3)
        self.x0 = self.IC[0]
        self.t = np.linspace(0, duration - 1, duration)
        self.Q = np.eye(3) / 1000.0
        self.w = lambda x: np.array([np.tanh(x), 5 * np.sin(x), np.cos(x) * np.exp(-x / 5)]).T

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
        self.E_N_series.append(self.E_N)
        x_ = self.A @ x + self.B @ u_N
        self.prev_t_N = timestep
        return x_

    def dxdt_disturbed(self, x, timestep):
        # w = np.random.multivariate_normal((0, 0, 0), self.Q).T
        u_D = -self.B.T @ expm(self.A.T * (self.tf - timestep)) @ np.linalg.inv(self.W) @ expm(
            self.A * self.tf) @ self.x0
        u_D += -self.B.T @ expm(self.A.T * (self.tf - timestep)) @ np.linalg.inv(self.W) @ self.R()
        self.E_D += (u_D.T @ u_D) * (timestep - self.prev_t_D)
        self.E_D_series.append(self.E_D)
        x_ = self.A @ x + self.B @ u_D + self.w(timestep)
        self.prev_t_D = timestep
        return x_


if __name__ == "__main__":
    c = Controller(100, 20)
    # nominal = np.array(odeint(c.dxdt, c.x0, c.t))
    # nominal = np.array([np.linalg.norm(i) for i in nominal])
    # disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
    # disturbed = np.array([np.linalg.norm(i) for i in disturbed])

    # c.E_N_series = np.array([c.E_N_series[i] for i in range(0, 199, 10)])
    # c.E_D_series = np.array([c.E_D_series[i] for i in range(0, 343, 17)])

    # ed = plt.plot(c.E_D_series, label="Disturbed")
    # en = plt.plot(c.E_N_series, label="Nominal")
    # d = plt.plot(c.E_D_series, label="Disturbed")
    # n = plt.plot(c.E_N_series, label="Nominal")

    MM_series = []
    AM_series = []

    for i in range(len(c.IC)):
        c.x0 = c.IC[i]
        nominal = np.array(odeint(c.dxdt, c.x0, c.t))
        disturbed = np.array(odeint(c.dxdt_disturbed, c.x0, c.t))
        MM_series.append(c.E_N / c.E_D)
        AM_series.append(c.E_D - c.E_N)

    plt.plot(AM_series, label='E_D - E_N')
    plt.legend()
    plt.title("Energy Metric")
    plt.xlabel('Range')
    plt.ylabel('Metric')
    plt.savefig('./figures/AM.png')
    plt.show()
