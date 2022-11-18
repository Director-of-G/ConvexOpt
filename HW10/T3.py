from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt


verbose = False

class T3():
    def __init__(self, alpha, beta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8
        self.A = np.diag([1, 100]).astype(np.double)
        self.x0 = np.array([100, 1]).astype(np.double)
        self.x_list = []
        self.f_list = []

    def f(self, x):
        return 0.5 * x.T @ self.A @ x

    def df(self, x):
        return self.A @ x

    def hf(self, x):
        return self.A

    def solve_hb(self):
        x0 = self.x0
        self.x_list.clear()
        self.f_list.clear()
        x = x0
        k = 0
        while True:
            # update variable and function value
            self.x_list.append(x.tolist())
            self.f_list.append(self.f(x))

            # x_{k} - x_{k-1}
            if k == 0:
                delta_x = np.zeros_like(x0)
            else:
                delta_x = x - np.array(self.x_list[-2])

            # calculate gradient
            gradient = self.df(x)

            # stop criterion
            if np.linalg.norm(gradient, ord=2) < self.eps:
                return deepcopy(self.x_list), deepcopy(self.f_list)

            # calculate new x
            x = x - self.alpha * gradient + self.beta * delta_x

            # update iteration step k
            k = k + 1

    def solve_gd(self):
        x0 = self.x0
        self.x_list.clear()
        self.f_list.clear()
        x = x0
        while True:
            # update variable and function value
            self.x_list.append(x.tolist())
            self.f_list.append(self.f(x))

            # calculate gradient
            gradient = self.df(x)

            # stop criterion
            if np.linalg.norm(gradient, ord=2) < self.eps:
                return deepcopy(self.x_list), deepcopy(self.f_list)

            # calculate new x
            x = x - (2 / 101) * gradient

def plot_contour_and_curve(X1, X2, F1, F2):
    A = np.diag([1, 100]).astype(np.double)

    # generate raw data for contour line
    x_grid = np.arange(-1, 101, 0.1)
    y_grid = np.arange(-15, 15, 0.1)
    x, y = np.meshgrid(x_grid, y_grid)
    z = 0.5 * (A[0, 0] * x**2 + A[1, 1] * y**2)
    plt.figure()

    # plot contour line
    plt.subplot(1, 2, 1)
    cont = plt.contour(x, y, z, [50, 200] + np.arange(500, 7500, 500).tolist())
    plt.clabel(cont, inline=True, fontsize=5)
    X1, X2 = np.array(X1), np.array(X2)
    plt.plot(X1[:, 0], X1[:, 1], marker='o', markersize=2.0, color='coral', linewidth=1)
    plt.plot(X2[:, 0], X2[:, 1], marker='o', markersize=2.0, color='deepskyblue', linewidth=1)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('contour line and iteration X path')

    # plot value curve
    plt.subplot(1, 2, 2)
    plt.plot(F1, marker='o', markersize=1.0, color='coral', linewidth=0.5)
    plt.plot(F2, marker='o', markersize=1.0, color='deepskyblue', linewidth=0.5)
    plt.xlim([0, 114])
    
    plt.xlabel("k")
    plt.ylabel(r"$log(f(x_k))$")
    plt.grid('on')
    plt.semilogy()
    plt.title('function value with iterations k')

    plt.show()


if __name__ == '__main__':
    alpha = 4 / 121
    beta = 81 / 121
    problem = T3(alpha=alpha,
                 beta=beta)
    x_list1, f_list1 = problem.solve_hb()
    x_list2, f_list2 = problem.solve_gd()
    plot_contour_and_curve(x_list1, x_list2, f_list1, f_list2)
