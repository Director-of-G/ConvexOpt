import numpy as np
from matplotlib import pyplot as plt


class T2():
    def __init__(self, alpha, beta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8
        self.x_list = []
        self.f_list = []

    def f(self, x):
        x1, x2 = x[0], x[1]
        return (10 * x1 ** 2 + x2 ** 2) / 2 + 5 * np.log(1 + np.exp(-x1 - x2))

    def gradient(self, x):
        x1, x2 = x[0], x[1]
        exp = np.exp(-x1 - x2)
        D1 = 10 * x1 - 5 * (exp / (1 + exp))
        D2 = x2 - 5 * (exp / (1 + exp))
        return np.array([D1, D2])

    def hessian(self, x):
        x1, x2 = x[0], x[1]
        exp = np.exp(-x1 - x2)
        H11 = 10 + 5 * (exp / (1 + exp) ** 2)
        H12 = 5 * (exp / (1 + exp) ** 2)
        H21 = 5 * (exp / (1 + exp) ** 2)
        H22 = 1 + 5 * (exp / (1 + exp) ** 2)
        return np.array([[H11, H12],
                         [H21, H22]])

    def backtracking(self, x, dx):
        t = 1.0
        gradient = self.gradient(x)
        while True:
            if self.f(x + t * dx) <= self.f(x) + self.alpha * t * np.inner(gradient, dx):
                return t
            else:
                t = self.beta * t

    def solve(self, x0):
        self.x_list.clear()
        x = np.array(x0)
        while True:
            self.x_list.append(x.tolist())
            self.f_list.append(self.f(x))
            hessian = self.hessian(x)
            gradient = self.gradient(x)
            dx = -np.linalg.inv(hessian) @ gradient
            if np.linalg.norm(gradient, ord=2) < self.eps:
                break
            t = self.backtracking(x, dx)
            x = x + t * dx
        print('x: ', self.x_list)
        print('iters: ', len(self.x_list) - 1)
        print('x*: ', x.tolist())
        print('p*: ', self.f(x))

        return self.x_list, self.f_list

def plot_contour_and_curve(X, F):
    # generate raw data for contour line
    x_grid = np.arange(-2, 2, 0.01)
    y_grid = np.arange(-1, 3, 0.01)
    x, y = np.meshgrid(x_grid, y_grid)
    z = (10 * x ** 2 + y ** 2) / 2 + 5 * np.log(1 + np.exp(-x - y))
    plt.figure()

    # plot contour line
    plt.subplot(1, 2, 1)
    cont = plt.contour(x, y, z, [1.0, 1.5, 2.0, 2.1, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    plt.clabel(cont, inline=True, fontsize=5)
    X = np.array(X)
    plt.plot(X[:, 0], X[:, 1], marker='o', markersize=2.0, color='coral', linewidth=1)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect(1)
    plt.title('contour line and iteration X path')

    # plot value curve
    plt.subplot(1, 2, 2)
    plt.plot(np.log(np.array(F)), marker='o', markersize=2.0, color='coral', linewidth=1.0)
    
    plt.xlabel("k")
    plt.ylabel(r"$log(f(x_k))$")
    plt.grid('on')
    # plt.semilogy()
    plt.title('function value with iterations k')

    plt.show()


if __name__ == '__main__':
    alpha = 0.4
    beta = 0.5
    problem = T2(alpha=alpha,
                 beta=beta)
    X, F = problem.solve(x0=[0, 0])
    print(F)
    plot_contour_and_curve(X, F)
