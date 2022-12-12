import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class T1():
    def __init__(self, alpha) -> None:
        self.data_dir = './week_13_data/'
        self.alpha = alpha
        self.eps = 1e-5
        self.A = None
        self.b = None

    def load_matrix(self):
        self.A = np.array(pd.read_csv(self.data_dir+'A.csv', header=None))
        self.b = np.array(pd.read_csv(self.data_dir+'b.csv', header=None)).squeeze()

    def soft_threshold(self, x, lamda):
        return np.sign(x) * np.maximum(np.abs(x) - lamda, 0)

    def f(self, x):
        return np.linalg.norm(x, ord=1)

    def solve(self, x0):
        m, n = self.A.shape
        u, v = np.zeros(n,), np.zeros(m,)
        x, y = np.ones(n,), np.ones(n,)
        x_list = []
        f_list = []

        k = 1
        while True:
            x_list.append(x.tolist())
            f = self.f(x)
            f_list.append(f)

            if len(x_list) >= 2:
                print(np.abs(f_list[-1] - f_list[-2]))
                if np.abs(f_list[-1] - f_list[-2]) < self.eps:
                    return x_list, f_list
            
            # iterate
            x = self.soft_threshold(y - (1 / self.alpha) * u, (1 / self.alpha))
            y = np.linalg.inv(np.eye(n) + self.A.T @ self.A) @ (x + (1 / self.alpha) * u + self.A.T @ (self.b  - (1 / self.alpha) * v))
            u = u + self.alpha * (x - y)
            v = v + self.alpha * (self.A @ y - self.b)
            k = k + 1

def plot_figure(x_list, f_list):
    x_list, f_list = np.array(x_list), np.array(f_list)

    x_lim = 195

    plt.figure()
    plt.plot(np.linalg.norm(x_list[:-1, :] - x_list[-1, :], ord=2, axis=1))
    plt.ylabel(r"$\Vert x_{k}-x^{*} \Vert_{2}$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$\Vert x_{k}-x^{*} \Vert_{2}$ versus $k$")
    plt.xlim([0, x_lim])
    plt.grid('on')
    plt.savefig('./week_13_data/1.pdf')

    plt.figure()
    plt.plot(np.abs(f_list[:-1] - f_list[-1]))
    plt.ylabel(r"$\Vert x_k \Vert_{1}-\Vert x^{*} \Vert_{1}$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$\Vert x_k \Vert_{1}-\Vert x^{*} \Vert_{1}$ versus $k$")
    plt.xlim([0, x_lim])
    plt.grid('on')
    plt.savefig('./week_13_data/2.pdf')

    plt.show()


if __name__ == '__main__':
    # numerical experiment
    problem = T1(alpha=1.)
    problem.load_matrix()

    x0 = np.zeros((problem.A.shape[1]))
    x_list, f_list = problem.solve(x0)

    # plot figures
    plot_figure(x_list, f_list)
