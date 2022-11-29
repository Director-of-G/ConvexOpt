import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class T2():
    def __init__(self, alpha) -> None:
        self.data_dir = './data/'
        self.alpha = alpha
        self.eps = 0.02
        self.A = None
        self.b = None

        self.load_matrix()

    def load_matrix(self):
        self.A = np.array(pd.read_csv(self.data_dir+'1A.csv', header=None))
        self.b = np.array(pd.read_csv(self.data_dir+'1b.csv', header=None)).squeeze()

    def f(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b, ord=2) ** 2

    def solve(self, x0):
        x_list = []
        f_list = []
        x = x0
        while True:
            x_list.append(x.tolist())
            f = self.f(x)
            f_list.append(f)
            print(f)
            if f <= self.eps:
                return x_list, f_list
            
            # iterate
            x = x + np.linalg.inv(self.A.T @ self.A + (1 / self.alpha)*np.eye(self.A.shape[1])) @ self.A.T @ (self.b - self.A @ x)


def plot_figure(x_list, f_list):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(np.linalg.norm(x_list[:-1, :] - x_list[-1, :], ord=2, axis=1))
    plt.ylabel(r"$\Vert x_k-x* \Vert$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.xlim([0, len(x_list)+1000])
    plt.title(r"$\Vert x_k-x* \Vert$ versus $k$")
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.plot(f_list[:-1])
    plt.ylabel(r"$f(x_k)$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.xlim([0, len(x_list)+1000])
    plt.title(r"$f(x_k)$ versus $k$")
    plt.grid('on')
    plt.show()


if __name__ == '__main__':
    # numerical experiment
    problem = T2(alpha=16.0)
    x_list, f_list = problem.solve(np.zeros(problem.A.shape[1]))
    np.save('./data/1x.npy', np.array(x_list))
    np.save('./data/1f.npy', np.array(f_list))

    # plot figures
    x_list = np.load('./data/1x.npy')
    f_list = np.load('./data/1f.npy')
    plot_figure(x_list, f_list)
