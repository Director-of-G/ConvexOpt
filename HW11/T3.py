import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class T2():
    def __init__(self, c, beta) -> None:
        self.data_dir = './data/'
        self.beta = beta
        self.c = c
        self.eps = 1e-8
        self.A = None
        self.b = None

        self.load_matrix()

    def load_matrix(self):
        self.A = np.array(pd.read_csv(self.data_dir+'2A.csv', header=None))
        self.b = np.array(pd.read_csv(self.data_dir+'2b.csv', header=None)).squeeze()

    def subgradient(self, x):
        return self.A.T @ (self.A @ x - self.b) + np.sign(x)

    def f(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b, ord=2) ** 2 + np.linalg.norm(x, ord=1)

    def solve(self, x0, method='1'):
        x_list = []
        f_list = []
        x = x0

        if method == '2':
            m = np.min(np.linalg.svd(self.A)[1]) ** 2

        k = 1
        while True:
            x_list.append(x.tolist())
            f = self.f(x)
            f_list.append(f)
            if len(x_list) >= 2:
                print(np.linalg.norm(np.array(x_list[-1]) - np.array(x_list[-2]), ord=2) ** 2)
            if len(x_list) >= 2 and np.linalg.norm(np.array(x_list[-1]) - np.array(x_list[-2]), ord=2) ** 2 < self.eps:
                return x_list, f_list
            
            # iterate
            if method == '1':
                alpha = self.c * (k ** -(self.beta))
            elif method == '2':
                alpha = 1 / (m * k)
            x = x - alpha * self.subgradient(x)
            k = k + 1

def plot_figure(x_list, f_list):
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.plot(np.linalg.norm(x_list[1:, :] - x_list[:-1, :], ord=2, axis=1))
    plt.ylabel(r"$\Vert x_{k+1}-x_k \Vert$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$\Vert x_{k+1}-x_k \Vert$ versus $k$")
    plt.grid('on')

    plt.subplot(1, 3, 2)
    plt.plot(np.linalg.norm(x_list[:-1, :] - x_list[-1, :], ord=2, axis=1))
    plt.ylabel(r"$\Vert x_k-x* \Vert$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$\Vert x_k-x* \Vert$ versus $k$")
    plt.grid('on')

    plt.subplot(1, 3, 3)
    plt.plot(f_list)
    plt.ylabel(r"$h(x_k)$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$h(x_k)$ versus $k$")
    plt.grid('on')
    plt.show()


if __name__ == '__main__':
    parse=argparse.ArgumentParser()
    parse.add_argument('-m', '--method', type=int, default=1, help='Solution method')
    args=parse.parse_args()

    # numerical experiment
    problem = T2(c=0.01,
                 beta=0.5)
    x0 = np.array(pd.read_csv('./data/2x0.csv', header=None)).squeeze()
    x_list, f_list = problem.solve(x0, method=str(args.method))
    np.save('./data/2x.npy', np.array(x_list))
    np.save('./data/2f.npy', np.array(f_list))

    # plot figures
    x_list = np.load('./data/2x.npy')
    f_list = np.load('./data/2f.npy')
    plot_figure(x_list, f_list)
