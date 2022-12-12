import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class T1():
    def __init__(self) -> None:
        self.data_dir = './problem1data/'
        self.alpha = 0.
        self.eps = 1e-5
        self.A = None
        self.b = None

    def load_matrix(self, idx):
        self.A = np.array(pd.read_csv(self.data_dir+'A%d.csv'%idx, header=None))
        self.b = np.array(pd.read_csv(self.data_dir+'b%d.csv'%idx, header=None)).squeeze()

    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)

    def proximal(self, x):
        return np.sign(x) * np.maximum(np.abs(x) - self.alpha, 0)

    def compute_step(self):
        self.alpha = 1 / np.max(np.linalg.svd(self.A.T @ self.A)[1])

    def f(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b, ord=2) ** 2 + np.linalg.norm(x, ord=1)

    def solve(self, x0):
        x_list = []
        f_list = []
        x = x0

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
            gradient = self.gradient(x)
            x = self.proximal(x - self.alpha * gradient)
            k = k + 1

def plot_figure(i, x_list, f_list):
    print(len(x_list))
    x_list, f_list = np.array(x_list), np.array(f_list)

    if i == 1:
        x_lim = 550.
    elif i == 2:
        x_lim = 69000.

    plt.figure()
    plt.plot(np.linalg.norm(x_list[:-1, :] - x_list[-1, :], ord=2, axis=1))
    plt.ylabel(r"$\Vert x_{k+1}-x^{*} \Vert$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$\Vert x_{k+1}-x^{*} \Vert$ versus $k$")
    plt.xlim([0, x_lim])
    plt.grid('on')
    plt.savefig('./problem1data/%d-1.pdf'%1)

    plt.figure()
    plt.plot(np.abs(f_list[:-1] - f_list[1:]))
    plt.ylabel(r"$h(x_{k-1})-h(x_k)$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$h(x_{k-1})-h(x_k) versus k$")
    plt.xlim([0, x_lim])
    plt.grid('on')
    plt.savefig('./problem1data/%d-2.pdf'%i)

    plt.figure()
    plt.plot(f_list[:-1] - f_list[-1])
    plt.ylabel(r"$h(x_k)$")
    plt.xlabel(r"$k$")
    plt.semilogx()
    plt.semilogy()
    plt.title(r"$h(x_k)$ versus $k$")
    plt.xlim([0, x_lim])
    plt.grid('on')
    plt.savefig('./problem1data/%d-3.pdf'%i)

    plt.show()


if __name__ == '__main__':
    parse=argparse.ArgumentParser()
    parse.add_argument('-i', '--index', type=int, default=1, help='Data Index')
    args=parse.parse_args()

    # numerical experiment
    problem = T1()
    problem.load_matrix(idx=args.index)
    problem.compute_step()

    x0 = np.zeros((problem.A.shape[1]))
    x_list, f_list = problem.solve(x0)

    # plot figures
    plot_figure(args.index, x_list, f_list)
