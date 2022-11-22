import numpy as np
from matplotlib import pyplot as plt
from scipy import io


verbose = False

class T1():
    def __init__(self, alpha, beta) -> None:
        self.P = None
        self.q = None
        self.A = None
        self.b = None
        self.LHS = None
        self.LHS_inv = None
        self.m = 0
        self.n = 0
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-5
        self.x_list = []
        self.v_list = []
        self.f_list = []
        self.t_list = []
        
        self.dir = './Q1_data/'
        self.load_mat()

    def f(self, x):
        return 0.5 * x.T @ self.P @ x + self.q.T @ x

    def gradient(self, x):
        return self.P @ x + self.q

    def hessian(self, x):
        return self.P
    
    def decrement(self, x, dx):
        return np.sqrt(dx.T @ self.hessian(x) @ dx)
    
    def r(self, x, v):
        r_dual = self.gradient(x) + self.A.T @ v
        r_prim = self.A @ x - self.b
        return np.r_[r_dual, r_prim]

    def backtracking(self, x, dx):
        t = 1.0
        gradient = self.gradient(x)
        while True:
            if self.f(x + t * dx) <= self.f(x) + self.alpha * t * np.inner(gradient, dx):
                return t
            else:
                t = self.beta * t
                
    def backtracking_norm(self, x, v, dx, dv):
        t = 1.0
        while True:
            if np.linalg.norm(self.r(x + t * dx, v + t * dv), ord=2) <= \
                (1 - self.alpha * t) * np.linalg.norm(self.r(x, v), ord=2):
                    return t
            else:
                t = self.beta * t

    def solve(self, x0, v0):
        self.x_list.clear()
        self.v_list.clear()
        self.f_list.clear()
        self.t_list.clear()
        x = x0
        v = v0
        self.t_list.append(0)
        while True:
            self.x_list.append(x.tolist())
            self.v_list.append(v.tolist())
            self.f_list.append(self.f(x))
            RHS = self.r(x, v)
            dxv = -self.LHS_inv @ RHS
            dx, dv = dxv[:self.n], dxv[self.n:]
            if self.decrement(x, dx) ** 2 < self.eps:
                break
            t = self.backtracking_norm(x, v, dx, dv)
            self.t_list.append(t)
            x = x + t * dx
            v = v + t * dv
        print('iters: ', len(self.x_list) - 1)
        print('x*: ', x.tolist())
        print('v*: ', v.tolist())
        print('p*: ', self.f(x))

        return self.x_list, self.v_list, self.f_list, self.t_list
    
    def load_mat(self):
        self.P = io.loadmat(self.dir + 'P.mat')['P']
        self.q = io.loadmat(self.dir + 'q.mat')['q'].squeeze()
        self.A = io.loadmat(self.dir + 'A.mat')['A']
        self.b = io.loadmat(self.dir + 'b.mat')['b'].squeeze()
        n, m = self.P.shape[0], self.A.shape[0]
        self.LHS = np.block([[self.P, self.A.T], [self.A, np.zeros((m, m))]])
        self.LHS_inv = np.linalg.inv(self.LHS)
        self.m, self.n = m, n

def plot_figure(f_list, t_list):
    plt.figure()

    # plot curve 1
    plt.subplot(1, 2, 1)
    plt.ylabel(r"$\log(f(x_k)-p^{*})$")
    plt.xlabel(r"$k$")
    plt.plot(np.log(f_list - (np.min(f_list) - 1e-10)), marker='o', markersize=2.0, color='coral', linewidth=1.0)
    plt.xlim([0, 1])
    plt.grid('on')
    plt.title(r"$\log(f(x_k)-p^{*})$ versus $k$")

    # plot curve 2
    plt.subplot(1, 2, 2)
    plt.ylabel(r"$t_k$")
    plt.xlabel(r"$k$")
    plt.plot(t_list, marker='o', markersize=2.0, color='coral', linewidth=1.0)
    plt.xlim([0, 1])
    plt.grid('on')
    plt.title(r"$t_k$ versus $k$")

    plt.show()


if __name__ == '__main__':
    alpha = 0.4
    beta = 0.5
    m, n = 100, 200
    problem = T1(alpha=alpha,
                 beta=beta)
    x_list, v_list, f_list, t_list = problem.solve(x0=np.zeros(n,), v0=np.zeros(m,))
    plot_figure(f_list, t_list)
