import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy import io


verbose = False  # whether to output debug information or not

class SubProblem():
    def __init__(self, alpha, beta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.P = None
        self.q = None
        self.A = None
        self.b = None
        self.x0 = None
        self.la = None
        self.v0 = None
        
        self.m, self.n = 0, 0
        
        self.dir = './Q2_data/'
        
    def f0(self, x):
        return 0.5 * x.T @ self.P @ x + self.q.T @ x
    
    def df0(self, x):
        return self.P @ x + self.q
    
    def hf0(self, x):
        return self.P
    
    def phi(self, x):
        return -np.sum(np.log(x))
    
    def dphi(self, x):
        return - 1 / x
    
    def hphi(self, x):
        return np.diag(np.power(1 / x, 2))

    def f(self, t, x):
        return t * self.f0(x) + self.phi(x)
    
    def gradient(self, t, x):
        return t * self.df0(x) + self.dphi(x)
    
    def hessian(self, t, x):
        return t * self.hf0(x) + self.hphi(x)

    def decrement(self, t, x, dx):
        return np.sqrt(dx.T @ self.hessian(t, x) @ dx)
    
    def r(self, t, x, v):
        gradient = self.gradient(t, x)
        r_dual = gradient + self.A.T @ v
        r_prim = self.A @ x - self.b
        return np.r_[r_dual, r_prim]

    def backtracking(self, t, x, dx):
        step = 1.0
        gradient = self.gradient(t, x)
        while True:
            if np.all(x + step * dx >= 0) and \
               self.f(t, x + step * dx) <= self.f(t, x) + self.alpha * step * np.inner(gradient, dx):
                return step
            else:
                step = self.beta * step
    
    def backtracking_norm(self, t, x, v, dx, dv):
        step = 1.0
        return 1 / (1 + self.decrement(t, x, dx))
        # if self.decrement(t, x, dx) < 1/4:
        #     return step
        while True:
            if np.all(x + step * dx >= 0) and \
               np.linalg.norm(self.r(t, x + step * dx, v + step * dv), ord=2) <= \
                (1 - self.alpha * step) * np.linalg.norm(self.r(t, x, v), ord=2):
                    return step
            else:
                step = self.beta * step
                
    def load_mat(self):
        self.P = io.loadmat(self.dir + 'P.mat')['P'].astype(np.double)
        self.q = io.loadmat(self.dir + 'q.mat')['q'].astype(np.double).squeeze()
        self.A = io.loadmat(self.dir + 'A.mat')['A'].astype(np.double)
        self.b = io.loadmat(self.dir + 'b.mat')['b'].astype(np.double).squeeze()
        self.x0 = io.loadmat(self.dir + 'x_0.mat')['x_0'].astype(np.double).squeeze()
        self.la = io.loadmat(self.dir + 'lambda.mat')['lambda'].astype(np.double).squeeze()
        self.v0 = io.loadmat(self.dir + 'mu.mat')['mu'].astype(np.double).squeeze()
        self.m, self.n = self.A.shape[0], self.P.shape[0]
    
    def solve(self, t, x0, v0):
        x = x0
        v = v0

        # feasible start Newton method
        if np.linalg.norm(self.A @ x - self.b) <= 1e-8:
            while True:
                hessian = self.hessian(t, x)
                lhs = np.block([[hessian, self.A.T], [self.A, np.zeros((self.m, self.m))]])
                rhs = np.r_[-self.gradient(t, x), np.zeros(self.m,)]
                dxw = np.linalg.inv(lhs) @ rhs
                dx, w = dxw[:self.n], dxw[self.n:]

                if verbose:
                    print('la: ', 0.5 * self.decrement(t, x, dx) ** 2)
                    print('r: ', np.linalg.norm(self.r(t, x, w)))

                if 0.5 * self.decrement(t, x, dx) ** 2 <= 1e-8:
                    v = w
                    break

                step = self.backtracking(t, x, dx)
                x = x + step * dx

        # infeasible start Newton method
        else:
            while True:
                hessian = self.hessian(t, x)
                lhs = np.block([[hessian, self.A.T], [self.A, np.zeros((self.m, self.m))]])
                rhs = self.r(t, x, v)
                dxv = - np.linalg.inv(lhs) @ rhs
                dx, dv = dxv[:self.n], dxv[self.n:]
                if verbose:
                    print(np.linalg.norm(rhs, ord=2))
                if np.linalg.norm(rhs, ord=2) <= 1e-8:
                    break
                step = self.backtracking_norm(t, x, v, dx, dv)
                # if np.linalg.norm(self.A @ x - self.b) > 1e-8:
                x = x + step * dx
                v = v + step * dv
        la = 1 / (t * x)
        v = v / t
        
        return x, la, v, self.f0(x)

    def pd_linesearch(self, x, la, v, t, dx, dla, dv):
        # determine the largest s
        idx = np.where(dla < 0)
        s = min(1, np.min(-la[idx] / dla[idx]))
        while True:
            if np.all(x + s * dx >= 0) and \
               np.linalg.norm(self.rt(x + s * dx, la + s * dla, v + s * dv, t), ord=2) <= \
               (1 - self.alpha * s) * np.linalg.norm(self.rt(x, la, v, t), ord=2):
                return s
            else:
                s = self.beta * s

    def rt(self, x, la, v, t):
        Df = -np.eye(self.n,)
        r_dual = self.df0(x) + Df.T @ la + self.A.T @ v
        r_cent = -np.diag(la) @ (-x) - (1 / t) * np.ones(self.n,)
        r_pri = self.A @ x - self.b
        rhs = np.r_[r_dual, r_cent, r_pri]
        return rhs

    # calculate primal-dual search direction
    def d_pd(self, x, la, v, t):
        hf0 = self.hf0(x)
        Df = -np.eye(self.n,)
        lhs = np.block([[hf0, Df.T, self.A.T],
                        [-np.diag(la) @ Df, np.diag(x), np.zeros((self.n, self.m))],
                        [self.A, np.zeros((self.m, self.n)), np.zeros((self.m, self.m))]])
        rhs = self.rt(x, la, v, t)

        return -np.linalg.inv(lhs) @ rhs

    def yita_hat(self, x, la):
        return x.T @ la

    def solve_pd(self, miu):
        # initialize x, lambda and miu
        x, la, v = self.x0, self.la, self.v0
        r_pri_list, r_dual_list, yita_list = [], [], []

        while True:
            # compute yita, t and search direction
            yita = self.yita_hat(x, la)
            t = miu * self.n / yita
            d_ypd = self.d_pd(x, la, v, t)
            
            # compute r_dual, r_cent and r_prim
            rt = self.rt(x, la, v, t)
            r_pri, r_dual = rt[self.n+self.n:], rt[0:self.n]

            r_pri_list.append(np.linalg.norm(r_pri))
            r_dual_list.append(np.linalg.norm(r_dual))
            yita_list.append(yita)

            if verbose:
                print('r_pri: ', np.linalg.norm(r_pri))
                print('r_dual: ', np.linalg.norm(r_dual))
                print('yita: ', yita)

            # stop criterion
            if np.linalg.norm(r_pri) <= 1e-8 and np.linalg.norm(r_dual) <= 1e-8 and yita <= 1e-8:
                return x, la, v, r_pri_list, r_dual_list, yita_list

            # compyte dx, dla and dv
            dx, dla, dv = d_ypd[0:self.n], d_ypd[self.n:self.n+self.n], d_ypd[self.n+self.n:]
            step = self.pd_linesearch(x, la, v, t, dx, dla, dv)

            # update x, la and v
            x = x + step * dx
            la = la + step * dla
            v = v + step * dv

class T2():
    def __init__(self, alpha, beta) -> None:
        self.subp = SubProblem(alpha=alpha,
                               beta=beta)
        self.miu = 10
        self.eps = 1e-8
        self.f_list = []
        self.t_list = []
        
        self.subp.load_mat()

    def solve_barrier(self, t0):
        self.f_list.clear()
        self.t_list.clear()
        t = t0
        x = self.subp.x0
        v = self.subp.v0
        while True:
            if verbose:
                print('t: ', t)
            self.t_list.append(t)
            # solve sub-problem
            x_star, la_star, v_star, f_star = self.subp.solve(t, x, v)
            self.f_list.append(f_star)
            # update
            x = x_star
            if self.subp.n / t < self.eps:
                return x_star, la_star, v_star, self.t_list
            else:
                t = self.miu * t

    def solve_pd(self):
        x_star, la_star, v_star, r_pri, r_dual, yita = self.subp.solve_pd(miu=self.miu)
        return x_star, la_star, v_star, r_pri, r_dual, yita

def plot_barrier(t_list):
    plt.figure()

    # plot curve
    plt.ylabel(r"$\log(\frac{n}{t})$")
    plt.xlabel(r"$k$")
    plt.plot(np.log(200 / np.array(t_list)), marker='o', markersize=2.0, color='coral', linewidth=1.0)
    plt.xlim([0, 11])
    plt.grid('on')
    plt.title(r"$\log(\hat{\eta})$" + ' versus ' + r"$k$")

    plt.show()

def plot_pd(r_pri, r_dual, yita):
    plt.figure()

    # plot curve 1
    plt.subplot(1, 2, 1)
    plt.ylabel(r"$\log(\hat{\eta})$")
    plt.xlabel(r"$k$")
    plt.plot(yita, marker='o', markersize=2.0, color='coral', linewidth=1.0)
    plt.xlim([0, 152])
    plt.grid('on')
    plt.semilogy()
    plt.title(r"$\log(\hat{\eta})$" + ' versus ' + r"$k$")

    # plot curve 2
    plt.subplot(1, 2, 2)
    plt.ylabel(r"$r_{feas}$")
    plt.xlabel(r"$k$")
    r_feas = np.sqrt(np.power(r_pri, 2) + np.power(r_dual, 2))
    plt.plot(r_feas, marker='o', markersize=2.0, color='coral', linewidth=1.0)
    plt.xlim([0, 152])
    plt.grid('on')
    plt.semilogy()
    plt.title(r"$r_{feas}$ versus $k$")

    plt.show()


if __name__ == '__main__':
    parse=argparse.ArgumentParser()
    parse.add_argument('-m', '--method', type=str, help='Solution method')
    parse.add_argument('-v', '--verbose', default=False, action="store_true", help='Verbose debugging information')
    args=parse.parse_args()

    alpha = 0.1
    beta = 0.5
    problem = T2(alpha=alpha,
                 beta=beta)

    if args.verbose:
        verbose = True
    
    if args.method == 'barrier':
        x_star, la_star, v_star, t_list = problem.solve_barrier(t0=1)
        print('x_star: ', x_star)
        print('la_star: ', la_star)
        print('v_star: ', v_star)
        print('p_star: ', problem.subp.f0(x_star))
        plot_barrier(t_list)
    elif args.method == 'pd':
        x_star, la_star, v_star, r_pri, r_dual, yita = problem.solve_pd()
        print('x_star: ', x_star)
        print('la_star: ', la_star)
        print('v_star: ', v_star)
        print('p_star: ', problem.subp.f0(x_star))
        plot_pd(r_pri, r_dual, yita)
