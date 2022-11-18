import numpy as np
from matplotlib import pyplot as plt
from scipy import io


class T3():
    def __init__(self, file='A_50') -> None:
        self.mat = None
        self.dir = './data/'
        self.eps = 1e-5
        self.x_list = []
        self.f_list = []
        self.t_list = []

        self.load_mat(file=file)

    def f(self, x):
        return -np.sum(np.log(1 - self.mat @ x)) - np.sum(np.log(1 - np.power(x, 2)))

    def gradient(self, x):
        den = 1 - self.mat @ x
        frac = np.divide(self.mat, den.reshape(-1, 1))
        gradient = 2 * x / (1 - np.power(x, 2)) + np.sum(frac, axis=0)

        return gradient

    def hessian(self, x):
        den = np.power(1 - self.mat @ x, 2)
        mat2 = np.power(self.mat, 2)
        frac = np.divide(mat2, den.reshape(-1, 1))
        hessian_ = (2 + 2 * np.power(x, 2)) / np.power(1 - np.power(x, 2), 2)
        hessian1 = np.diag(hessian_)
        hessian2 = np.zeros_like(hessian1)
        for k in range(self.mat.shape[0]):
            ak = self.mat[k, :].reshape(-1, 1)
            hessian2 += ak @ ak.T / (1 - np.inner(ak.squeeze(), x)) ** 2

        return hessian1 + hessian2

    def load_mat(self, file='A_50'):
        mat = io.loadmat(self.dir+file)
        self.mat = mat[file]

    def getstep(self, la):
        return 1 / (1 + la)

    def solve(self, x0):
        self.x_list.clear()
        x = np.array(x0)
        while True:
            self.x_list.append(x.tolist())
            self.f_list.append(self.f(x))
            hessian = self.hessian(x)
            gradient = self.gradient(x)
            hessian_inv = np.linalg.inv(hessian)
            dx = -hessian_inv @ gradient
            la = np.sqrt(gradient.T @ hessian_inv @ gradient)
            if len(self.x_list) == 1:
                print('dx([0...0]): ', dx)
                print('la([0...0]): ', la)
            if la ** 2 < self.eps:
                break
            t = self.getstep(la)
            self.t_list.append(t)
            x = x + t * dx
        print('iters: ', len(self.x_list) - 1)
        print('x*: ', x.tolist())
        print('p*: ', self.f(x))

        return self.x_list, self.f_list, [0] + self.t_list

def plot_value_and_step(X, F, T):
    # plot value curve
    p_star = F[-1]
    plt.subplot(1, 2, 1)
    plt.plot(np.array(F[:-1]) - p_star, marker='o', markersize=2.0, color='coral', linewidth=1.0)
    
    plt.xlabel(r"$k$")
    plt.ylabel(r"$log(f(x_k))$")
    plt.xticks(np.arange(0, 46, 2))
    plt.xlim([0, 46])
    # plt.xticks(np.arange(0, 30, 2))
    # plt.xlim([0, 30])
    plt.grid('on')
    plt.semilogy()
    plt.title('function value versus iterations k')

    plt.subplot(1, 2, 2)
    plt.plot(T, marker='o', markersize=2.0, color='coral', linewidth=1.0)
    
    plt.xlabel(r"$k$")
    plt.ylabel(r"$t$")
    plt.xticks(np.arange(0, 46, 2))
    plt.xlim([0, 46])
    # plt.xticks(np.arange(0, 30, 2))
    # plt.xlim([0, 30])
    plt.grid('on')
    plt.title('step length versus iterations k')

    plt.show()

if __name__ == '__main__':
    # m, n = 50, 50
    # file = 'A_50'
    m, n = 100, 100
    file = 'A_100'
    problem = T3(file=file)
    x_list, f_list, t_list = problem.solve(x0 = np.zeros(n,))
    plot_value_and_step(x_list, f_list, t_list)
    print(problem.f(-0.5*np.ones(50,)))
