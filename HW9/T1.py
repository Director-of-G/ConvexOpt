import numpy as np
from matplotlib import pyplot as plt


class T1(object):
    def __init__(self, alpha, beta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.A = np.array([[1, 0],
                           [0, 100]])
        self.x0 = np.array([100, 1])
        self.eps = 1e-8
        
    def get_function_value(self, x):
        return 0.5 * x.T @ self.A @ x
        
    def get_gradient(self, x):
        return self.A @ x

    def backtracking(self, x, dx):
        t = 1.0
        gradient = self.get_gradient(x)
        while True:
            if self.get_function_value(x + t * dx) <= \
                self.get_function_value(x) + self.alpha * t * np.inner(gradient, dx):
                return t
            else:
                t = self.beta * t

    def gradient_descent(self, method='constant step', step=1):
        x = self.x0
        x_list, f_list = [], []
        if method == 'constant step':
            while True:
                x_list.append(x.reshape(-1,).tolist())
                f_list.append(self.get_function_value(x).squeeze())
                grad = self.get_gradient(x)
                if (np.linalg.norm(grad, ord=2) <= self.eps):
                    break
                dx = -grad
                x = x + step * dx
        elif method == 'exact search':
            while True:
                x_list.append(x.reshape(-1,).tolist())
                f_list.append(self.get_function_value(x).squeeze())
                grad = self.get_gradient(x)
                if (np.linalg.norm(grad, ord=2) <= self.eps):
                    break
                dx = -grad
                step = (x.T @ self.A @ self.A @ x) / (x.T @ self.A @ self.A @ self.A @ x)
                x = x + step * dx
        elif method == 'backtracking':
            while True:
                x_list.append(x.reshape(-1,).tolist())
                f_list.append(self.get_function_value(x).squeeze())
                grad = self.get_gradient(x)
                if (np.linalg.norm(grad, ord=2) <= self.eps):
                    break
                dx = -grad
                step = self.backtracking(x, dx)
                x = x + step * dx
        else:
            print('Wrong method assigned!')
                
        return x_list, f_list
                
def plot_contour_and_curve(X, F):
    # generate raw data for contour line
    x_grid = np.arange(-1, 101, 0.1)
    y_grid = np.arange(-15, 15, 0.1)
    x, y = np.meshgrid(x_grid, y_grid)
    z = 0.5 * (1 * x**2 + 100 * y**2)
    plt.figure()

    # plot contour line
    plt.subplot(1, 2, 1)
    cont = plt.contour(x, y, z, [50, 200] + np.arange(500, 7500, 500).tolist())
    plt.clabel(cont, inline=True, fontsize=5)
    X = np.array(X)
    plt.plot(X[:, 0], X[:, 1], marker='o', markersize=2.0, color='coral', linewidth=1)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('contour line and iteration X path')

    # plot value curve
    plt.subplot(1, 2, 2)
    plt.plot(F, marker='o', markersize=1.0, color='coral', linewidth=0.5)
    plt.xlim([0, 974])
    
    plt.xlabel("k")
    plt.ylabel(r"$log(f(x_k))$")
    plt.grid('on')
    plt.semilogy()
    plt.title('function value with iterations k')

    plt.show()
                
                
if __name__ == '__main__':
    alpha = 0.4
    beta = 0.5
    problem = T1(alpha=alpha,
            beta=beta)
    x_list, f_list = problem.gradient_descent(method='backtracking', step=2/101)
    print("迭代次数(包含初始点): ", len(x_list))
    plot_contour_and_curve(np.array(x_list), np.array(f_list))
