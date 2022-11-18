import numpy as np
import math
from matplotlib import pyplot as plt

class QP(object):
    def __init__(self, A, x0, yita) -> None:
        self.A = A
        self.x0 = x0.reshape(-1, 1)
        self.yita = yita
        
    def get_function_value(self, x):
        return 0.5 * x.T @ self.A @ x
        
    def get_gradient(self, x):
        return self.A @ x

    def gradient_descent(self, method='constant step', step=1):
        x = self.x0
        x_list, f_list = [], []
        if method == 'constant step':
            while True:
                x_list.append(x.reshape(-1,).tolist())
                f_list.append(self.get_function_value(x).squeeze())
                grad = self.get_gradient(x)
                if (np.linalg.norm(grad, ord=2) <= self.yita):
                    break
                dx = -grad
                x = x + step * dx
        elif method == 'exact search':
            while True:
                x_list.append(x.reshape(-1,).tolist())
                f_list.append(self.get_function_value(x).squeeze())
                grad = self.get_gradient(x)
                if (np.linalg.norm(grad, ord=2) <= self.yita):
                    break
                dx = -grad
                step = (x.T @ self.A @ self.A @ x) / (x.T @ self.A @ self.A @ self.A @ x)
                x = x + step * dx
                
        return x_list, f_list
                
    def plot_contour_and_curve(self, X, F):
        # generate raw data for contour line
        x_grid = np.arange(-1, 101, 0.1)
        y_grid = np.arange(-15, 15, 0.1)
        x, y = np.meshgrid(x_grid, y_grid)
        z = 0.5 * (self.A[0, 0] * x**2 + self.A[1, 1] * y**2)
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
        plt.xlim([0, 1169])
        
        plt.xlabel("k")
        plt.ylabel(r"$log(f(x_k))$")
        plt.grid('on')
        plt.semilogy()
        plt.title('function value with iterations k')

        plt.show()
                
                
if __name__ == '__main__':
    qp = QP(A = np.array([[1, 0], [0, 100]]), x0=np.array([100, 1]), yita=1e-8)
    x_list, f_list = qp.gradient_descent(method='constant step', step=2/101)
    print("迭代次数(包含初始点): ", len(x_list))
    qp.plot_contour_and_curve(np.array(x_list), np.array(f_list))
