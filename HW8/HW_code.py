from symtable import Symbol
from matplotlib import pyplot as plt
import numpy as np
from sympy import *
import pdb

"""
    f(x, y) = (1 - x)^2 + 2*(x^2 - y)^2
    df(x, y) = (8x^3 + 2x - 8xy -2, -4x^2 + 4y)
"""

def plot_contour_and_curve(X, F):
    # generate raw data for contour line
    x_grid = np.arange(-0.5, 1.5, 0.01)
    y_grid = np.arange(-0.5, 1.5, 0.01)
    x, y = np.meshgrid(x_grid, y_grid)
    z = (1 - x) ** 2 + 2 * (x ** 2 - y) ** 2
    plt.figure()

    # plot contour line
    plt.subplot(1, 2, 1)
    cont = plt.contour(x, y, z, [0.005, 0.01, 0.02, 0.05, 0.1] + np.arange(0.2, 2, 0.2).tolist())
    plt.clabel(cont, inline=True, fontsize=5)
    X = np.array(X)
    plt.plot(X[:, 0], X[:, 1], marker='o', markersize=4, color='coral', linewidth=1)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('contour line and iteration X path')

    # plot value curve
    plt.subplot(1, 2, 2)
    plt.plot(F, marker='o', markersize=4, color='coral', linewidth=1)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.title('value curve')

    plt.show()

# get function value f(x, y)
def f(X):
    x, y = X
    return (1 - x) ** 2 + 2 * (x ** 2 - y) ** 2

# get the partial derivative of df(x, y)/d(x, y)
def df(X):
    x, y = X
    return [8 * x ** 3 + 2 * x - 8 * x * y - 2, -4 * x ** 2 + 4 * y]

# nonlinear optimization with 'l1', 'l2', 'linf' types of norm
def nonlinear_optimazation(initial=np.array([0, 0]), method='linf'):
    X = initial.astype(np.float64)
    total_iters = 1
    X_list = []  # list for solution X
    F_list = []  # list for value f(X)
    while np.linalg.norm(df(X), ord=2) > 1e-4:  # stop criterion
        print('> iteration: %d' % total_iters)
        X_list.append(X.tolist())
        F_list.append(f(X))
        total_iters += 1
        # descending direction
        df_value = np.array(df(X))
        if method == 'l1':
            d = np.sign(-df_value) * (np.abs(df_value) == np.linalg.norm(df_value, ord=np.inf))
            d = d / np.linalg.norm(d, ord=1)
        elif method == 'l2':
            d = np.sign(-df_value) * np.abs(df_value) * (1 / np.linalg.norm(df_value, ord=2))
            d = d / np.linalg.norm(d, ord=2)
        elif method == 'linf':
            d = np.sign(-df_value)
            d = d / np.linalg.norm(d, ord=np.inf)

        # 1-dimentional precise search
        t = Symbol('t')
        x, y = X
        x_, y_ = x + t * d[0], y + t * d[1]
        dx, dy = df((x_, y_))
        solution = solve([dx * d[0] + dy * d[1]], [t])  # solve the monadic equation with respect to t
        if isinstance(solution, list):
            real_solution = []
            # get the real number (if there exactly exists a real solution)
            for ti in solution:
                ti_cplx = complex(ti[0])
                if ti_cplx.imag == 0 and ti_cplx.real > 0:
                    real_solution.append(ti_cplx.real)
            # if failed, get the complex with positive real part and smaller image part
            if len(real_solution) == 0:
                for ti in solution:
                    ti_cplx = complex(ti[0])
                    if ti_cplx.real > 0:
                        t = ti_cplx.real
                        break
            else:
                t = np.min(real_solution)  # get the smallest real number
        # if the solution is unique, directly assign t with it (if there exactly exists a real solution)
        elif isinstance(solution, dict):
            t = solution[t]

        # update X
        X = (X + t * d).astype(np.float64)
    
    print('Optimized solution: ', X.tolist())
    print('Optimized value: ', f(X))
    plot_contour_and_curve(X_list, F_list)

def conjugate_gradient(initial=np.array([0, 0]), method='FR'):
    X = initial.astype(np.float64)
    total_iters = 1
    X_list = []  # list for solution X
    F_list = []  # list for value f(X)
    D_last, X_last = 0, 0
    X_list.append(X.tolist())
    F_list.append(f(X))
    while np.linalg.norm(df(X), ord=2) > 1e-4:  # stop criterion
        print('> iteration: %d' % total_iters)
        
        # descending direction
        df_value = np.array(df(X))
        if total_iters == 1:
            D = -df_value
        else:
            if method == 'FR':
                alpha = np.linalg.norm(df(X), ord=2) / np.linalg.norm(df(X_last), ord=2)
                alpha = alpha ** 2
            elif method == 'PR':
                df_Xk = np.array(df(X))
                df_Xk_last = np.array(df(X_last))
                alpha = (df_Xk.reshape(1, 2) @ (df_Xk - df_Xk_last)) / (np.linalg.norm(df_Xk_last, ord=2) ** 2)
            D = -df_value + alpha * D_last

        # 1-dimentional precise search
        t = Symbol('t')
        x, y = X
        x_, y_ = x + t * D[0], y + t * D[1]
        dx, dy = df((x_, y_))
        solution = solve([dx * D[0] + dy * D[1]], [t])  # solve the monadic equation with respect to t
        if isinstance(solution, list):
            real_solution = []
            # get the real number (if there exactly exists a real solution)
            for ti in solution:
                ti_cplx = complex(ti[0])
                if ti_cplx.imag == 0 and ti_cplx.real > 0:
                    real_solution.append(ti_cplx.real)
            # if failed, get the complex with positive real part and smaller image part
            if len(real_solution) == 0:
                for ti in solution:
                    ti_cplx = complex(ti[0])
                    if ti_cplx.real > 0:
                        t = ti_cplx.real
                        break
            else:
                t = np.min(real_solution)  # get the smallest real number
        # if the solution is unique, directly assign t with it (if there exactly exists a real solution)
        elif isinstance(solution, dict):
            t = solution[t]

        # update X
        X_last = X
        X = (X + t * D).astype(np.float64)
        # record X and f(X)
        X_list.append(X.tolist())
        F_list.append(f(X))

        D_last = D
        total_iters += 1

    print('Optimized solution: ', X.tolist())
    print('Optimized value: ', f(X))
    plot_contour_and_curve(X_list, F_list)
    

if __name__ == '__main__':
    # solve optimization problem with 'l1', 'l2' or 'linf' types of norm
    nonlinear_optimazation(initial=np.array([0, 0]), method='l2')

    # solve optimization problem with 'FR' or 'PR' conjugate gradient methods
    conjugate_gradient(initial=np.array([0, 0]), method='FR')
