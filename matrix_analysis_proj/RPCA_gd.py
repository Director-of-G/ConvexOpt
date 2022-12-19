import cv2
from copy import deepcopy
import numpy as np


alpha = 0.1
beta = 0.5


def f(A, E, lam):
    eig_sum = np.sum(np.linalg.svd(A)[1])
    norm_1 = np.linalg.norm(E, ord=1)

    return eig_sum + lam * norm_1

def grad_A(A, E):
    U, S_, VT = np.linalg.svd(A, full_matrices=False)
    S = np.zeros_like(S_)
    # dA = U @ S @ np.linalg.pinv(np.sqrt(S.T @ S)) @ VT
    dA = U @ VT

    return dA

def grad_E(A, E):
    max_j = np.argmax(np.sum(np.abs(E), axis=0))
    dE = np.zeros_like(E)
    dE[:, max_j] = np.sign(E[:, max_j])

    return dE

def project_gradient(A, E, lam):
    dA = grad_A(A, E)
    dE = lam * grad_E(A, E)
    dAE = np.concatenate((np.expand_dims(dA, axis=0), np.expand_dims(dE, axis=0)), axis=0)

    u = np.array([[1], [-1]])
    u = u / np.linalg.norm(u, ord=2)
    proj_mat = u @ u.T
    dAE = np.tensordot(proj_mat, dAE, axes=1)

    return dAE

def line_search(A, E, dA, dE, lam):
    step = 1.0
    while True:
        if f(A + step * dA, E + step * dE, lam) <= f(A, E, lam) \
            - alpha * step * (np.linalg.norm(dA, ord='fro') ** 2 + np.linalg.norm(dE, ord='fro') ** 2) or step < 1e-5:
            return step
        else:
            step = beta * step


def RPCA_solve(D):
    m, n = D.shape
    lam = 1 / np.sqrt(max(m, n))
    A = deepcopy(D)
    E = np.zeros_like(D)

    iters = 0

    while True:
        # import pdb; pdb.set_trace()
        dAE = project_gradient(A, E, lam)
        dA = -dAE[0, ...]
        dE = -dAE[1, ...]

        # print('norm: ', np.linalg.norm(dA, ord='fro') + np.linalg.norm(dE, ord='fro'))
        if (np.linalg.norm(dA, ord='fro') + np.linalg.norm(dE, ord='fro')) < 1e-8:
            return A, E

        # step = line_search(A, E, dA, dE, lam)
        step = 1 / 100
        iters = iters + 1
        # print('step: ', step)
        print('f: ', f(A, E, lam))

        if iters > 10000:
            # import pdb; pdb.set_trace()
            return A, E

        A = A + step * dA
        E = E + step * dE


if __name__ == '__main__':
    # D = np.random.rand(5, 6)
    # D = np.load('data.npy')
    # A, E = RPCA_solve(D)
    # print('A: ', A)
    # print('E: ', E)
    img = cv2.imread('./3.jpg')
    img = img.astype(np.float)
    img /= 255.
    nrow, ncol, nchn = img.shape
    img += 0.0 * np.random.rand(nrow, ncol, nchn)
    img_r = img[..., 0]
    img_g = img[..., 1]
    img_b = img[..., 2]
    A1, E1 = RPCA_solve(img_r)
    A2, E2 = RPCA_solve(img_g)
    A3, E3 = RPCA_solve(img_b)

    low_rank = np.concatenate((np.expand_dims(A1, 2), np.expand_dims(A2, 2), np.expand_dims(A3, 2)), axis=2)
    sparse = np.concatenate((np.expand_dims(E1, 2), np.expand_dims(E2, 2), np.expand_dims(E3, 2)), axis=2)
    
    # save matrix
    np.save('./data/gd/A3.npy', low_rank)
    np.save('./data/gd/E3.npy', sparse)

    cv2.imshow('low rank', np.concatenate((np.expand_dims(A1, 2), np.expand_dims(A2, 2), np.expand_dims(A3, 2)), axis=2))
    cv2.imshow('spase', np.concatenate((np.expand_dims(E1, 2), np.expand_dims(E2, 2), np.expand_dims(E3, 2)), axis=2))
    cv2.imshow('origin', img)
    cv2.waitKey(0)
