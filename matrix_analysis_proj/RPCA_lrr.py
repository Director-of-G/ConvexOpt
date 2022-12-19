import cv2
from copy import deepcopy
import numpy as np


def initialization(D):
    """
    Auto decide the hyperparameters
    :param D: The given data matrix.
    :return rho, miu, lam, thresh: The hyper-parameters.
    """
    m, n = D.shape
    lam = 1 / np.sqrt(max(m, n))

    norm_1 = np.linalg.norm(D, ord=1)
    norm_fro = np.linalg.norm(D, ord='fro')

    rho = (m * n) / (4 * norm_1)
    miu = 1.5
    thresh = 1e-7 * norm_fro

    return rho, miu, lam, thresh

def redecomposition(X, thresh):
    """
    Solution to proximal minimization with nuclear-norm.
    """
    U, S_, VT = np.linalg.svd(X)
    S = np.zeros_like(X)
    m, n = S.shape
    S[:min(m, n), :min(m, n)] = np.diag(S_)
    return U @ shrinkage(S, thresh) @ VT

def shrinkage(X, thresh):
    """
    The soft threshold function.
    Solution to proximal minimization with 1-norm.
    """
    return np.sign(X) * np.maximum(np.abs(X) - thresh, 0)

def segmentation(X, thresh):
    """
    Solution to the proximal minimization with {2,1}-norm
    """
    norm = np.linalg.norm(X, ord=2, axis=0)
    return np.maximum(np.sign(norm - thresh), 0) * ((norm - thresh) / norm) * X

def LRR_solve(M):
    """
    Solve the LRR problem.
    :param M: The given data matrix.
    :return A: Low rank matrix.
    :return E: Sparse matrix.
    """
    # initialize the primal and dual matrices
    D = deepcopy(M)
    X = deepcopy(M)
    A = np.zeros((M.shape[1], M.shape[1]))
    B = np.zeros((M.shape[1], M.shape[1]))
    E = np.zeros_like(M)
    Y1 = np.zeros_like(M)
    Y2 = np.zeros((M.shape[1], M.shape[1]))
    Y3 = np.zeros_like(M)

    # decide the hyper-parameters
    rho, miu, lam, thresh = initialization(D)
    # allow modification
    lam = lam * 1.0
    rho = 1e-7  # the bigger, converge the faster, low rank the clearer
    rho_max = 1e8
    miu = 1.1

    while True:
        # alternative direction descend
        B = redecomposition(A + Y2 / rho, 1 / rho)
        A = np.linalg.inv(X.T @ X + np.eye(M.shape[1])) @ (X.T @ (D - E + Y1 / rho) + (B - Y2 / rho))
        D = M
        E = segmentation(D - X @ A + Y1 / rho, lam / rho)
        X = ((D - E + Y1 / rho) @ A.T + (D - Y3 / rho)) @ np.linalg.inv(A @ A.T + np.eye(M.shape[1]))

        # update the dual matrices
        Y1 = Y1 + rho * (D - X @ A - E)
        Y2 = Y2 + rho * (A - B)
        Y3 = Y3 + rho * (X - D)

        # update the penalty coefficient
        rho = min(miu * rho, rho_max)

        iters = 0
        # compute the stopping criterion
        if max(np.linalg.norm(D - X @ A - E, ord='fro'), np.linalg.norm(A - B, ord='fro'), np.linalg.norm(X - D, ord='fro')) < thresh \
            or iters > 1000:
           return A, E
        else:
            iters = iters + 1
            norm_eq1 = np.linalg.norm(D - X @ A - E, ord='fro')
            norm_eq2 = np.linalg.norm(A - B, ord='fro')
            norm_eq3 = np.linalg.norm(X - D, ord='fro')
            print('norm1: ', norm_eq1)
            print('norm2: ', norm_eq2)
            print('norm3: ', norm_eq3)


if __name__ == '__main__':
    # load test picture
    img = cv2.imread('./3.jpg')
    img = img.astype(np.float)
    img /= 255.
    nrow, ncol, nchn = img.shape

    # add gaussian noise
    img += 0.0 * np.random.rand(nrow, ncol, nchn)

    # RGB channel segmentation
    img_r = img[..., 0]
    img_g = img[..., 1]
    img_b = img[..., 2]

    # compute low rank and sparse matrices
    A1, E1 = LRR_solve(img_r)
    A2, E2 = LRR_solve(img_g)
    A3, E3 = LRR_solve(img_b)

    # save matrix
    # A1 = img_r @ A1
    # A2 = img_g @ A2
    # A3 = img_b @ A3
    low_rank = np.concatenate((np.expand_dims(img_r @ A1, 2), np.expand_dims(img_g @ A2, 2), np.expand_dims(img_b @ A3, 2)), axis=2)
    sparse = np.concatenate((np.expand_dims(E1, 2), np.expand_dims(E2, 2), np.expand_dims(E3, 2)), axis=2)
    np.save('./data/lrr_miu_ascend/A3.npy', low_rank)
    np.save('./data/lrr_miu_ascend/E3.npy', sparse)

    low_rank = np.concatenate((np.expand_dims(img_r @ A1, 2), np.expand_dims(img_g @ A2, 2), np.expand_dims(img_b @ A3, 2)), axis=2)
    sparse = np.concatenate((np.expand_dims(E1, 2), np.expand_dims(E2, 2), np.expand_dims(E3, 2)), axis=2)
    cv2.imshow('low rank', low_rank)
    cv2.imshow('sparse', sparse)
    cv2.imshow('origin', img)
    # save image
    cv2.imwrite('./data/lrr_miu_ascend/lr_1e-7_1e8_1p1_1p0.png', low_rank*255)
    cv2.imwrite('./data/lrr_miu_ascend/sp_1e-7_1e8_1p1_1p0.png', sparse*255)
    cv2.waitKey(0)
