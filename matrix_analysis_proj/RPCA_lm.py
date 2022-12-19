import cv2
import numpy as np


def initialization(D):
    """
    Auto decide the hyperparameters
    :param D: The given data matrix.
    :return rho, rho_max, miu, lam, thresh: The hyper-parameters.
    """
    m, n = D.shape
    lam = 1.0 / np.sqrt(max(m, n))  # the larger, low rank the clearer

    norm_1 = np.linalg.norm(D, ord=1)
    norm_fro = np.linalg.norm(D, ord='fro')

    # rho = 1.25 / norm_2

    # param1
    # rho = (m * n) / (4 * norm_1)
    # rho_max = 1e5
    # miu = 1.5

    # param2 
    lam = lam * 1.0
    rho = 1e-7
    rho_max = 1e8
    miu = 1.1  # the bigger, converge the faster, low rank the clearer

    thresh = 1e-7 * norm_fro

    return rho, rho_max, miu, lam, thresh

def shrinkage(X, thresh):
    """
    The soft threshold function.
    Solution to proximal minimization with 1-norm.
    """
    return np.sign(X) * np.maximum(np.abs(X) - thresh, 0)

def RPCA_solve(D):
    """
    Solve the RPCA problem.
    :param M: The given data matrix.
    :return A: Low rank matrix.
    :return E: Sparse matrix.
    :return Y: Dual matrix.
    """
    # decide the hyper-parameters
    rho, rho_max, miu, lam, thresh = initialization(D)
    print('rho:{0}, rho_m:{1}, miu:{2}, lam:{3}, thresh:{4}'.format(rho, rho_max, miu, lam, thresh))
    # initialize the primal and dual matrices
    Y = np.zeros_like(D)
    A, E = np.zeros_like(D), np.zeros_like(D)
    norm = 0
    iters = 0

    while True:
        # alternative direction descend
        U, S_, VT = np.linalg.svd(D - E + Y / rho)
        S = np.zeros_like(D - E + Y / rho)
        m, n = S.shape
        S[:min(m, n), :min(m, n)] = np.diag(S_)
        A = U @ shrinkage(S, 1 / rho) @ VT
        E = shrinkage(D - A + Y / rho, lam / rho)

        # update the dual matrix
        Y = Y + rho * (D - A - E)

        # rho_ascend (can promote convergence speed)
        rho = min(rho_max, miu * rho)

        # compute the stopping criterion
        if np.linalg.norm(D - A - E, ord='fro') < thresh or iters > 10000:
           return A, E, Y
        else:
            iters = iters + 1
            norm = np.linalg.norm(D - A - E, ord='fro')
            print('error: ', norm)


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
    A1, E1, Y1 = RPCA_solve(img_r)
    A2, E2, Y2 = RPCA_solve(img_g)
    A3, E3, Y3 = RPCA_solve(img_b)

    low_rank = np.concatenate((np.expand_dims(A1, 2), np.expand_dims(A2, 2), np.expand_dims(A3, 2)), axis=2)
    sparse = np.concatenate((np.expand_dims(E1, 2), np.expand_dims(E2, 2), np.expand_dims(E3, 2)), axis=2)
    
    # save matrix
    # np.save('./data/lm_miu_const/A3.npy', low_rank)
    # np.save('./data/lm_miu_const/E3.npy', sparse)

    cv2.imshow('low rank', low_rank)
    cv2.imshow('sparse', sparse)
    cv2.imshow('origin', img)
    # save image
    cv2.imwrite('./data/lm_miu_ascend/lr_1e-7_1e8_1p1_1p0.png', low_rank*255)
    cv2.imwrite('./data/lm_miu_ascend/sp_1e-7_1e8_1p1_1p0.png', sparse*255)
    cv2.waitKey(0)
