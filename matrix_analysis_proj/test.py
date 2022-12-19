from copy import deepcopy
import numpy as np


# A = np.diag([1, 2, 3, 0]).astype(np.double)
A = np.arange(16).reshape(4, 4).astype(np.double)

U, S_, VT = np.linalg.svd(A, full_matrices=False)

import pdb; pdb.set_trace()

S = np.diag(S_)

print(U @ VT)

df = np.zeros_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        B = deepcopy(A)
        B[i, j] += 1e-9
        df[i, j] = (np.sum(np.linalg.svd(B)[1]) - np.sum(np.linalg.svd(A)[1])) / 1e-9

print(df)

import pdb; pdb.set_trace()
