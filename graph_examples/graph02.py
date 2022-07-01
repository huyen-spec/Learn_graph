import numpy as np

A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)


X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float)

print(X)


D = np.array(np.sum(A, axis=0))[0]
D = np.matrix(np.diag(D))
print(D)


# D = np.array(np.sum(A, axis=1))
# D = np.matrix(np.diag(D))
# print(D)


print("D**-1 * A", D**-1 * A)


# In [6]: A * X
# Out[6]: matrix([
#             [ 1., -1.],
#             [ 5., -5.],
#             [ 1., -1.],
#             [ 2., -2.]]

print('D**-1 * A * X\n', D**-1 * A * X)

W = np.matrix([
             [1, -1],
             [-1, 1]
         ])

print('D_hat**-1 * A_hat * X * W\n', D**-1 * A * X * W)