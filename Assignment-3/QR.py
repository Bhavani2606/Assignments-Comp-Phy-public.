import numpy as np


def qr(A):#to find out the QR factorization using Gram-Scmidt orthogonalization
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def eignevlue(A, iterations = 50000):
    A_k = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        Q, R = qr(A_k)
        A_k = R @ Q
        ev = []
    for i in range (0, A_k.shape[0]):
        ev.append(A_k[i, i])
    ev = np.array(ev)
    return ev, A_k, QQ

