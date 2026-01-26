# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import qr, inv


def linear(y, X, Q=None, R=None, perm=None, RiQ=None, dR2i=None):

    sz1, sz2 = X.shape
    if R is None and perm is None and (RiQ is None or dR2i is None):
        Q, R, perm = qr(X, mode='economic', pivoting=True)
        if R.size == 0:
            p = 0
        else:
            p = np.sum(np.abs(np.diag(R)) > max(sz1, sz2) * np.spacing(R[0, 0]))
        if p < sz2:
            R = R[:p, :p]
            Q = Q[:, :p]
            perm = perm[:p]

    b = np.zeros(sz2)  # float 64 bit
    if len(RiQ) == 0:
        b[perm] = np.linalg.solve(R, Q.T @ y)
    else:
        b[perm] = RiQ @ y
    r = y - X @ b

    return b, r

##
# preparation for linear regression (faster version).
# returns Q, R, perm, RiQ and dR2i
# input:
#  X       observations x regressors matrix

def prepare(X):
    # QR decomposition of X
    Q, R, perm = qr(X, mode='economic', pivoting=True)

    sz1, sz2 = X.shape
    if R.size == 0:
        p = 0
    else:
        p = np.sum(np.abs(np.diag(R)) > max(sz1, sz2) * np.spacing(R[0, 0]))

    if p < sz2:
        R = R[:p, :p]
        Q = Q[:, :p]
        perm = perm[:p]

    RiQ = None
    dR2i = None

    if R.shape[0] == R.shape[1]:
        RiQ = inv(R) @ Q.T

    if R.shape[0] == R.shape[1]:
        R2i = inv_qr(R.T @ R)
        dR2i = np.diag(R2i)

    return Q, R, perm, RiQ, dR2i

##
# matrix inversion based on QR decomposition (accurate version)
# X should be Square matrix

def inv_qr(X):
    sz1, sz2 = X.shape
    Q, R, perm = qr(X, mode='economic', pivoting=True)
    p = np.sum(np.abs(np.diag(R)) > max(sz1, sz2) * np.spacing(R[0, 0]))
    if p < sz2:
        R = R[:p, :p]
        Q = Q[:, :p]
        perm = perm[:p]
    Ci = np.linalg.inv(R) @ Q.T
    Xi = np.zeros((sz2, sz1), dtype=X.dtype)
    Xi[perm, :] = Ci
    return Xi
