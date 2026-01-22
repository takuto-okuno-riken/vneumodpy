# -*- coding: utf-8 -*-
##
# Calculate General Linear Model with prewhitening (Tukey-Taper, Auto-correlation estimation Version)
# based on M.W.Woolrich et al. (2001)
#   Y = X * B + e
#   e ~ N(0, s^2 * V)
#   S * Y = S * X * B + r
#   S = inv(K)
#   V = K * K'
#   V is auto-corerlation matrix estimated by Tukey-Taper
#   r ~ N(0, s^2 * SVS') therefore N(0, s^2 * I)
# This function is used for single session.
# returns predictor variables (B), Residual Sum of Squares (RSS), degree of freedom (df),
#   inv(X' * X) for contrast (X2is), trace(R) for contrast (tRs), full of Residuals (R)
# input:
#  Y         ROI or voxel time series (time series x node)
#  X         design matrix (time series x predictor variables)
#  tuM       Tukey-Taper window size (default: sqrt(time series length))

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
import time
import os
from scipy.linalg import toeplitz, cholesky, inv
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor


def regress(y, X):
    # Equivalent to MATLAB regress: returns b, residuals
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, r

def loop_fn(i, Yi, X, tuWin, tuM, isOutX2is):
    # 1st step OLS regression
    _, r = regress(Yi, X)

    # if residuals are all zero, probably original signal is just zero. ignore this voxel
    if np.sum(r == 0) == r.shape[0]:
        return i, 0, 0

    # calc AR coefficients by Tukey-Taper of frequency domain
    C = np.correlate(r, r, mode='full') / np.dot(r, r)
    mid = len(C) // 2
    Rxx = C[mid:mid + tuM]
    Pxx = np.zeros(r.shape[0], dtype=np.float32)
    Pxx[:tuM] = Rxx[:tuM] * tuWin[:tuM]
    V1 = toeplitz(Pxx)
    K1 = np.linalg.cholesky(V1, upper=False)

    # second time regression
#    Ki1 = np.linalg.inv(K1)
#    Ya = Ki1 @ Yi
#    Xt = Ki1 @ X
    Ya = np.linalg.solve(K1, Yi)
    Xt = np.linalg.solve(K1, X)
    b, r = regress(Ya, Xt)

    rss = r.T @ r
    if not isOutX2is:
        return i, b, rss

    # used for contrast
    C = np.correlate(r, r, mode='full') / np.dot(r, r)
    Rxx = C[mid:mid + tuM]
    Pxx = np.zeros(r.shape[0], dtype=np.float32)
    Pxx[:tuM] = Rxx[:tuM] * tuWin[:tuM]

    V2 = toeplitz(Pxx)
    x2is = inv(X.T @ (inv(V2) @ X))
    IR = np.eye(Xt.shape[0]) - Xt @ inv(Xt.T @ Xt) @ Xt.T
    trs = np.trace(IR)
    return i, b, rss, x2is, trs

def calc(Y, X, tuM=None, isOutX2is=False):
    if tuM is None:
        tuM = int(np.floor(np.sqrt(X.shape[0])))

    print('process GLM with Tukey-Taper(' +str(tuM)+ ') estimation ...')
    roiNum = Y.shape[1]
    xsz = X.shape[1]
    X2is = np.full((roiNum, xsz, xsz), np.nan, dtype=np.float32)
    tRs = np.full((roiNum,1), np.nan, dtype=np.float32)
    B = np.full((roiNum, xsz), np.nan, dtype=np.float32)
    RSS = np.full((roiNum,1), np.nan, dtype=np.float32)
    df = X.shape[0] - X.shape[1]

    # make Tukey window
    tuWin = np.zeros(tuM, dtype=np.float32)
    for k in range(tuM):
        tuWin[k] = 0.5 * (1 + np.cos(np.pi * (k+1) / tuM))

    start = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # -----(2)
        futures = set()
        for i in range(roiNum):
            future = executor.submit(loop_fn, i, Y[:, i], X, tuWin, tuM, isOutX2is)
            futures.add(future)

    for f in futures:
        r = f.result()
        i = r[0]
        B[i, :] = r[1]
        RSS[i] = r[2]
        if isOutX2is:
            X2is[i, :, :] = r[3]
            tRs[i] = r[4]

    print('done t=' + str(time.time() - start) + ' sec')
    return B, RSS, df, X2is, tRs
