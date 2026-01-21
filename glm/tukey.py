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

import numpy as np
import time
from scipy.linalg import toeplitz, cholesky, inv
from scipy.stats import linregress

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
    CR = [None] * roiNum
    df = X.shape[0] - X.shape[1]

    # make Tukey window
    tuWin = np.zeros(tuM)
    for k in range(tuM):
        tuWin[k] = 0.5 * (1 + np.cos(np.pi * (k+1) / tuM))

    def regress(y, X):
        # Equivalent to MATLAB regress: returns b, residuals
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return b, r

    start = time.time()
    for i in range(roiNum):
        Yi = Y[:, i]
        # 1st step OLS regression
        _, r = regress(Yi, X)

        # if residuals are all zero, probably original signal is just zero. ignore this voxel
        if np.sum(r == 0) == r.shape[0]:
            B[i, :] = 0
            RSS[i] = 0
            CR[i] = r
            continue

        # calc AR coefficients by Tukey-Taper of frequency domain
        C = np.correlate(r, r, mode='full') / np.dot(r, r)
        mid = len(C) // 2
        Rxx = C[mid:mid + tuM]
        Pxx = np.zeros(r.shape[0])
        Pxx[:tuM] = Rxx[:tuM] * tuWin[:tuM]
        V1 = toeplitz(Pxx)
        K1 = np.linalg.cholesky(V1, upper=False)

        # second time regression
#        Ki1 = np.linalg.inv(K1)
#        Ya = Ki1 @ Yi
#        Xt = Ki1 @ X
        Ya = np.linalg.solve(K1, Yi)
        Xt = np.linalg.solve(K1, X)
        b, r = regress(Ya, Xt)

        B[i, :] = b
        RSS[i] = r.T @ r
        if not isOutX2is:
            continue

        # used for contrast
        C = np.correlate(r, r, mode='full') / np.dot(r, r)
        Rxx = C[mid:mid + tuM]
        Pxx = np.zeros(r.shape[0])
        Pxx[:tuM] = Rxx[:tuM] * tuWin[:tuM]

        V2 = toeplitz(Pxx)
        X2is[i, :, :] = inv(X.T @ (inv(V2) @ X))
        IR = np.eye(Xt.shape[0]) - Xt @ inv(Xt.T @ Xt) @ Xt.T
        tRs[i] = np.trace(IR)

    print('done t=' + str(time.time() - start) + ' sec')
    return B, RSS, df, X2is, tRs
