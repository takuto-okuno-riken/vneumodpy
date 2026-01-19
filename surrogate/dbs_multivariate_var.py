# -*- coding: utf-8 -*-
# Simulated DBS surrogate multivariate signal generation by multivariate VAR
# based on autoregressive (AR) surrogates (R. Liegeois et al., 2017)
# returns surrogated signals (Y)
# input:
#  X            multivariate time series matrix (node x time series)
#  exSignal     multivariate time series matrix (exogenous input x time series) (optional)
#  net          mVAR network
#  A            DBS stimulus multivariate time series matrix (add) (node x time series)
#  M            DBS stimulus multivariate time series matrix (multi) (node x time series)
#  dist         distribution of noise to yield surrogates ('gaussian'(default), 'residuals')
#  surrNum      output number of surrogate samples (default:1)
#  yRange       range of Y value (default:[Xmin-Xrange/5, Xmax+Xrange/5])
#  nBaset       noise base time (default:empty (randperm))
#  Cin          coefficient matrix of VAR (default:[])
#  Errin        Err matrix of VAR (default:[])

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from numpy.random import default_rng
from scipy.stats import multivariate_normal

def calc(X, exSignal, net, A, M, dist='gaussian', surrNum=1, yRange=None, nBaset=None, Cin=None, Errin=None):
    nodeNum = X.shape[0]
    sigLen = X.shape[1]
    if len(exSignal) > 0:
        exNum = exSignal.shape[0]
        Xorg = np.vstack((X, exSignal))
    else:
        exNum = 0
        Xorg = X
    inputNum = nodeNum + exNum
    lags = net.lags

    # calc Y range
    if yRange is None:
        t = np.max(X)
        d = np.min(X)
        r = t - d
        yRange = np.array([d - r / 5, t + r / 5])

    if Errin is None:
        # set residual matrix
        rvlen = len(net.residuals[0])
        for i in range(1, nodeNum):
            if rvlen > len(net.residuals[i]):
                rvlen = len(net.residuals[i])
        Err = np.zeros((nodeNum, rvlen), dtype=np.float32)
        for i in range(nodeNum):
            Err[i, :] = net.residuals[i][:rvlen]
    else:
        Err = Errin

    # get coefficient matrix
    if Cin is None:
        C = np.zeros((nodeNum, inputNum * lags + 1), dtype=np.float32)
        for i in range(nodeNum):
            C[i, 0:C.shape[1]-1] = net.lr_objs[i].coef_
            C[i, C.shape[1] - 1] = net.lr_objs[i].intercept_
    else:
        C = Cin

    if dist == 'gaussian':
        P = np.mean(Err, axis=1)
        EC = np.cov(Err, bias=True)
        rng = default_rng()
        Err = rng.multivariate_normal(P, EC, size=Err.shape[1]).T.astype(np.float32)
    noise = Err

    S2 = np.ones((inputNum * lags + 1,), dtype=np.float32)

    # set noise permutation
    if nBaset is not None:
        if isinstance(nBaset, list) or isinstance(nBaset, tuple):
            perm = nBaset[0]
            perm = perm + nBaset[1]
        elif np.isscalar(nBaset):
            perm = np.arange(nBaset, noise.shape[1] - lags)
        else:
            perm = nBaset
    else:
        rng = default_rng()
        perm = rng.permutation(noise.shape[1] - lags)
    perm = perm.astype(np.int32)

    Y = np.full((nodeNum, sigLen, surrNum), np.nan, dtype=np.float32)
    for k in range(surrNum):
        print(f'surrogate sample : {k+1}')
        S = Xorg.astype(np.float32)
        # random is not so robust. use fixed end frame signals
        S[:nodeNum, :lags] = S[:nodeNum, sigLen - lags:sigLen]
        perm2 = np.roll(perm, -sigLen * k)

        for t in range(lags, sigLen):
            T = S[:, t].copy()  # next output

            for p in range(1, lags + 1):
                S2[(inputNum * (p - 1)):(inputNum * p)] = S[:, t - p]
            T[:nodeNum] = C @ S2 + noise[:, perm2[t - lags]-1]   # hmm...this one takes time

            # DBS stimulation
            if M is not None:
                Mt = M[:, t]
                idx = ~np.isnan(Mt)
                if np.any(idx):
                    T[idx] = T[idx] * Mt[idx]

            if A is not None:
                At = A[:, t]
                idx = ~np.isnan(At)
                if np.any(idx):
                    T[idx] = T[idx] + At[idx]

            # fixed over shoot values
            if yRange is not None:
                T[T < yRange[0]] = yRange[0]
                T[T > yRange[1]] = yRange[1]

            S[:, t] = T

        Y[:, :, k] = S[:nodeNum, :]

    return Y, C, Err, perm