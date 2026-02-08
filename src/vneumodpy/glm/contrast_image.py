# -*- coding: utf-8 -*-
##
# Calculate GLM Contrast Image with prewhitening.
# based on K.J.Friston et al. (2000), M.W.Woolrich et al. (2001), K.J.Worsley (2001)
# returns cells of T-value matrix (Ts)
# input:
#  Cs           cells of contrast vectors (contrasts (predictor size) x 1)
#  B            predictor variables (node x predictor variables)
#  RSS          Residual Sum of Squares (node x 1)
#  X2is         Vector or single value of inv(X' * X) for contrast
#  tRs          Vector or single value of trace(R) for contrast

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from math import sqrt

def calc(Cs, B, RSS, X2is, tRs):
    # GLM contrast image
    # T = (c' * B) / sqrt(c' * X' * inv(V) * X * c * (RSS / trace(R)))
    Ts = []
    roiNum = RSS.shape[0]
    T2 = np.zeros((roiNum,1), dtype=np.float32)

    for c in Cs:
        for i in range(roiNum):
            if X2is.shape[0] == roiNum:
                X2i = X2is[i, :, :]
                d = sqrt(c.T @ X2i @ c)
            else:
                d = sqrt(c.T @ X2is @ c)

            if tRs.shape[0] == roiNum:
                se2 = sqrt(RSS[i].item() / tRs[i].item()) # to scalar
            else:
                se2 = sqrt(RSS[i].item() / tRs) # to scalar

            T2[i] = (c.T @ B[i, :].T) / (d * se2)
        Ts.append(T2.copy())
    return Ts
