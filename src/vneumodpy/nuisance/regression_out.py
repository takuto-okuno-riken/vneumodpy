from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from ..models.regress import linear
from ..models.regress import prepare

def get(V, Xn, maskV):
    sz = V.shape[0] * V.shape[1] * V.shape[2]
    V = V.reshape(sz, V.shape[3])
    M = np.nanmean(V, axis=1)
    V = V - M[:, None]
    _, _, perm, RiQ, dR2i = prepare(Xn)
    CR = [None] * sz
    idxs = np.where(maskV.flatten() > 0)[0]
    for i in range(len(idxs)):
        idx = idxs[i]
        # 1st step OLS regression
        _, V[idx, :] = linear(V[idx, :].T, Xn, perm=perm, RiQ=RiQ, dR2i=dR2i)
    V = V + M[:, None]
    Vout = V.reshape(maskV.shape[0], maskV.shape[1], maskV.shape[2], V.shape[1])
    return Vout