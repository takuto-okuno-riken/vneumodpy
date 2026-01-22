# -*- coding: utf-8 -*-
##
# get NIfTI 4D volume from ROI time-series (matrix).
# returns nifti 4D volume (V)(X x Y x Z x frames)
# input:
#  X            ROI time-series (ROIs x frames)
#  atlasV       nifti 3D atlas (X x Y x Z)

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np

def get(X, atlasV):
    roinum = X.shape[0]

    xs, ys, zs = atlasV.shape
    V = np.full((xs, ys, zs, X.shape[1]), np.nan, dtype=np.float32)

    idxs = [None] * roinum
    aVf = atlasV.flatten()
    aVfidx = np.array(range(aVf.shape[0]))
    for i in range(roinum):
        idxs[i] = aVfidx[aVf == (i+1)]

    for t in range(X.shape[1]):
        A = np.full((xs, ys, zs), np.nan, dtype=np.float32)
        for i in range(roinum):
            A.flat[idxs[i]] = X[i, t]
        V[:, :, :, t] = A

    return V