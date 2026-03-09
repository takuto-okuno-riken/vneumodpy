# -*- coding: utf-8 -*-
##
# get ROI time-series (matrix) from NIfTI 4D volume.
# returns ROI time-series (X)
# input:
#  V            nifti 4D volume (X x Y x Z x frames)
#  atlasV       nifti 3D atlas (X x Y x Z)
#  operation    calc operation for each plane ('mode'(default),'max','min','mean','median','sum')

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from scipy.stats import mode

def get(V, atlasV, operation='mode'):
    roiIdx = np.unique(atlasV)
    roiIdx = roiIdx[roiIdx != 0]  # remove 0

    X = np.zeros((len(roiIdx), V.shape[3]), dtype=np.float32)
    A = V.reshape(-1, V.shape[3])
    for i in range(len(roiIdx)):
        j = roiIdx[i]
        B = A[atlasV.flatten() == j, :]
        if operation == 'mode':
            m = mode(B, axis=0, nan_policy='omit').mode[0]
        elif operation == 'min':
            m = np.nanmin(B, axis=0)
        elif operation == 'max':
            m = np.nanmax(B, axis=0)
        elif operation == 'mean':
            m = np.nanmean(B, axis=0)
        elif operation == 'median':
            m = np.nanmedian(B, axis=0)
        elif operation == 'sum':
            m = np.nansum(B, axis=0)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        X[i, :] = m
    return X
