# -*- coding: utf-8 -*-
# get GLM HRF (hemodynamic response function) design matrix
# input:
#  onsets       cells of task set start time
#  durations    cells of task set duration
#  frames       fMRI time frames
#  TR           fMRI TR
#  res          sampling resolution of HRF
#  sp           sampling starting point (in resolution)
#  hrf          Canonical hemodynamic response function (optional)

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
import glm

def get(onsets, durations, frames, TR, res, sp, hrf=[]):

    # get Canonical hemodynamic response function
    if len(hrf) == 0:
        dt = TR / res
        t, hrf = glm.canonical_hrf.get(dt)

    tasknum = len(onsets)
    X = np.zeros((frames * res, tasknum), float)
    U = np.zeros((frames * res, tasknum), float)
    for k in range(tasknum):
        onset = onsets[k]
        duration = durations[k]
        for i in range(len(onset)):
            t1 = int(np.ceil(onset[i] / TR * res))
            t2 = int(np.ceil(t1 + duration[i] / TR * res))
            U[t1-1:t2,k] = 1
        # get design matrix
        C = np.convolve(U[:,k], hrf)
        X[:,k] = C[0:frames * res]

    # final sampling
    X = X[np.arange((sp-1),X.shape[0],res),:]
    U = U[np.arange((sp-1),U.shape[0],res),:]
    return X, U
