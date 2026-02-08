# -*- coding: utf-8 -*-
##
# Get Addition and Multiplication time-series for virtual neuromodulation
# returns cells of Addition time-series (CA), cells of HRF time-series (Chrf), and cells of Multiplication time-series (CM).
# input:
#  CX              cells of multivariate time series matrix {node x time series}
#  dbsidx          target modulation ROI index (for CX)
#  surrnum         output number of surrogate samples
#  srframes        frame length of surrogate time-series
#  dbsoffsec       neuromodulation off duration (seconds)
#  dbsonsec        neuromodulation on duration (seconds)
#  dbspw           neuromodulation power
#  TR              TR of fMRI data (CX)
#  res             HRF sampling resolution (optional)
#  sp              HRF sampling starting point (optional)

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np

from ..glm.canonical_hrf import get as canonical_hrf  # for package
from ..glm.hrf_design_matrix import get as hrf_design_matrix

def get(CX, dbsidx, surrnum=40, srframes=160, dbsoffsec=28, dbsonsec=22, dbspw=0.15, TR=1.0, res=16, sp=8):

    # get HRF
    dt = TR / res
    [t, hrf] = canonical_hrf(dt) # human's HRF;

    # find time series range
    cxlen = len(CX)
    shape = (CX[0].shape[0], CX[0].shape[1], cxlen) # CX[0] should have biggest matrix size
    X3 = np.full(shape, np.nan, dtype=np.float32)
    for k in range(cxlen):
        X = CX[k]
        X3[0:X.shape[0], 0:X.shape[1], k] = X
    m = np.nanmean(X3.flatten())
    s = np.nanstd(X3.flatten(), ddof=0)
    del X3 # clear

    # DBS block design (off, on, off, on, ...)
    dbsidx = dbsidx - 1  # for matlab compatibility
    CA = [None]*surrnum
    Chrf = [None]*surrnum
    CM = [None]*surrnum
    for i in range(surrnum):
        n = CX[i].shape[0]
        bmax = int(np.floor((srframes * TR) / (dbsoffsec+dbsonsec)))
        ons = np.zeros(bmax)
        dur = np.zeros(bmax)
        for j in range(bmax):
            ons[j] = dbsoffsec + j*(dbsoffsec+dbsonsec)
            dur[j] = dbsonsec
        onsets = [ons]
        durations = [dur]
        ch, U = hrf_design_matrix(onsets, durations, srframes, TR, res, sp, hrf)
        Chrf[i] = ch.astype(np.float32)
#        sio.savemat('temp_chrf.mat',{'chrf':Chrf[i],'U':U}) # for debug
        block = np.full((n,srframes), np.nan, dtype=np.float32) # need to allocate new memory
        block[dbsidx,:] = np.tile(Chrf[i].T * dbspw * s, (len(dbsidx),1))
        CA[i] = block.copy()
        block[dbsidx,:] = np.tile(1 - 0.5 * Chrf[i].T, (len(dbsidx),1))
        CM[i] = block.copy()

#    sio.savemat('temp_cacm.mat',{'CA0':CA,'Chrf0':Chrf,'CM0':CM}) # for debug
    return CA, Chrf, CM
