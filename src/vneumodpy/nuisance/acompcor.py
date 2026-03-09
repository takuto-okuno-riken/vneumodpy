# get nuisance time-series of aCompCor (white matter & csf ROI) component
# time-series based on Y.Behzadi et al. (2007).
# returns csf <compNum> components, wm <compNum> components.
# input:
#  V            4D fMRI time-series
#  csfV         mask for CSF
#  wmV          mask for White matter
#  Sd           detrend signal (time-series x node) before PCA (optional)
#  maskTh       mask threshold (percentile) for ROI of aCompCor (default:99)
#  compNum      component number of aCompCor (default:6)

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..models.regress import linear
from ..models.regress import prepare

def get(V, csfV, wmV, Sd=None, maskTh=99, compNum=6):
    # get ROI index for aCompCor
    csfIdx = np.where(csfV.flatten() >= np.percentile(csfV, maskTh))[0]  # extract top 1% voxels
    wmIdx = np.where(wmV.flatten() >= np.percentile(wmV, maskTh))[0]    # extract top 1% voxels

    # detrend
    V = V.reshape(-1, V.shape[3])
    M = np.nanmean(V, axis=1, keepdims=True)
    V = V - M
    csfX = np.squeeze(V[csfIdx, :]).astype(np.float32)  # hmm...
    wmX = np.squeeze(V[wmIdx, :]).astype(np.float32)   # hmm...

    if Sd is not None:
        _, _, perm, RiQ, dR2i = prepare(Sd)
        # 1st step OLS regression
        for i in range(len(csfIdx)):
            _, csfX[i, :] = linear(csfX[i, :].T, Sd, perm=perm, RiQ=RiQ, dR2i=dR2i)
        for i in range(len(wmIdx)):
            _, wmX[i, :] = linear(wmX[i, :].T, Sd, perm=perm, RiQ=RiQ, dR2i=dR2i)

    # get PCA time-series
    Xn = None
    st = 0
    ed = st + compNum
    if csfX.size > 0:
        scaler = StandardScaler()
        pca_csf = PCA(n_components=compNum)
#        X_scaled = scaler.fit_transform(csfX.T)  # not compatible with MATLAB
        Xn = pca_csf.fit_transform(csfX.T)
        st = compNum
        ed = st + compNum
    if wmX.size > 0:
        scaler = StandardScaler()
        pca_wm = PCA(n_components=compNum)
#        X_scaled = scaler.fit_transform(wmX.T)  # not compatible with MATLAB
        score_wm = pca_wm.fit_transform(wmX.T)
        if Xn is None:
            Xn = score_wm
        else:
            Xn = np.hstack((Xn, score_wm))

    return Xn
