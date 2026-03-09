# get nuisance time-series of Global mean, Global signal, mean CSF, mean WM
# returns global mean, global signal, mean csf, mean white matter.
# input:
#  V            4D fMRI time-series
#  csfV         mask for CSF (optional)
#  wmV          mask for White matter (optional)
#  gsV          mask for Global signal (whole brain mask)(optional)

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np

def get(V, csfV=None, wmV=None, gsV=None):
    Xn = np.full((V.shape[3], 4), np.nan)
    for i in range(V.shape[3]):
        Vi = V[:, :, :, i]
        Xn[i, 0] = np.nanmean(Vi)  # global mean
        if gsV is not None:
            V1 = Vi * gsV.astype(float)
            V1[V1 <= 0] = np.nan
            Xn[i, 1] = np.nanmean(V1)  # global signal
        if csfV is not None:
            V1 = Vi * csfV.astype(float)
            V1[V1 <= 0] = np.nan
            Xn[i, 2] = np.nanmean(V1)  # csf mean
        if wmV is not None:
            V1 = Vi * wmV.astype(float)
            V1[V1 <= 0] = np.nan
            Xn[i, 3] = np.nanmean(V1)  # wm mean

    Xn = Xn - np.tile(np.nanmean(Xn, axis=0),(Xn.shape[0],1))
    if gsV is not None and gsV.size > 0:
        Xn = Xn / (np.nanstd(Xn[:, 1], ddof=0) * 4)  # normalize around [-1 1] range

    if wmV is None or wmV.size == 0:
        Xn = np.delete(Xn, 3, axis=1)
    if csfV is None or csfV.size == 0:
        Xn = np.delete(Xn, 2, axis=1)
    if gsV is None or gsV.size == 0:
        Xn = np.delete(Xn, 1, axis=1)

    return Xn