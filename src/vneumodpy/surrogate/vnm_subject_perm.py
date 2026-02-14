# -*- coding: utf-8 -*-
##
# Generate subject permutation time-series for virtual neuromodulation surrogate
# returns permutated time-series (perm), uxtime and residual length (reslen).
# input:
#  CX              cells of multivariate time series matrix {node x time series}
#  lags            number of lags for autoregression

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from datetime import datetime
import time

def get(CX, lags):
    perm = np.empty(0)
    cxlen = len(CX)
    reslen = CX[0].shape[1] - lags  # residual length

    # ordered residual with subject permutation
    uxtime = np.uint32(int(time.mktime(datetime.now().timetuple())))
    np.random.seed(uxtime)
    rp = np.random.permutation(cxlen)
    for i in range(cxlen):
        perm = np.concatenate([perm, np.arange(reslen) + 1 + rp[i]*reslen])  # matlab compatible
    return perm, uxtime, reslen
