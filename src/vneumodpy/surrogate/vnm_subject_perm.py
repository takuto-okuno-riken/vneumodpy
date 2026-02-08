# -*- coding: utf-8 -*-
##
# Generate subject permutation time-series for virtual neuromodulation surrogate
# returns permutated time-series (perm).
# input:
#  CX              cells of multivariate time series matrix {node x time series}

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
from datetime import datetime
import time

def get(CX):
    perm = np.empty(0)
    cxlen = len(CX)
    frames = CX[0].shape[1]

    # ordered residual with subject permutation
    uxtime = np.uint32(int(time.mktime(datetime.now().timetuple())))
    np.random.seed(uxtime)
    rp = np.random.permutation(cxlen)
    for i in range(cxlen):
        perm = np.concatenate([perm, np.arange(frames) + 1 + rp[i]*frames])  # matlab compatible
    return perm, uxtime
