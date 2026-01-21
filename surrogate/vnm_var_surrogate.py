# -*- coding: utf-8 -*-
# Calculate virtual neuromodulation surrogate by Vector Auto-Regression
# returns modulated surrogate time-series (S).
# input:
#  net             mVAR network (struct)
#  CX              cells of multivariate time series matrix {node x time series}
#  CA              cells of Addition time-series {node x time series}
#  CM              cells of Multiplication time-series {node x time series}
#  perm            subject permutation for ordered residual
#  surrNum         output number of surrogate samples
#  srframes        frame length of surrogate time-series

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np
import surrogate
import time
#import scipy.io as sio
#from line_profiler import LineProfiler

def calc(net, CX, CA, CM, perm, surrnum, srframes):
    dist = 'residuals'
    cxlen = len(CX)
    frames = CX[0].shape[1]

    # Virtual Neuromodulation VAR surrogate
    S = [None] * surrnum
    C = None
    Err = None
    for i in range(surrnum):
        X = CX[np.mod(i*2,cxlen)]
        for k in range(1,cxlen):
            if X.shape[1] >= srframes:
                break
            X = np.concatenate([X, CX[np.mod(i * 2 + k, cxlen)]], 1)
        if i > len(S) or S[i] is None:
            start = time.time()
            # ordered residual with subject permutation
            nBaset = [perm[frames*i:frames*(i+1)], 0]

#            lp = LineProfiler() # check profile
#            lp_wrapper = lp(surrogate.dbs_multivariate_var)
#            lp_wrapper(X[:,0:srframes], [], net, CA[i], CM[i], dist, 1, None, nBaset, C, Err)
#            lp.print_stats()

            S[i], C, Err, _ = surrogate.dbs_multivariate_var(X[:,0:srframes], [], net, CA[i], CM[i], dist, 1, None, nBaset, C, Err)
            print('done t=' +str(time.time() - start)+ ' sec')
#        sio.savemat('tempS.mat',{'S0':S[i]}) # for debug
    return S
