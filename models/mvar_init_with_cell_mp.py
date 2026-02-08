# -*- coding: utf-8 -*-

import os
import numpy as np
import joblib
from concurrent.futures import ProcessPoolExecutor

try:
    from .. import models
except ImportError:
    from models.regress import linear  # for command mode

# global values
g_xti1 = None
g_perm = None
g_RiQ = None
g_dR2i = None

# for multiprocessing function
def init_with_cell_loop_fn(i, yi):
    global g_xti1
    global g_perm
    global g_RiQ
    global g_dR2i
    print('calc node ' + str(i))
    b, r = linear(yi, g_xti1, perm=g_perm, RiQ=g_RiQ, dR2i=g_dR2i)  # around 2 sec
    return i, b, r

#    lr = LinearRegression(fit_intercept=True)
#    start = time.time()
#    lr.fit(xti, yi)  # around 1000 sec (a bit faster)
#    pred = lr.predict(xti)
#    r = (yi - pred)
#    print('lr.fit t=' + str(time.time() - start) + ' sec')

#    start = time.time()
#    b0, _, _, _ = np.linalg.lstsq(xti2, yi, rcond=None) # around 1779 sec (slow)
#    print('np.linalg.lstsq t=' + str(time.time() - start) + ' sec')

#    start = time.time()
#    b1 = np.linalg.solve(xti2, yi) # not work
#    print('np.linalg.solve t=' + str(time.time() - start) + ' sec')


def call_executor(node_num, xt, xti1, perm, RiQ, dR2i, n_jobs=-1):
    # use global values and sharedmem (does not work well. still slow.)
    global g_xti1
    global g_perm
    global g_RiQ
    global g_dR2i

    g_xti1 = xti1
    g_perm = perm
    g_RiQ = RiQ
    g_dR2i = dR2i
    futures = joblib.Parallel(n_jobs=n_jobs, require='sharedmem', verbose=1)(
        joblib.delayed(init_with_cell_loop_fn)(i, xt[:, i]) for i in range(node_num))

    # clear memory
    g_xti1 = None
    g_perm = None
    g_RiQ = None
    g_dR2i = None

    # does not work well
#    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
#        futures = set()
#        for i in range(node_num):
#            v = [i, xt[:, i], xti1, perm, RiQ, dR2i]
#            future = executor.submit(init_with_cell_loop_fn, v)
#            futures.add(future)
    return futures
