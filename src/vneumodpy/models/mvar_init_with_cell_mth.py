# -*- coding: utf-8 -*-

from ..models.regress import linear
from concurrent.futures import ThreadPoolExecutor


# for multithreading function
def init_with_cell_loop_fn(i, yi, xti1, perm, RiQ, dR2i):
    print('calc node ' + str(i))
    b, r = linear(yi, xti1, perm=perm, RiQ=RiQ, dR2i=dR2i)  # around 2 sec
    return i, b, r


def call_executor(node_num, xt, xti1, perm, RiQ, dR2i, n_threads=-1):
    # this may not work well with CPython, because of GIL
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = set()
        for i in range(node_num):
            future = executor.submit(init_with_cell_loop_fn, i, xt[:, i], xti1, perm, RiQ, dR2i)
            futures.add(future)

    return futures
