# -*- coding: utf-8 -*-

import numpy as np

def get(cx):
    a = cx[0]
    for i in range(1, len(cx)):
        a = np.concatenate([a, cx[i]], 1)
    r = {
        'min': np.min(a),
        'max': np.max(a),
        'm': np.nanmean(a),
        's': np.nanstd(a)}
    return r


def get_dic(dic):
    if type(dic) is np.ndarray:
        r = {
            'min': dic['min'][0, 0][0, 0],
            'max': dic['max'][0, 0][0, 0],
            'm': dic['m'][0, 0][0, 0],
            's': dic['s'][0, 0][0, 0]}
    else:  # h5py
        r = {
            'min': dic['min'][0, 0],
            'max': dic['max'][0, 0],
            'm': dic['m'][0, 0],
            's': dic['s'][0, 0]}
    return r
