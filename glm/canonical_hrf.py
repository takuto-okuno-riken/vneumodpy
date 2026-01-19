# -*- coding: utf-8 -*-
##
# get GLM Canonical hemodynamic response function
# returns time range (t), HRF time-series (hrf)
# input:
#  dt                 time resolution (sec) (default:0.045)
#  responseDelay      delay of response (gamma a)(sec) (default:6)
#  underShootDelay    delay of undershoot (gamma a)(sec) (default:16)
#  kernelSec          kernel time length (sec) (default: 32)
#  underShootRatio    ratio of underShoot (default: 0.167)
#  hrfScale           HRF scale (gamma b) (default: 0.9)

import numpy as np
from scipy.stats import gamma

def get(dt=0.045, response_delay=6, under_shoot_delay=16, kernel_sec=32, under_shoot_ratio=0.167, hrf_scale=0.9):

    t = np.arange(0, kernel_sec+dt, dt)
    hrf = gamma.pdf(t,response_delay,scale=hrf_scale) - gamma.pdf(t,under_shoot_delay,scale=hrf_scale) * under_shoot_ratio
    hrf = hrf.transpose() / np.sum(hrf)
    return t, hrf
