# -*- coding: utf-8 -*-
##
# multivariate Vector Auto-Regression network class and Create mVAR network
# input:
#  x              multivariate time series matrix (node x time series)
#  ex_signal      multivariate time series matrix (exogenous input x time series) (optional)
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  lags           number of lags for autoregression (default:3)

from __future__ import print_function, division

import os
import pickle
import numpy as np
import array
import time
import h5py
import hdf5storage

from ..models.regress import prepare  # for command mode
from ..models.regress import linear   # for command mode
from ..models.mvar_init_with_cell_mth import call_executor as mvar_init_with_cell_mth

class MultivariateVARNetwork(object):
    def __init__(self):
        self.node_num = 0
        self.sig_len = 0
        self.ex_num = 0
        self.node_max = 0
        self.lags = 0
        self.bvec = []
        self.residuals = []

    def init_with_mat(self, dic):
        rvec = []
        bvec = []
        if type(dic) is np.ndarray:
            self.node_num = int(dic['nodeNum'][0, 0])
            self.sig_len = int(dic['sigLen'][0, 0])
            self.ex_num = int(dic['exNum'][0, 0])
            self.lags = int(dic['lags'][0, 0])
            self.cx_m = dic['cxM'][0, 0].flatten()
            self.cx_cov = dic['cxCov'][0, 0]
            bv = dic['bvec'][0, 0].flatten()
            rv = dic['rvec'][0, 0].flatten()
            for j in range(self.node_num):
                bvec.append(bv[j].flatten())
                rvec.append(rv[j].flatten())
        else:  # h5py
            self.node_num = int(dic['nodeNum'][0, 0])
            self.sig_len = int(dic['sigLen'][0, 0])
            self.ex_num = int(dic['exNum'][0, 0])
            self.lags = int(dic['lags'][0, 0])
            self.cx_m = dic['cxM'][::].flatten()
            self.cx_cov = dic['cxCov'][::]
            brefs = dic['bvec'][::].flatten()
            rrefs = dic['rvec'][::].flatten()
            for i in range(len(rrefs)):
                bvec.append(dic[brefs[i]][::].flatten())
                rvec.append(dic[rrefs[i]][::].flatten())
        self.node_max = self.node_num + self.ex_num
        for i in range(self.node_num):
            self.bvec.append(bvec[i])
            self.residuals.append(rvec[i])

    def init(self, x, ex_signal=[], lags=1):
        self.node_num = x.shape[0]
        self.sig_len = x.shape[1]
        self.lags = lags
        if len(ex_signal):
            self.ex_num = ex_signal.shape[0]
            x = np.concatenate([x, ex_signal], 0)
        else:
            self.ex_num = 0
        self.node_max = self.node_num + self.ex_num

        x = x.transpose()
        y = np.flipud(x)
        xti = np.zeros((self.sig_len-lags, lags*self.node_max), dtype=x.dtype)
        for p in range(lags):
            xti[:, self.node_max*p:self.node_max*(p+1)] = y[1+p:self.sig_len-lags+1+p, :]
        xti1 = np.concatenate([xti, np.ones((self.sig_len-lags, 1), dtype=x.dtype)], 1)

        _, _, perm, RiQ, dR2i = prepare(xti1)

        for i in range(self.node_num):
            yi = y[0:self.sig_len - lags, i]
            b, r = linear(yi, xti1, perm=perm, RiQ=RiQ, dR2i=dR2i)

            self.bvec.append(b)
            self.residuals.append(r)


    def init_with_cell(self, cx, cex_signal=[], lags=1, usecache=False, n_jobs=-1):
        self.node_num = cx[0].shape[0]
        self.sig_len = cx[0].shape[1]
        self.lags = lags
        if len(cex_signal):
            self.ex_num = cex_signal[0].shape[0]
        else:
            self.ex_num = 0
        self.node_max = self.node_num + self.ex_num
        dtype = cx[0].dtype

        # calculate mean and covariance of each node
        y = cx[0]
        for i in range(1, len(cx)):
            y = np.concatenate([y, cx[i]], 1)
        self.cx_m = np.mean(y, axis=1, dtype=dtype)
        self.cx_cov = np.cov(y, dtype=dtype)
        del y  # clear memory

        all_in_len = 0
        for i in range(len(cx)):
            all_in_len += cx[i].shape[1] - lags

        # this implementation is memory consumption
        print('get regression inputs')
        xts = 0
        xt = np.empty((all_in_len, self.node_num), dtype=dtype)  # smaller memory
        xti = np.empty((all_in_len, lags*self.node_max), dtype=dtype)  # smaller memory
        for i in range(len(cx)):
            x = cx[i]
            if len(cex_signal):
                x = np.concatenate([x, cex_signal[i]], 0)
            y = np.flipud(x.transpose())

            slen = y.shape[0]
            sl = slen - lags
            yt = np.empty((sl, lags*self.node_max), dtype=dtype)
            for p in range(lags):
                yt[:, self.node_max*p:self.node_max*(p+1)] = y[1+p:sl+1+p, :]

            xt[xts:xts+sl, :] = y[0:sl, 0:self.node_num]
            xti[xts:xts+sl, :] = yt[:, :]
            xts = xts + sl
            del x, y, yt
        xti1 = np.concatenate([xti, np.ones((all_in_len, 1), dtype=dtype)], 1)

        # prepare regress
        cacheName = 'results/cache-mvar-prepare-' +str(xti.shape[0])+ 'x' +str(xti.shape[1])+ '-l' +str(lags)+ '.mat'
        if usecache and os.path.isfile(cacheName):
            print('load regress.prepare cache file: ' + cacheName)
            dic = h5py.File(cacheName, 'r') # -v7.3
            perm = np.array(dic['perm']).T.astype(np.int32)
            perm = perm - 1  # matlab to python compatible
            RiQ = np.array(dic['RiQ']).T
            dR2i = np.array(dic['dR2i']).T
        else:
            print('prepare regression')
            start = time.time()
            _, _, perm, RiQ, _ = prepare(xti1)
            print('regress.prepare t=' + str(time.time() - start) + ' sec')

            '''
            path_name = 'results/cache-mvar-prepare-' + str(xti.shape[0]) + 'x' + str(xti.shape[1]) + '-l' + str(lags)
            if not os.path.isdir(path_name):
                os.makedirs(path_name, exist_ok=True)
            filename = path_name + os.sep + 'perm.dat'
            with open(filename, 'wb') as p:
                pickle.dump(array.array('i', perm.flatten()), p)
            filename = path_name + os.sep + 'RiQ.dat'
            with open(filename, 'wb') as p:
                pickle.dump(array.array('f', RiQ.flatten()), p)
            '''
            '''
            if usecache:
                # memory is too big. hdf5storage.write does not work
                matdata = {}
                perm = perm.astype(np.float32) + 1  # python to matlab compatible
                matdata['perm'] = perm
                matdata['RiQ'] = RiQ
                hdf5storage.write(matdata, filename=cacheName, matlab_compatible=True)
           '''

        # call multithread to calc regressions in each node. multiprocess does not change speed so much
        # !! caution !! this consumes huge memory.
        start = time.time()
        futures = mvar_init_with_cell_mth(self.node_num, xt, xti1, perm, RiQ, None, n_threads=n_jobs)

        self.bvec = [None] * len(futures)
        self.residuals = [None] * len(futures)
        for f in futures:
            f = f.result()  # for multithread. joblib does not return obj
            i = f[0]
            self.bvec[i] = f[1].reshape(-1, 1)
            self.residuals[i] = f[2].reshape(-1, 1)

        print('done t=' + str(time.time() - start) + ' sec')

    def load(self, path_name):
        list_file = path_name + os.sep + 'list.dat'
        with open(list_file, 'rb') as p:
            dat = pickle.load(p)
        self.node_num = dat[0]
        self.sig_len = dat[1]
        self.ex_num = dat[2]
        self.node_max = dat[3]
        self.lags = dat[4]
        resi_file = path_name + os.sep + 'residuals.dat'
        with open(resi_file, 'rb') as p:
            rvec = pickle.load(p)
            for i in range(len(rvec)):
                rvec[i] = np.array(rvec[i])
            self.residuals = rvec
        reg_file = path_name + os.sep + 'regress.dat'
        with open(reg_file, 'rb') as p:
            bvec = pickle.load(p)
            for i in range(len(bvec)):
                bvec[i] = np.array(bvec[i])
            self.bvec = bvec

    def save(self, path_name):
        if not os.path.isdir(path_name):
            os.makedirs(path_name, exist_ok=True)
        list_file = path_name + os.sep + 'list.dat'
        dat = [self.node_num, self.sig_len, self.ex_num, self.node_max, self.lags]
        with open(list_file, 'wb') as p:
            pickle.dump(dat, p)
        resi_file = path_name + os.sep + 'residuals.dat'
        with open(resi_file, 'wb') as p:
            # ndarray is not readable in MATLAB
            rvec = [None] * len(self.residuals)
            for i in range(len(self.residuals)):
                rvec[i] = array.array('f', self.residuals[i].flatten())  # changed to float32
            pickle.dump(rvec, p)
        reg_file = path_name + os.sep + 'regress.dat'
        with open(reg_file, 'wb') as p:
            # ndarray is not readable in MATLAB
            bvec = [None] * len(self.bvec)
            for i in range(len(self.bvec)):
                bvec[i] = array.array('d', self.bvec[i].flatten())
            pickle.dump(bvec, p)

    def save_mat(self, path_name, gRange=None):
        net = {}
        net['nodeNum'] = self.node_num
        net['exNum'] = self.ex_num
        net['sigLen'] = self.sig_len
        net['lags'] = self.lags
        net['cxM'] = self.cx_m
        net['cxCov'] = self.cx_cov
        rvec = [None] * self.node_num
        bvec = [None] * self.node_num
        for i in range(len(self.bvec)):
            bvec[i] = self.bvec[i].reshape(-1, 1)
            rvec[i] = self.residuals[i].reshape(-1, 1)
        net['rvec'] = rvec
        net['bvec'] = bvec

        matdata = {}
        matdata['net'] = net
        if gRange is not None:
            matdata['gRange'] = gRange
        hdf5storage.write(matdata, filename=path_name, matlab_compatible=True)
