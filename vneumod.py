# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import hdf5storage
import nibabel as nib

from src import vneumodpy as vnm
from utils.parse_vneumod_options import ParseOptions
#from line_profiler import LineProfiler

# -------------------------------------------------------------------------
# util functions

# -------------------------------------------------------------------------
# main
if __name__ == '__main__':
    options = ParseOptions()
    opt = options.parse()

    if type(opt.outpath) is list:
        opt.outpath = opt.outpath[0]  # replaced by string

    # extract rois
    if len(opt.roi) == 0:
        print('no modulation target ROI number. please specify ROI.')
        exit(-1)
    elif ':' in opt.roi[0]:
        t = opt.roi[0].split(':')
        rois = list(range(int(t[0]), int(t[1])+1))
    elif ',' in opt.roi[0]:
        t = opt.roi[0].split(',')
        rois = [int(item) for item in t]
    else:
        rois = [int(opt.roi[0])]

    # load cube atlas nifti file
    if len(opt.targatl) == 0:
        print('no modulation target atlas file. please specify nifti file.')
        exit(-1)
    if len(opt.atlas) == 0:
        print('when target atlas is specified, an atlas file must be specified.')
        exit(-1)
    targetDat = nib.load(opt.targatl[0])
    targetV = targetDat.get_fdata()
    targetV = vnm.adjust_volume_dir(targetV, targetDat)
    atlasDat = nib.load(opt.atlas[0])
    atlasV = atlasDat.get_fdata()
    atlasV = vnm.adjust_volume_dir(atlasV, atlasDat)

    dbsidxs = []
    for j in range(len(rois)):
        dbsidx = np.unique(atlasV[targetV==rois[j]])
        dbsidxs.append(dbsidx.astype(np.int32))
    if len(dbsidxs) == 0:
        print('error: empty modulation target. bad ROI=' + opt.roi[0])
        exit(-1)

    # get save name from CX
    savename = os.path.splitext(os.path.basename(opt.cx[0]))
    savename = savename[0]
    print('set savename=' + savename)
    CX = []
    mat_net = None
    vnpm = opt.vnparam
    hrfpm = opt.hrfparam

    # load subject time-series (CX)
    print('load subject time-series file: ' + opt.cx[0])
    try:
        dic = sio.loadmat(opt.cx[0])
    except NotImplementedError:  # -v7.3
        dic = h5py.File(opt.cx[0], 'r')
    if dic.get('CX') is None:
        print('no cells of subject time-series (CX) file. please specify .mat file.')
        exit(-1)
    if type(dic) is np.ndarray:
        print('error: old mat file is currently not supported: ' + opt.cx[0])
    else:  # h5py
        cx = dic['CX']
        for j in range(len(cx)):
            hdf5ref = cx[j,0]
            x = dic[hdf5ref]
            CX.append(np.array(x).T)
        dic.close()

    # load group surrogate model (net)
    if len(opt.model) > 0:
        print('load model file: ' + opt.model[0])
        try:
            dic = sio.loadmat(opt.model[0])
        except NotImplementedError:  # -v7.3
            dic = h5py.File(opt.model[0], 'r')
        if dic.get('net') is None:
            print('no group surrogate model file. please specify .mat file.')
            exit(-1)
        if type(dic) is np.ndarray:
            print('error: old mat file is currently not supported: ' + opt.cx[0])
        else:  # h5py
            mat_net = dic['net']
            net = vnm.MultivariateVARNetwork()
            net.init_with_mat(mat_net)
            dic.close()
    elif len(opt.pymodel) > 0:
        print('load model path: ' + opt.pymodel[0])
        net = vnm.MultivariateVARNetwork()
        net.load(opt.pymodel[0])
    else:
        print('no group surrogate model file. please specify .mat file.')
        exit(-1)

    # init
    isMatf = len(opt.in_files) > 0
    if isMatf:
        N = len(opt.in_files)
    else:
        N = opt.out

    if opt.outfrom-1 >= N:
        print('outfrom is too big (' +str(opt.outfrom)+'). please specify less than out num.')
        exit(-1)

    # process each file
    for i in range(opt.outfrom-1, N):
        perm = []
        # read subject permutation file
        if isMatf:
            if not os.path.isfile(opt.in_files[i]):
                print('bad file name. ignore : ' + opt.in_files[i])
                continue

            print('loading subject permutation : ' + opt.in_files[i])
            try:
                dic = sio.loadmat(opt.in_files[i])
            except NotImplementedError:  # -v3.7
                dic = h5py.File(opt.in_files[i], 'r')
            if type(dic) is np.ndarray:
                print('error: old mat file is currently not supported: ' +opt.in_files[i])
            else:  # h5py
                perm = np.array(dic['perm']).T[0]  # Dataset to single array 1xlength
                dic.close()
        else:
            permf = opt.outpath + '/perm' + str(i+1) + '_' + savename + '.mat'
            if os.path.isfile(permf):
                print('loading subject permutation : ' + permf)
                try:
                    dic = sio.loadmat(permf)
                except NotImplementedError:  # -v3.7
                    dic = h5py.File(permf, 'r')
                if type(dic) is np.ndarray:
                    print('error: old mat file is currently not supported: ' + permf)
                else:  # h5py
                    perm = np.array(dic['perm']).T[0]  # Dataset to single array 1xlength
                    perm = perm.astype(np.int32)
                    dic.close()

        if len(perm) == 0 or perm is None:
            # generate subject permutation
            permf = opt.outpath + '/perm' + str(i + 1) + '_' + savename + '.mat'
            perm, uxtime, reslen = vnm.vnm_subject_perm(CX, net.lags)
            matdata = {}
            matdata['perm'] = perm
            matdata['uxtime'] = uxtime
            matdata['reslen'] = reslen
            hdf5storage.write(matdata, filename=permf, matlab_compatible=True)
            print('save perm file : ' + permf)

        # loop for neuromodulation target rois
        for j in range(len(rois)):
            S = []
            roi = rois[j]
            dbsidx = dbsidxs[j]

            # get modulation (add & mul) time-series for vertual neuromodulation
            print('generate modulation (add & mul) time-series, target roi=' + str(roi) + ', srframes=' +str(opt.srframes)+ ', dbsoffsec=' +str(vnpm[0])+ ', dbsonsec=' +str(vnpm[1])+ ', dbspw=' +str(vnpm[2]))
            print('convolution params tr=' +str(opt.tr)+ ', res=' +str(hrfpm[0])+ ', sp=' +str(hrfpm[1]))
            CA, Chrf, CM = vnm.vnm_addmul_signals(CX, dbsidx, opt.surrnum, opt.srframes, vnpm[0], vnpm[1], vnpm[2], opt.tr, hrfpm[0], hrfpm[1])

            # load virtual neuromodulation surrogate
            sessionName = savename+ '_' +str(roi)+ 'sr' +str(opt.surrnum)+ 'pr' +str(i+1)
            outfname = opt.outpath+ '/' +sessionName+ '.mat'
            if os.path.isfile(outfname):
                print('load surrogate file : ' +outfname) # load prev surrogate result
                try:
                    dic = sio.loadmat(outfname)
                except NotImplementedError:  # -v3.7
                    dic = h5py.File(outfname, 'r')
                if type(dic) is np.ndarray:
                    print('error: old mat file is currently not supported')
                else:  # h5py
                    s = dic['S']
                    for k in range(len(s)):
                        hdf5ref = s[k,0]
                        x = dic[hdf5ref]
                        S.append(np.array(x).T)
                    dic.close()

            if len(S) == 0:
                # calc virtual neuromodulation VAR surrogate
                print('calc virtual neuromodulation surrogate. roi=' +str(roi)+ ', surrnum=' +str(opt.surrnum))
                S = vnm.vnm_var_surrogate(net, CX, CA, CM, perm, opt.surrnum, opt.srframes)

                # Save the dictionary to a .mat file
                # use 'matlab_compatible=True' to ensure it can be read by MATLAB
                if not opt.nocache:
                    matdata = {}
                    matdata['S'] = S
                    hdf5storage.write(matdata, filename=opt.outpath+'/'+sessionName+'.mat', matlab_compatible=True)
                    print('save virtual neuromodulation surrogate file : ' +outfname)

            # calcurate 2nd level GLM and save nifti file
            if opt.glm:
                # load cube atlas file
                atlasDat = nib.load(opt.atlas[0])
                atlasV = atlasDat.get_fdata()
                atlasV = vnm.adjust_volume_dir(atlasV, atlasDat)

                tuM = 8 # GLM tukey-taper size
                betaBmat = opt.outpath +'/'+ sessionName +'_2nd-Tukey' + str(tuM) +'.mat'
                if os.path.isfile(betaBmat):
                    print('load 2nd level GLM result file : ' + betaBmat)  # load prev surrogate result
                    try:
                        dic = sio.loadmat(betaBmat)
                    except NotImplementedError:  # -v3.7
                        dic = h5py.File(betaBmat, 'r')
                    if type(dic) is np.ndarray:
                        print('error: old mat file is currently not supported')
                    else:  # h5py
                        B = np.array(dic['B']).T
                        RSS = np.array(dic['RSS']).T
                        X2is = np.array(dic['X2is']).T
                        tRs = np.array(dic['tRs']).T
                        df = np.array(dic['df']).item()
                        dic.close()
                else:
                    # calc 1st-level GLM
                    print('calc 1st-level GLM...')

                    surrNum = len(S)
                    bmatC = [None] * surrNum
                    for k in range(surrNum):
                        Xorg = Chrf[k]
                        Xt = np.concatenate([Xorg, np.ones((Xorg.shape[0], 1), dtype=np.float32)],1)
                        Sk = np.squeeze(S[k])
                        B2, RSS, df, _, _ = vnm.tukey_mp(Sk.T, Xt, tuM=tuM, isOutX2is=False)
                        bmatC[k] = B2
#                        lp = LineProfiler() # check profile
#                        lp_wrapper = lp(glm.tukey.calc)
#                        lp_wrapper(S[k][:,:,0].T, Xt, tuM=tuM, isOutX2is=False)
#                        lp.print_stats()

#                    dic = sio.loadmat('tempGLM1st.mat')
#                    bmatC = dic['bmatC']

                    # calc 2nd-level estimation
                    print('calc 2nd-level GLM...')
                    B1 = bmatC[0][:,[0,1]].T
                    X2 = np.eye(B1.shape[0])
                    for k in range(1,surrNum):
                        # 2nd-level Y vector
                        B2 = bmatC[k][:,[0,1]].T # include design and intercept(we need more than 8 length for tukey taper)
                        B1 = np.concatenate([B1, B2],0)

                        # 2nd-level design matrix
                        X2 = np.concatenate([X2,np.eye(B2.shape[0])],0)

                    B1[np.isnan(B1)] = 0  # there might be nan

                    # calc 2nd-level estimation
                    B, RSS, df, X2is, tRs = vnm.tukey_mp(B1, X2, tuM=tuM, isOutX2is=True, n_jobs=opt.njobs)

                    # Save the dictionary to a .mat file
                    # use 'matlab_compatible=True' to ensure it can be read by MATLAB
                    if not opt.nocache:
                        matdata = {}
                        matdata['B'] = B
                        matdata['RSS'] = RSS
                        matdata['X2is'] = X2is
                        matdata['tRs'] = tRs
                        matdata['df'] = df
                        hdf5storage.write(matdata, filename=betaBmat, matlab_compatible=True)
                        print('save 2nd level GLM result file : ' + betaBmat)

                # GLM contrast images
                contrasts = [np.array([1,0]).T] # GLM contrust
                Ts = vnm.contrast_image(contrasts, B, RSS, X2is, tRs) # this is fast enough
                V2 = vnm.roi_ts_to4dimage(Ts[0], atlasV)  # returns 4D image. this is slow
                V2 = np.squeeze(V2)

                # output nifti file
                V2 = vnm.adjust_volume_dir(V2.astype(np.float32), atlasDat)
                nifti_image = nib.Nifti1Image(V2, atlasDat.affine)
                outniiname = opt.outpath +'/'+ sessionName +'_2nd-Tukey' + str(tuM) +'.nii.gz'
                nib.save(nifti_image, outniiname)
                print('save nifti file : '+outniiname)
