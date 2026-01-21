# -*- coding: utf-8 -*-
# resampling NIfTI volume.
# returns re-sampled volume (outV)
# input:
#  V            nifti 3D volume (X x Y x Z)
#  stepXY       XY axes resampling rate
#  stepZ        Z axes resampling rate
#  operation    operation for each plane ('mode'(default),'max','min','mean','median')

import numpy as np
from scipy.stats import mode

def resampling_nifti_volume(V, stepXY, stepZ, operation='mode'):
    xs = int(np.floor(V.shape[0] / stepXY))
    ys = int(np.floor(V.shape[1] / stepXY))
    zs = int(np.floor(V.shape[2] / stepZ))
    outV = np.zeros((xs, ys, zs), dtype=np.float32)

    if stepXY >= 1 and stepZ >= 1:
        # scale down
        for z in range(1,zs+1):
            for y in range(1,ys+1):
                for x in range(1,xs+1):
                    x_start = int(round(x * stepXY - (stepXY - 1))) - 1
                    x_end = int(round(x * stepXY))
                    y_start = int(round(y * stepXY - (stepXY - 1))) - 1
                    y_end = int(round(y * stepXY))
                    z_start = int(round(z * stepZ - (stepZ - 1))) - 1
                    z_end = int(round(z * stepZ))
                    A = V[x_start:x_end, y_start:y_end, z_start:z_end]

                    if operation == 'mode':
                        A_flat = A[~np.isnan(A)]
                        if A_flat.size == 0:
                            m = np.nan
                        else:
                            m = mode(A_flat, axis=None).mode[0]
                    elif operation == 'min':
                        m = np.nanmin(A)
                    elif operation == 'max':
                        m = np.nanmax(A)
                    elif operation == 'mean':
                        m = np.nanmean(A)
                    elif operation == 'median':
                        m = np.nanmedian(A)
                    else:
                        raise ValueError(f"Unsupported operation: {operation}")

                    outV[x-1, y-1, z-1] = m
    else:
        # scale up
        out_shape = (
            int(np.ceil(V.shape[0] / stepXY)),
            int(np.ceil(V.shape[1] / stepXY)),
            int(np.ceil(V.shape[2] / stepZ))
        )
        outV = np.zeros(out_shape, dtype=np.float32)

        for z in range(1,V.shape[2]+1):
            for y in range(1,V.shape[1]+1):
                for x in range(1,V.shape[0]+1):
                    m = V[x-1, y-1, z-1]
                    xx_start = int(round((x-1) / stepXY))
                    xx_end = int(round(x / stepXY))
                    yy_start = int(round((y-1) / stepXY))
                    yy_end = int(round(y / stepXY))
                    zz_start = int(round((z-1) / stepZ))
                    zz_end = int(round(z / stepZ))
                    outV[xx_start:xx_end, yy_start:yy_end, zz_start:zz_end] = m

    return outV