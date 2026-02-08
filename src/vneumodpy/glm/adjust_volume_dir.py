# -*- coding: utf-8 -*-
##
# Adjust NIfTI volume direction based on NIfTI info Transpose matrix
# returns adjusted NIfTI volume
# input:
#  V            nifti 4D volume (X x Y x Z x frames)

from __future__ import print_function, division   # for Python 2 compatible

import numpy as np


def adjust_volume_dir(V, info):
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    if np.sum(np.abs(A * info.affine[:3, :3])) > 0:
        print('nifti volume transformation is not supported.')
        return V
    if info.affine[0, 0] < 0:  # check flip X axis
        V = np.flipud(V).copy()
    if info.affine[1, 1] < 0:  # check flip Y axis
        V = np.fliplr(V).copy()
    if info.affine[2, 2] < 0:  # check flip Z axis
        V = np.flip(V, axis=2).copy()
    return V
