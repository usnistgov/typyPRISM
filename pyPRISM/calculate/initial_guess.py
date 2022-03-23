#!python
from __future__ import division,print_function
from pyPRISM.core.PairTable import PairTable
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.Space import Space
import warnings
import numpy as np

def initial_guess(PRISM):
    r'''Calculate an initial guess from a previously solved PRISM object. This is intended to construct an initial guess from a reference system for use with molecular closures

    Parameters
    ----------
    PRISM: pyPRISM.core.PRISM
        A **solved** PRISM object.
    
    Returns
    -------
    guess.reshape((-1,)): array of guess values
        array of guess values for h_ij - w_i*c_ij*w_j, where * is a convolution integral
    '''

    if PRISM.directCorr.space == Space.Real:
        PRISM.sys.domain.MatrixArray_to_fourier(PRISM.directCorr)

    if PRISM.omegaConvolution.space == Space.Real:
        PRISM.sys.domain.MatrixArray_to_fourier(PRISM.omegaConvolution)

    if PRISM.totalCorr.space == Space.Real:
        PRISM.sys.domain.MatrixArray_to_fourier(PRISM.totalCorr)
    
    x = MatrixArray(length=PRISM.sys.domain.length,rank=PRISM.sys.rank,space=Space.Fourier,types=PRISM.sys.types)

    for i,t1 in enumerate(PRISM.sys.types):
        for j,t2 in enumerate(PRISM.sys.types):
            x[t1,t2] = PRISM.totalCorr[t1,t2] - PRISM.omegaConvolution[t1,t1] * PRISM.directCorr[t1,t2] * PRISM.omegaConvolution[t2,t2]

    PRISM.sys.domain.MatrixArray_to_real(x)

    guess = x.data

    return guess.reshape((-1,))
        
        
