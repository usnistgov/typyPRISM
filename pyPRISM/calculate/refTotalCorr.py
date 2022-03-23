#!python
from __future__ import division,print_function
from pyPRISM.core.PairTable import PairTable
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.Space import Space
import warnings
import numpy as np

def refTotalCorr(PRISM):
    r'''return the reference total correlation function for molecular closures

    Parameters
    ----------
    PRISM: pyPRISM.core.PRISM
        A **solved** PRISM object.
    
    Returns
    -------
    hk0.reshape((-1,)): array of values
        array of all reference direct correlation values
    '''
    if PRISM.totalCorr.space == Space.Real:
        PRISM.sys.domain.MatrixArray_to_fourier(PRISM.totalCorr)

    hk0 = PRISM.totalCorr.data

    return hk0.reshape((-1,))
        
        
