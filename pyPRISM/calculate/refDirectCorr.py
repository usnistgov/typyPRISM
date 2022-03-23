#!python
from __future__ import division,print_function
from pyPRISM.core.PairTable import PairTable
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.Space import Space
import warnings
import numpy as np

def refDirectCorr(PRISM):
    r'''Calculate the reference direct correlation function for molecular closures

    Parameters
    ----------
    PRISM: pyPRISM.core.PRISM
        A **solved** PRISM object.
    
    Returns
    -------
    cr0.reshape((-1,)): array of values
        array of direct correlation values

    '''

    if PRISM.directCorr.space == Space.Fourier:
        PRISM.sys.domain.MatrixArray_to_real(PRISM.directCorr)

    cr0 = PRISM.directCorr.data

    return cr0.reshape((-1,))
        
        
