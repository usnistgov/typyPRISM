#!python
from __future__ import division,print_function
from typyPRISM.core.PairTable import PairTable
from typyPRISM.core.MatrixArray import MatrixArray
from typyPRISM.core.Space import Space
from typyPRISM.calculate.structure_factor import structure_factor
import numpy as np

def solvation_potential(PRISM,closure='HNC'):
    r'''Calculate the pairwise decomposed medium-induced solvation potential
    
        
    Parameters
    ----------
    PRISM: typyPRISM.core.PRISM
        A **solved** PRISM object.

    closure: str ('PY' or 'HNC')
        closure used to derive the potential 

    
    Returns
    -------
    psi: typyPRISM.core.MatrixArray
        MatrixArray of the *Real-space* solvation potentials


    **Mathematical Definition**

    .. math::

        \text{PY: } \Delta \hat{\Psi}^{PY}(k) = - k_B T \log(1 + \hat{C}(k)\hat{S}(k)\hat{C}(k))

    .. math::

        \text{HNC: } \Delta \hat{\Psi}^{HNC}(k) = - k_B T \hat{C}(k)\hat{S}(k)\hat{C}(k)

    
    **Variable Definitions**

        - :math:`\Delta \hat{\Psi}^{PY}`, :math:`\Delta \hat{\Psi}^{HNC}`
            Percus-Yevick and Hypernetted Chain derived pairwise decomposied
            solvation potential MatrixArrays. This implies that the
            multiplication in the above equation is actually *matrix*
            multiplication and the individual solvation potentials are
            extracted as curves of the MatrixArrays. Note that the solvation
            potential MatrixArrays are inverted back to Real-space for use. 

        - :math:`\hat{C}(k)`
            Direct correlation function MatrixArray at a wavenumber :math:`k`

        - :math:`\hat{S}(k)`
            Structure factor MatrixArray at a wavenumber :math:`k`

        - :math:`k_B T`
            Thermal temperature written as the product of the Boltzmann
            constant and temperature.

    **Description**

        To be added...


    .. warning::

        Passing an unsolved PRISM object to this function will still produce
        output based on the default values of the attributes of the PRISM
        object.
    

    Example
    -------
    .. code-block:: python

        import typyPRISM

        sys = typyPRISM.System(['A','B'])

        # ** populate system variables **
        
        PRISM = sys.createPRISM()

        PRISM.solve()

        psi = typyPRISM.calculate.solvation_potential(PRISM)

        psi_BB = psi['B','B']
    
    '''
    
    assert PRISM.sys.rank>1,'the psi calculation is only valid for multicomponent systems'
    
    if PRISM.directCorr.space == Space.Real:
        PRISM.domain.MatrixArray_to_fourier(PRISM.directCorr)

    if PRISM.totalCorr.space == Space.Real:
        PRISM.sys.domain.MatrixArray_to_fourier(PRISM.totalCorr)
        
    if PRISM.omega.space == Space.Real:
        PRISM.sys.domain.MatrixArray_to_fourier(PRISM.omega)

    structureFactor = structure_factor(PRISM)
    #(PRISM.totalCorr*PRISM.sys.pairDensityMatrix + PRISM.omega)/PRISM.sys.siteDensityMatrix

    if closure == 'HNC':
        psi = PRISM.directCorr.dot(structureFactor).dot(PRISM.directCorr) * -PRISM.sys.kT 
    elif closure == 'PY':
        psi = PRISM.directCorr.dot(structureFactor).dot(PRISM.directCorr)
        psi.data = np.log(1 + psi.data) * -PRISM.sys.kT

    PRISM.sys.domain.MatrixArray_to_real(psi)

    return psi