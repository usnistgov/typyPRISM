#!python
from __future__ import division,print_function
from pyPRISM.core.Domain import Domain
from pyPRISM.closure.MolecularClosure import MolecularClosure
import numpy as np
import warnings 

class ReferenceMolecularMeanSphericalApproximation(MolecularClosure):
    r'''Reference Molecular Mean Spherical Approximation closure

    **Mathematial Definition**

        .. math:: c_{\alpha,\beta}(r) = -U_{\alpha,\beta}(r)


    **Variables Definitions**

        - :math:`c_{\alpha,\beta}(r)`
            Direct correlation function value at distance :math:`r` between
            sites :math:`\alpha` and :math:`\beta`.
    
        - :math:`U_{\alpha,\beta}(r)`
            Interaction potential value at distance :math:`r` between sites
            :math:`\alpha` and :math:`\beta`.
    

    **Description**
        
        The Mean Spherical Approximation (MSA) closure assumes an interaction
        potential that contains a hard-core interaction and a tail interaction.
        See Reference [1] for a derivation and discussion of this closure.
        
        The MSA does a good job of describing the properties of the square-well 
        fluid, and allows for the analytical solution of the PRISM/RISM 
        equations for some systems. The MSA closure reduces to the PercusYevick 
        closure if the tail is ignored.

    
    References
    ----------
    #. Hansen, J.P.; McDonald, I.R.; Theory of Simple Liquids; Chapter 4, Section 4; 
       4th Edition (2013), Elsevier [`link
       <https://www.sciencedirect.com/science/book/9780123870322>`__]

    Example
    -------
    .. code-block:: python

        import pyPRISM

        sys = pyPRISM.System(['A','B'])
        
        sys.closure['A','A'] = pyPRISM.closure.PercusYevick()
        sys.closure['A','B'] = pyPRISM.closure.PercusYevick()
        sys.closure['B','B'] = pyPRISM.closure.MeanSphericalApproximation()

        # ** finish populating system object **

        PRISM = sys.createPRISM()

        PRISM.solve()
    
    '''
    def __init__(self,apply_hard_core=False):
        '''Contstructor

        Parameters
        ----------
        apply_hard_core: bool
            If *True*, the total correlation function will be assumed to be -1
            inside the core (:math:`r_{i,j}<(d_i + d_j)/2.0`) and the closure
            will not be applied in this region.
        '''
        self.name = "RMMSA"
        self.potential = None
        self.value = None
        self.sigma = None
        self.apply_hard_core = apply_hard_core

        if apply_hard_core == False:
            warnings.warn(
                    '''The MSA closure does not work for divergent potentials
                    when the hard core condition is not manually applied. This
                    will likely result in a cryptic crash of the simulation if
                    attempted. Using MSA(apply_hard_core=True) will avoid this
                    warning. This warning should be ignored if hard-core
                    interactions are not being used.''')

        
    def __repr__(self):
        return '<MolecularClosure:ReferenceMolecularMeanSphericalApproximation>'
    
    def calculate(self,domain,gamma,omega_k_i,omega_k_j,cr0,hk0,hk):
        '''Calculate direct correlation function based on supplied :math:`\gamma`

        Arguments
        ---------
        r: np.ndarray
            array of real-space values associated with :math:`\gamma`

        gamma: np.ndarray
            array of :math:`\gamma` values used to calculate the direct
            correlation function
        
        '''
        r=domain.r        

        assert self.potential is not None,'Potential for this closure is not set!'
        
        assert len(gamma) == len(self.potential),'Domain mismatch!'
    
        potential_calculation = self.potential 

        mask = r<=self.sigma #we need to set the potential for r<=sigma equal to 0 for the fft to work correctly
        potential_calculation[mask]=0.0

        cr0_k = Domain.to_fourier(domain,array=cr0)
        convoluted_cr0_k = omega_k_i*cr0_k*omega_k_j
        convoluted_cr0=Domain.to_real(domain,array=convoluted_cr0_k)

        potential_k = Domain.to_fourier(domain,array=potential_calculation)
        convoluted_potential_k = omega_k_i*potential_k*omega_k_j
        convoluted_potential = Domain.to_real(domain,array=convoluted_potential_k)

        if self.apply_hard_core:
            # apply hard core condition 
            self.value = -1 - gamma
            
            # calculate closure outside hard core
            mask = r>self.sigma
            
            # self.value is the convoluted c(r)
            self.value[mask] = convoluted_cr0[mask] - convoluted_potential[mask]
                    
        else:
           # self.value is the convoluted c(r)

            self.value = convoluted_cr0 - convoluted_potential
        
        return self.value

class RMMSA(ReferenceMolecularMeanSphericalApproximation):
    '''Alias of ReferenceMolecularMeanSphericalApproximation'''
    pass
