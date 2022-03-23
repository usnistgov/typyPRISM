#!python 
from __future__ import division,print_function
import warnings
import numpy as np
from itertools import product
from pyPRISM.core.PRISM import PRISM
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.PairTable import PairTable
from pyPRISM.core.ValueTable import ValueTable
from pyPRISM.core.Space import Space
from pyPRISM.core.Density import Density
from pyPRISM.core.Diameter import Diameter

from pyPRISM.closure.AtomicClosure import AtomicClosure
from pyPRISM.closure.MolecularClosure import MolecularClosure

class System:
    '''Primary class used to spawn PRISM calculations

    **Description**

        The system object contains tables that fully describe a system to be
        simulated. This includes the domain definition, all site densities,
        site diameters, interaction potentials, intra-molecular correlation
        functions (:math:`\hat{\omega}(k)`), and closures. This class also
        contains a convenience function for spawning a PRISM object. 
    
    
    Example
    -------
    .. code-block:: python

        import pyPRISM

        sys = pyPRISM.System(['A','B'])
        
        sys.domain = pyPRISM.Domain(dr=0.1,length=1024)

        sys.density['A'] = 0.1
        sys.density['B'] = 0.75

        sys.diameter[sys.types] = 1.0
        
        sys.closure[sys.types,sys.types] = pyPRISM.closure.PercusYevick()

        sys.potential[sys.types,sys.types] = pyPRISM.potential.HardSphere()
        
        sys.omega['A','A'] = pyPRISM.omega.SingleSite()
        sys.omega['A','B'] = pyPRISM.omega.InterMolecular()
        sys.omega['B','B'] = pyPRISM.omega.Gaussian(sigma=1.0,length=10000)
        
        PRISM = sys.createPRISM()

        PRISM.solve()

    '''
    def __init__(self,types,kT=1.0):
        r'''
        Arguments
        ---------
        types: list
            Lists of the site types that define the system

        kT: float
            Thermal temperature where :math:`k` is the Boltzmann constant and
            :math:`T` temperature. This is typicaly specified in reduced units
            where :math:`k_{B}=1.0`.


        Attributes
        ----------
        types: list
            List of site types
            
        rank: int
            Number of site types
        
        density: :class:`pyPRISM.core.Density`
            Container for all density values
            
        potential: :class:`pyPRISM.core.PairTable`
            Table of pair potentials between all site pairs in real space
            
        closure: :class:`pyPRISM.core.PairTable`
            Table of closures between all site pairs
            
        omega: :class:`pyPRISM.core.PairTable`
            Table of omega correlation functions in Fourier-space
        
        domain: :class:`pyPRISM.core.Domain`
            Domain object which specifies the Real and Fourier space 
            solution grid.
            
        kT: float
            Value of the thermal energy level. Used to vary temperature and
            scale the potential energy functions.

        diameter: :class:`pyPRISM.core.ValueTable`
            Site diameters. 

            .. warning::
            
                These diameters are currently only passed to the closures. They
                are not passed to potentials and it is up to the user to set
                sane \sigma values that match these diameters. 
        '''

        self.types = types
        self.rank  = len(types)
        self.kT = kT
        
        self.domain    = None
        self.diameter  = Diameter(types)
        self.density   = Density(types)
        self.potential = PairTable(types,'potential')
        self.closure   = PairTable(types,'closure')
        self.omega     = PairTable(types,'omega')

    def check(self):
        '''Is everything in the system specified?

        Raises
        ------
        ValueError if all values are not set
        
        '''
        for table in [self.density,self.potential,self.closure,self.omega,self.diameter]:
            table.check()
        
        if self.domain is None:
            raise ValueError(('System has no domain! '
                              'User must instatiate and assign a domain to the system!'))

        tol = 1e-6
        # ensure that the diameters are on the real-space grid
        for i,t,d in self.diameter.diameter:
            check = np.any(np.abs(self.domain.r - d)<tol)
            if not check:
                warn_text  = 'Diameter for site {} = {} is not a multiple '.format(t,d)
                warn_text += 'of domain grid spacing dr = {}. '.format(self.domain.dr)
                warn_text += 'Rounding will occur in closures, potentials, and omega!'
                warnings.warn(warn_text)

        # ensure that the sigmas are on the real-space grid
        for i,(t1,t2),s in self.diameter.sigma:
            check = np.any(np.abs(self.domain.r - s)<tol)
            if not check:
                warn_text  = 'Sigma for pair {}-{} = {} is not a multiple '.format(t1,t2,s)
                warn_text += 'of domain grid spacing dr = {}. '.format(self.domain.dr)
                warn_text += 'Rounding will occur in closures, potentials, and omega!'
                warnings.warn(warn_text)

    def createPRISM(self):
        '''Construct a PRISM object

        .. note::

            This method calls :func:`~pyPRISM.core.System.System.check` before creating the PRISM object.

        Returns
        -------
        PRISM: pyPRISM.core.PRISM
            Fully specified PRISM object
            
        '''
        self.check() #sanity check
        
        return PRISM(self)

    def iterpairs(self,full=False,diagonal=True):
        '''Convenience function for looping over type pairs.
        
        Parameters
        ----------
        full: bool
            If *True*, all i,j pairs (upper and lower diagonal) will be looped over
            
        diagonal: bool
            If *True*, only the i==j (on-diagonal) pairs will be considered when looping
        
        '''
        
        if full:
            test = lambda i,j: True
        elif diagonal:
            test = lambda i,j: i<=j
        else:
            test = lambda i,j: i<j

        for (i,t1),(j,t2) in product(enumerate(self.types),enumerate(self.types)):
            if test(i,j):
                yield (i,j),(t1,t2)
            
    def solve_picard(self,*args,**kwargs):
        '''Construct a PRISM object and attempt a numerical solution

        .. note::

            See :func:`~pyPRISM.core.PRISM.PRISM.solve` for arguments to this function

        .. note::

            This method calls :func:`~pyPRISM.core.System.System.check` before creating the PRISM object.
        
        Returns
        -------
        PRISM: pyPRISM.core.PRISM
            **Solved** PRISM object
            
        '''
        self.check() #sanity check

        p = PRISM(self)

        p.solve_picard(*args,**kwargs)
        
        return p

    def solve(self,*args,**kwargs):
        '''Construct a PRISM object and attempt a numerical solution

        .. note::

            See :func:`~pyPRISM.core.PRISM.PRISM.solve` for arguments to this function

        .. note::

            This method calls :func:`~pyPRISM.core.System.System.check` before creating the PRISM object.
        
        Returns
        -------
        PRISM: pyPRISM.core.PRISM
            **Solved** PRISM object
            
        '''
        self.check() #sanity check

        p = PRISM(self)

        p.solve(*args,**kwargs)
        
        return p
        
