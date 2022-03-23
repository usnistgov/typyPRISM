#!python
from __future__ import division,print_function

from pyPRISM.core.Space import Space
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.IdentityMatrixArray import IdentityMatrixArray
from pyPRISM.closure.AtomicClosure import AtomicClosure
from pyPRISM.closure.MolecularClosure import MolecularClosure

from scipy.optimize import root

import numpy as np

from copy import deepcopy
import warnings

class PRISM:
    r'''Primary container for a storing a PRISM calculation
    
    Each pyPRISM.PRISM object serves as an encapsulation of a fully specified
    PRISM problem including all inputs needed for the calculation and the
    function to be numerically minimized. 
    
    Attributes
    ----------
    domain: pyPRISM.Domain
        The Domain object fully specifies the Real- and Fourier- space solution
        grids.
    
    directCorr: pyPRISM.MatrixArray
        The direct correlation function for all pairs of site types 

    directCorrForCost: pyPRISM.MatrixArray
        This is the direct correlation function for the cost function. For atomic closures, this is equal to the direct correlation function. However, for molecular closures, this is equal to a  
        convolution of the direct correlation function with the appropriate intra-molecular correlation function.

    referenceDirectCorr: pyPRISM.MatrixArray
        Thie reference direct correlation function for use in molecular closures.
        
    omega: pyPRISM.MatrixArray
        The intra-molecular correlation function for all pairs of site types

    omegaConvolution: pyPRISM.MatrixArray
        The intra-molecular correlation function for all pairs of site types for convolution purposes. This is differenct from omega, because it is not normalized by site density
    
    closure: pyPRISM.core.PairTable of pyPRISM.closure.Closure
        Table of closure objects used to generate the direct correlation
        functions (directCorr)
        
    pairCorr: pyPRISM.MatrixArray
        The *inter*-molecular pair correlation functions for all pairs of
        site types. Also commonly refered to as the radial distribution functions.
    
    totalCorr: pyPRISM.MatrixArray
        The *inter*-molecular total correlation function is simply the pair
        correlation  function y-shifted by 1.0 i.e.  totalCorr = pairCorr - 1.0

    referenceTotalCorr: pyPRISM.MatrixArray
        The *inter*-molecular total correlation function for the reference system. This is necssary for certain molecular closures such as the reference Laria, Wu, Chandler closure.
        
    potential: pyPRISM.MatrixArray
        Interaction potentials for all pairs of sites
        
    GammaIn,GammaOut: pyPRISM.MatrixArray
        Primary inputs and outputs of the PRISM cost function. Gamma is defined as
        "totalCorr - directCorrForCost" (in Fourier space) and results from a change
        of variables used to remove divergences in the closure relations. 
    
    OC,IOC,I,etc: pyPRISM.MatrixArray
        Various MatrixArrays used as intermediates in the PRISM functional.
        These arrays are pre-allocated and stored for efficiency. 
    
    x1,x2,y: float np.ndarray
        Current inputs and outputs of the cost function
    
    pairDensityMatrix: float np.ndarray
        Rank by rank array of pair densities between sites. See :class:`pyPRISM.core.Density`
            
    siteDensityMatrix: float np.ndarray
        Rank by rank array of site densities. See :class:`pyPRISM.core.Density`
    
    Methods
    -------
    cost:
        Primary cost function used to define the criteria of a "converged"
        PRISM solution. The numerical solver will be given this function 
        and will attempt to find the inputs (self.x) that make the outputs
        (self.y) as close to zero as possible. 
        
        
    '''
    def __init__(self,sys):
        self.sys = deepcopy(sys)

        # Need to set the potential for each closure object
        for (i,j),(t1,t2),U in self.sys.potential.iterpairs():
            #only set sigma if not set directly in potential
                if U.sigma is None:
                    U.sigma = self.sys.diameter[t1,t2]
                self.sys.closure[t1,t2].sigma = self.sys.diameter[t1,t2]
                self.sys.closure[t1,t2].potential = U.calculate(self.sys.domain.r) / self.sys.kT
        
        #cost function inputs and output
        self.x1         = np.zeros(sys.rank*sys.rank*sys.domain.length) #this is the cost function input
        self.x2         = np.zeros(sys.rank*sys.rank*sys.domain.length) #this is the reference direct correlation function input
        self.x3         = np.zeros(sys.rank*sys.rank*sys.domain.length) #this is the reference total correlation function input
        self.y         = np.zeros(sys.rank*sys.rank*sys.domain.length)

        # The omega objects must be converted to a MatrixArray of the actual correlation
        # function values rather than a table of OmegaObjects.
        applyFunc = lambda x: x.calculate(sys.domain.k)
        self.omega  = self.sys.omega.apply(applyFunc,inplace=False).exportToMatrixArray(space=Space.Fourier)
        self.omegaConvolution  = self.sys.omega.apply(applyFunc,inplace=False).exportToMatrixArray(space=Space.Fourier)
        self.omega *= sys.density.site #omega should always be scaled by site density 
	
        # Spaces are set based on when they are used in self.cost(...). In some cases,
        # this is redundant because these array's will be overwritten with copies and
        # then their space will be inferred from their parent MatrixArrays
        self.directCorr = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.directCorrForCost = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.referenceDirectCorr = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.totalCorr  = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        self.referenceTotalCorr  = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        self.GammaIn    = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.GammaOut   = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.OC         = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        self.I          = IdentityMatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)

    def __repr__(self):
        return '<PRISM length:{} rank:{}>'.format(self.sys.domain.length,self.sys.rank)

    def cost(self,x1,x2,x3):
        r'''Cost function 
        
        There are likely several cost functions that could be imagined using
        the PRISM equations. In this case we formulate a self-consistent 
        formulation where we expect the input of the PRISM equations to be
        identical to the output. 

        .. image:: ../../img/numerical_method.svg
            :width: 300px
        
        The goal of the solve method is to numerically optimize the input (:math:`\gamma_{in}`) 
        so that the output (:math:`(\gamma_{in}-\gamma_{out})`) is minimized to zero.
        
        '''
        self.x1 = x1 #store input
        self.x2 = x2
        self.x3 = x3

        # The np.copy is important otherwise x saves state between calls to
        # this function.
        self.GammaIn.data = np.copy(x1.reshape((-1,self.sys.rank,self.sys.rank)))

        self.referenceDirectCorr.data = np.copy(x2.reshape((-1,self.sys.rank,self.sys.rank)))

        self.referenceTotalCorr.data = np.copy(x3.reshape((-1,self.sys.rank,self.sys.rank)))

        # directCorr is calculated directly in Real space but immediately 
        # inverted to Fourier space. We must reset this from the last call.
        self.directCorr.space = Space.Real
        self.directCorrForCost.space = Space.Real
        for (i,j),(t1,t2),closure in self.sys.closure.iterpairs():
            if isinstance(closure,AtomicClosure):
                self.directCorrForCost[t1,t2] = closure.calculate(self.sys.domain.r,self.GammaIn[t1,t2])
                self.directCorr[t1,t2] = self.directCorrForCost[t1,t2]
            elif isinstance(closure,MolecularClosure):
                self.directCorrForCost[t1,t2] = closure.calculate(self.sys.domain,self.GammaIn[t1,t2],self.omegaConvolution[t1,t1],self.omegaConvolution[t2,t2],self.referenceDirectCorr[t1,t2],self.referenceTotalCorr[t1,t2],self.totalCorr[t1,t2])
                dummy = self.sys.domain.to_fourier(self.directCorrForCost[t1,t2]) 
                dummy = dummy/(self.omegaConvolution[t1,t1]*self.omegaConvolution[t2,t2])
                dummy = self.sys.domain.to_real(dummy)
                self.directCorr[t1,t2] = dummy
            else:
                raise ValueError('Closure type not recognized')
        
        self.sys.domain.MatrixArray_to_fourier(self.directCorr)
        self.sys.domain.MatrixArray_to_fourier(self.directCorrForCost)
	
        self.OC = self.omega.dot(self.directCorr) #CHANGE
        self.IOC = self.I - self.OC
        self.IOC.invert(inplace=True)
        
        self.totalCorr  = self.IOC.dot(self.OC).dot(self.omega)

        self.totalCorr /= self.sys.density.pair
        
        self.GammaOut  = self.totalCorr - self.directCorrForCost

        self.sys.domain.MatrixArray_to_real(self.GammaOut)

        self.y = self.GammaOut.data - self.GammaIn.data

        return self.y.reshape((-1,))

    def solve_picard(self,guess=None,step=None,tol=None,cr0=None,hk0=None,hk_initial=None): 
        '''Attempt to numerically solve the PRISM equations using Picard iteration
        
        Using the supplied inputs (in the constructor), we attempt to numerically
        solve the PRISM equations using the scheme laid out in :func:`cost`. If the 
        numerical solution process is successful, the attributes of this class
        will contain the solved values for a given input i.e. self.totalCorr will
        contain the numerically optimized (solved) total correlation functions.

        This function also does basic checks to ensure that the results are 
        physical. At this point, this consists of checking to make sure that
        the pair correlation functions are not negative. If this isn't true
        a warning is issued to the user. 
        
        Parameters
        ----------
        guess: np.ndarray, size (rank*rank*length)
            The initial guess of :math:`\gamma` to the numerical solution process.
            The numpy array should be of size rank x rank x length corresponding to 
            the a full flattened MatrixArray. If not specified, an initial guess
            of all zeros is used. 
            
        step: np.float
            New solutions are mixed with old solutions. This is the fraction of the new solution to use where 0 is none and 1 is all of the new solution.
            
        tol: np.float
            The minimum difference required between the new and old solutions to declare solution converged.

        cr0: np.ndarray, size (rank*rank*length)
            The reference direct correlation functions
        
        hk0: np.ndarray, size (rank*rank*length)
            The reference total correlation functions
        
        '''
        
        if guess is None:
            guess = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)

        if step is None:
            step = 0.1

        if tol is None:
            tol = 1.0E-6

        if cr0 is None:
            cr0 = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)

        if hk0 is None:
            hk0 = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)

        if hk_initial is None:
            hk_initial = hk0

        self.totalCorr.data = np.copy(hk_initial.reshape((-1,self.sys.rank,self.sys.rank)))

        input_solution = guess
        error = 1.0
        
        counter = 0
        while error > tol:
            self.minimize_result = self.cost(input_solution,cr0,hk0)#cost should return a new output solutions, which will be mixed with the input solution until convergence
            test_solution = np.copy((step*self.minimize_result) + (input_solution))
            error = np.amax(np.sum(np.abs(input_solution-test_solution),axis=0))
            print("counter:",counter,error)
            counter = counter + 1
            input_solution = np.copy(test_solution)
        
        if self.totalCorr.space == Space.Fourier:
            self.sys.domain.MatrixArray_to_real(self.totalCorr)

        tol = 1e-5
        warnstr = 'Pair correlations are negative (value = {:3.2e}) for {}-{} pair!'
        for i,(t1,t2),H in self.totalCorr.iterpairs():
            if np.any(H<-(1.0+tol)):
                val = np.min(H)
                warnings.warn(warnstr.format(val,t1,t2))
        
        return self.minimize_result

    def solve(self,guess=None,method='krylov',options=None,tol=None,cr0=None,hk0=None,hk_initial=None):
        '''Attempt to numerically solve the PRISM equations
        
        Using the supplied inputs (in the constructor), we attempt to numerically
        solve the PRISM equations using the scheme laid out in :func:`cost`. If the 
        numerical solution process is successful, the attributes of this class
        will contain the solved values for a given input i.e. self.totalCorr will
        contain the numerically optimized (solved) total correlation functions.

        This function also does basic checks to ensure that the results are 
        physical. At this point, this consists of checking to make sure that
        the pair correlation functions are not negative. If this isn't true
        a warning is issued to the user. 
        
        Parameters
        ----------
        guess: np.ndarray, size (rank*rank*length)
            The initial guess of :math:`\gamma` to the numerical solution process.
            The numpy array should be of size rank x rank x length corresponding to 
            the a full flattened MatrixArray. If not specified, an initial guess
            of all zeros is used. 
            
        method: string
            Set the type of optimization scheme to use. The scipy documentation
            for `scipy.optimize.root
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html>`__
            details the possible values for this parameter. 
            

        options: dict
            Dictionary of options specific to the chosen solver method. The
            scipy documentation for `scipy.optimize.root
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html>`__
            details the possible values for this parameter. 

        cr0: np.ndarray, size (rank*rank*length)
            The reference direct correlation functions
        
        hk0: np.ndarray, size (rank*rank*length)
            The reference total correlation functions
        
        '''
        if guess is None:
            guess = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)

        if cr0 is None:
            cr0 = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)

        if hk0 is None:
            hk0 = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)

        if hk_initial is None:
            hk_initial = hk0

        self.totalCorr.data = np.copy(hk_initial.reshape((-1,self.sys.rank,self.sys.rank)))
            
        if options is None:
            options = {'disp':True}

        if tol is None:
            tol = 1e-5
        
        warnstr = 'For Reference Molecular Percus Yevick closure (RMPY) use caution if using solve. solve_picard may provide better convergence.'
        for (i,j),(t1,t2),closure in self.sys.closure.iterpairs():
            if isinstance(closure,MolecularClosure):
                if closure.name == "RMPY":
                    warnings.warn(warnstr)

        self.minimize_result = root(self.cost,guess,args=(cr0,hk0),method=method,options=options)
        
        if self.totalCorr.space == Space.Fourier:
            self.sys.domain.MatrixArray_to_real(self.totalCorr)

        #tol = 1e-5
        warnstr = 'Pair correlations are negative (value = {:3.2e}) for {}-{} pair!'
        for i,(t1,t2),H in self.totalCorr.iterpairs():
            if np.any(H<-(1.0+tol)):
                val = np.min(H)
                warnings.warn(warnstr.format(val,t1,t2))
        
        return self.minimize_result
