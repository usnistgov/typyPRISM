#!python
from __future__ import division,print_function
from typyPRISM.core.Space import Space
import numpy as np
from scipy.fftpack import dst

class Domain(object):
    '''Define and transform between Real and Fourier space
    
    Domain describes the discretization of Real and Fourier space
    and also sets up the functions and coefficients for transforming
    data between them.
    
    Attributes
    ----------
    length: int
        Number of gridpoints in Real and Fourier space grid
        
    dr,dk: float
        Grid spacing in Real and Fourier space
    
    r,k: float ndarray
        Numpy arrays of grid locations in Real and Fourier space

    long_r: float ndarray
        Numpy array identical in content to r except reshaped so that it 
        broadcasts correctly when multiplying MatrixArrays
    
    DST_II_coeffs,DST_III_coeffs: float
        Coefficients needed for Discrete Sine Transforms. Note that these
        values are specific to each implementation of the DST and were 
        derived for (Scipy's interface to) FFTPACK. 
    '''
    def __init__(self,length,dr=None,dk=None):
        self._length = length
        
        if (dr is None) and (dk is None):
            raise ValueError('Real or Fourier grid spacing must be specified')
            
        elif (dr is not None) and (dk is not None):
            raise ValueError('Cannot specify **both** Real and Fourier grid spacings independently.')
            
        elif dr is not None:
            self.dr = dr #dk is set in property setter
            
        elif dk is not None:
            self.dk = dk #dr is set in property setter
            
            
        self.build_grid() #build grid should have been called already but we'll be safe
    
    def build_grid(self):
        '''Construct the Real and Fourier Space grids and transform coefficients'''
        self.r = np.arange(self._dr,self._dr*(self._length+1),self._dr)
        self.k = np.arange(self.dk,self.dk*(self._length+1),self.dk)
        self.DST_II_coeffs = 2.0*np.pi *self.r*self._dr 
        self.DST_III_coeffs = self.k * self.dk/(4.0*np.pi*np.pi)
        self.long_r = self.r.reshape((-1,1,1))
    
    @property
    def dr(self):
        return self._dr
    @dr.setter
    def dr(self,value):
        self._dr = value
        self._dk = np.pi/(self._dr*self._length)
        self.build_grid()#need to re-build grid since spacing has changed
    
    @property
    def dk(self):
        return self._dk
    @dk.setter
    def dk(self,value):
        self._dk = value
        self._dr = np.pi/(self._dk*self._length)
        self.build_grid()#need to re-build grid since spacing has changed
        
    @property
    def length(self):
        return self._length
    @length.setter
    def length(self,value):
        self._length = value
        self.build_grid()#need to re-build grid since length has changed
        
    def __repr__(self):
        return '<Domain length:{} dr/rmax:{:4.3f}/{:3.1f} dk/kmax:{:4.3f}/{:3.1f}>'.format(self.length,self.dr,self.r[-1],self.dk,self.k[-1])
    
    def to_fourier(self,array):
        ''' Discrete Sine Transform of a numpy array 
        
        Peforms a Real-to-Real Discrete Sine Transform  of type II 
        on a numpy array of non-complex values. For radial data that is 
        symmetric in \phi and \theta, this is **a** correct transform
        to go from Real-space to Fourier-space. 
        
        Parameters
        ----------
        array: float ndarray
            Real-space data to be transformed
            
        Returns
        -------
        array: float ndarray
            data transformed to fourier space
        
        '''
        return dst(self.DST_II_coeffs*array,type=2)/self.k
    
    def to_real(self,array):
        ''' Discrete Sine Transform of a numpy array 
        
        Peforms a Real-to-Real Discrete Sine Transform  of type III 
        on a numpy array of non-complex values. For radial data that is 
        symmetric in \phi and \theta, this is **a** correct transform
        to go from Fourier-space to Real space.
        
        Parameters
        ----------
        array: float ndarray
            Fourier-space data to be transformed
            
        Returns
        -------
        array: float ndarray
            data transformed to Real space
        
        '''
        return dst(self.DST_III_coeffs*array,type=3)/self.r
    
    def MatrixArray_to_fourier(self,marray):
        ''' Transform all columns of a MatrixArray to Fourier space in-place'''
        if marray.space == Space.Fourier:
            raise ValueError('MatrixArray is marked as already in Fourier space')
            
        for (i,j),(t1,t2),column in marray.itercolumn():
            marray[t1,t2] = self.to_fourier(column)
        
        marray.space = Space.Fourier
            
    def MatrixArray_to_real(self,marray):
        ''' Transform all columns of a MatrixArray to Real space in-place '''
        if marray.space == Space.Real:
            raise ValueError('MatrixArray is marked as already in Real space')
            
        for (i,j),(t1,t2),column in marray.itercolumn():
            marray[t1,t2] = self.to_real(column)
            
        marray.space = Space.Real
            